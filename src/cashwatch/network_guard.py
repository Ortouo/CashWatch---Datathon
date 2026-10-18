"""Process-local Linux IPv4/IPv6 socket denial for the trusted inference worker.

This is a narrow syscall filter, not a hostile-code sandbox or a host firewall.
AF_UNIX remains available. Do not expose a network proxy through a UNIX socket to
the inference process. No filter is installed merely by importing this module.

Primary interface references:
https://github.com/seccomp/libseccomp/blob/main/include/seccomp.h.in
https://man7.org/linux/man-pages/man3/seccomp_rule_add.3.html
https://man7.org/linux/man-pages/man3/seccomp_attr_set.3.html
https://docs.kernel.org/userspace-api/seccomp_filter.html
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import errno
import json
import os
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path

_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO_EPERM = 0x00050000 | errno.EPERM
_SCMP_FLTATR_CTL_NNP = 3
_SCMP_FLTATR_CTL_TSYNC = 4
_SCMP_CMP_EQ = 4
_installation_lock = threading.Lock()
_installed_report: dict | None = None


class NetworkIsolationError(RuntimeError):
    """Linux must not proceed to inference if the required guard cannot be applied."""


class _ArgComparison(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint), ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64),
    ]


def _load_libseccomp():
    name = ctypes.util.find_library("seccomp")
    if not name:
        raise NetworkIsolationError("libseccomp is unavailable; local Linux inference is blocked.")
    try:
        library = ctypes.CDLL(name, use_errno=True)
        library.seccomp_init.argtypes = [ctypes.c_uint32]
        library.seccomp_init.restype = ctypes.c_void_p
        library.seccomp_release.argtypes = [ctypes.c_void_p]
        library.seccomp_release.restype = None
        library.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
        library.seccomp_attr_set.restype = ctypes.c_int
        library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        library.seccomp_syscall_resolve_name.restype = ctypes.c_int
        library.seccomp_rule_add_array.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
            ctypes.POINTER(_ArgComparison),
        ]
        library.seccomp_rule_add_array.restype = ctypes.c_int
        library.seccomp_load.argtypes = [ctypes.c_void_p]
        library.seccomp_load.restype = ctypes.c_int
    except (OSError, AttributeError) as exc:
        raise NetworkIsolationError("The installed libseccomp cannot provide the required interface.") from exc
    return library


def _inspect_inherited_sockets() -> dict:
    """Inspect only this process's FDs, with no network I/O or endpoint reporting."""
    try:
        descriptors = os.listdir("/proc/self/fd")
    except OSError as exc:
        raise NetworkIsolationError("Cannot verify inherited descriptors through /proc/self/fd.") from exc
    inet_count = 0
    unix_count = 0
    for name in descriptors:
        if not name.isdigit():
            continue
        descriptor = int(name)
        try:
            if not stat.S_ISSOCK(os.fstat(descriptor).st_mode):
                continue
            duplicate = os.dup(descriptor)
            try:
                wrapper = socket.socket(fileno=duplicate)
            except BaseException:
                os.close(duplicate)
                raise
            with wrapper:
                if wrapper.family in {socket.AF_INET, socket.AF_INET6}:
                    inet_count += 1
                elif wrapper.family == socket.AF_UNIX:
                    unix_count += 1
        except OSError as exc:
            if exc.errno in {errno.EBADF, errno.ENOENT}:
                # /proc/self/fd enumeration itself uses a short-lived directory FD.
                continue
            raise NetworkIsolationError("An inherited socket descriptor could not be inspected.") from exc
    if inet_count:
        raise NetworkIsolationError("Existing IPv4/IPv6 sockets are not allowed in an inference worker.")
    return {"inherited_internet_fd_count": 0, "inherited_unix_fd_count": unix_count}


def _require_zero(code: int, stage: str) -> None:
    if code != 0:
        raise NetworkIsolationError(f"Network guard failed at {stage} (code {code}); inference is blocked.")


def _socket_checks() -> dict:
    checks = {}
    for family, label in ((socket.AF_INET, "ipv4"), (socket.AF_INET6, "ipv6")):
        try:
            opened = socket.socket(family, socket.SOCK_STREAM)
        except OSError as exc:
            checks[f"{label}_socket_errno"] = exc.errno
            checks[f"{label}_socket_denied"] = exc.errno == errno.EPERM
        else:
            opened.close()
            checks[f"{label}_socket_errno"] = None
            checks[f"{label}_socket_denied"] = False
    try:
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with left, right:
            left.sendall(b"ok")
            checks["unix_socketpair_works"] = right.recv(2) == b"ok"
    except OSError as exc:
        checks["unix_socketpair_works"] = False
        checks["unix_socket_errno"] = exc.errno
    return checks


def enforce_network_guard() -> dict:
    """Install and verify in an isolated worker; Linux failure raises before ML work.

    Linux's TSYNC applies the filter to existing threads and children inherit it.
    The filter cannot be removed by this API. Never call this in the HTTP process.
    macOS reports explicitly that this OS-level mechanism is unavailable.
    """
    global _installed_report
    if sys.platform != "linux":
        return {
            "os_enforced": False, "backend": None,
            "reason": "macos_requires_separate_network_isolation" if sys.platform == "darwin" else "unsupported_os",
            "scope": "No OS network filter installed by this module on this platform.",
        }
    with _installation_lock:
        if _installed_report is not None:
            return dict(_installed_report)
        inherited = _inspect_inherited_sockets()
        library = _load_libseccomp()
        context = library.seccomp_init(_SCMP_ACT_ALLOW)
        if not context:
            raise NetworkIsolationError("Could not initialize the Linux network guard.")
        try:
            _require_zero(library.seccomp_attr_set(context, _SCMP_FLTATR_CTL_NNP, 1), "no_new_privs")
            _require_zero(library.seccomp_attr_set(context, _SCMP_FLTATR_CTL_TSYNC, 1), "thread_sync")
            for syscall in (b"socket", b"socketpair"):
                number = library.seccomp_syscall_resolve_name(syscall)
                if number < 0:
                    raise NetworkIsolationError("The native socket syscall could not be resolved.")
                for family in (socket.AF_INET, socket.AF_INET6):
                    comparison = _ArgComparison(0, _SCMP_CMP_EQ, int(family), 0)
                    _require_zero(library.seccomp_rule_add_array(
                        context, _SCMP_ACT_ERRNO_EPERM, number, 1, ctypes.byref(comparison),
                    ), "socket_family_rule")
            _require_zero(library.seccomp_load(context), "filter_load")
        finally:
            library.seccomp_release(context)
        checks = _socket_checks()
        if not all(checks.get(key) for key in ("ipv4_socket_denied", "ipv6_socket_denied", "unix_socketpair_works")):
            raise NetworkIsolationError("The installed network guard failed its socket checks; inference is blocked.")
        _installed_report = {
            "os_enforced": True, "backend": "linux_libseccomp",
            "blocked_socket_families": ["AF_INET", "AF_INET6"],
            "filtered_syscalls": ["socket", "socketpair"],
            "thread_sync": True, "no_new_privs": True, **inherited, **checks,
            "scope": "Deny ordinary IPv4/IPv6 socket creation in this worker and descendants; allow AF_UNIX.",
            "is_host_firewall": False, "is_hostile_code_sandbox": False,
        }
        return dict(_installed_report)


def check_in_subprocess(timeout_sec: float = 10.0) -> dict:
    """The caller keeps its networking; only the disposable check child is filtered."""
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--_child-check"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=timeout_sec, close_fds=True, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "os_enforced": False, "error_code": "guard_check_failed", "error_class": type(exc).__name__}
    try:
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            raise ValueError("Invalid check report")
    except (ValueError, TypeError):
        return {"ok": False, "os_enforced": False, "error_code": "guard_check_invalid_result"}
    result["child_exit_code"] = completed.returncode
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Test in a disposable child process without network traffic.")
    parser.add_argument("--_child-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._child_check:
        try:
            report = enforce_network_guard()
            report["ok"] = report["os_enforced"]
        except NetworkIsolationError as exc:
            report = {
                "ok": False, "os_enforced": False, "error_code": "network_isolation_unavailable",
                "error_class": type(exc).__name__, "reason": str(exc),
            }
    elif args.check:
        report = check_in_subprocess()
    else:
        parser.error("Use --check; importing the module never installs a filter")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
