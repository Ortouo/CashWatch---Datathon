"""No external connections: test the Linux socket filter in disposable children."""
from __future__ import annotations

import ctypes.util
import errno
import json
import os
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from cashwatch import network_guard

ROOT = Path(__file__).resolve().parents[1]
CHILD_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def socket_outcome(family):
    try:
        connection = socket.socket(family, socket.SOCK_STREAM)
    except OSError as exc:
        return exc.errno
    connection.close()
    return 0


CAN_TEST_LINUX_FILTER = sys.platform == "linux" and socket_outcome(socket.AF_UNIX) == 0


def child_json(program):
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True,
        env=CHILD_ENV, close_fds=True, timeout=10, check=False,
    )
    if result.returncode:
        raise AssertionError(f"Local guard child failed with exit code {result.returncode}: {result.stderr}")
    return json.loads(result.stdout)


class NetworkGuardTests(unittest.TestCase):
    def test_import_has_no_network_side_effect(self):
        result = child_json("""
import json, socket
def outcomes():
    values = []
    for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
        try:
            item = socket.socket(family, socket.SOCK_STREAM)
        except OSError as exc:
            values.append(exc.errno)
        else:
            item.close()
            values.append(0)
    return values
before = outcomes()
import cashwatch.network_guard
print(json.dumps({'before': before, 'after': outcomes()}))
""")
        self.assertEqual(result["before"], result["after"])

    @unittest.skipUnless(CAN_TEST_LINUX_FILTER, "Requires Linux without an outer AF_UNIX denial")
    def test_real_child_denies_internet_allows_unix_and_leaves_parent_unchanged(self):
        before = [socket_outcome(family) for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX)]
        result = network_guard.check_in_subprocess()
        after = [socket_outcome(family) for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX)]
        self.assertEqual(before, after)
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["os_enforced"])
        self.assertEqual(result["ipv4_socket_errno"], errno.EPERM)
        self.assertEqual(result["ipv6_socket_errno"], errno.EPERM)
        self.assertTrue(result["unix_socketpair_works"])
        self.assertEqual(result["inherited_internet_fd_count"], 0)
        self.assertFalse(result["is_hostile_code_sandbox"])

    @unittest.skipUnless(CAN_TEST_LINUX_FILTER, "Requires Linux without an outer AF_UNIX denial")
    def test_preexisting_thread_and_exec_child_inherit_filter(self):
        result = child_json("""
import json, socket, subprocess, sys, threading
from pathlib import Path
from cashwatch.network_guard import enforce_network_guard
ready = threading.Event()
start = threading.Event()
thread_result = {}
def attempt():
    ready.set()
    start.wait()
    try:
        created = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        thread_result['errno'] = exc.errno
    else:
        created.close()
        thread_result['errno'] = 0
worker = threading.Thread(target=attempt)
worker.start()
ready.wait()
guard = enforce_network_guard()
start.set()
worker.join()
probe = subprocess.run([sys.executable, '-c', 'import socket; socket.socket(socket.AF_INET, socket.SOCK_STREAM)'], capture_output=True, text=True, close_fds=True)
status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
print(json.dumps({'thread_errno': thread_result['errno'], 'exec_child_failed': probe.returncode != 0, 'exec_child_permission_error': 'PermissionError' in probe.stderr, 'no_new_privs': status['NoNewPrivs'].strip(), 'guard': guard['os_enforced']}))
""")
        self.assertEqual(result["thread_errno"], errno.EPERM)
        self.assertTrue(result["exec_child_failed"])
        self.assertTrue(result["exec_child_permission_error"])
        self.assertEqual(result["no_new_privs"], "1")
        self.assertTrue(result["guard"])

    @unittest.skipUnless(CAN_TEST_LINUX_FILTER and socket_outcome(socket.AF_INET) == 0, "Requires permission to create an unconnected inherited test socket")
    def test_existing_internet_fd_blocks_before_inference(self):
        result = child_json("""
import json, socket
from cashwatch.network_guard import enforce_network_guard, NetworkIsolationError
opened = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    enforce_network_guard()
except NetworkIsolationError:
    print(json.dumps({'blocked': True}))
else:
    print(json.dumps({'blocked': False}))
finally:
    opened.close()
""")
        self.assertTrue(result["blocked"])

    def test_missing_linux_library_is_a_hard_error(self):
        with patch.object(network_guard.sys, "platform", "linux"), \
             patch.object(network_guard, "_installed_report", None), \
             patch.object(network_guard, "_inspect_inherited_sockets", return_value={}), \
             patch.object(network_guard.ctypes.util, "find_library", return_value=None):
            with self.assertRaises(network_guard.NetworkIsolationError):
                network_guard.enforce_network_guard()

    def test_macos_explicitly_reports_no_os_enforcement(self):
        with patch.object(network_guard.sys, "platform", "darwin"), \
             patch.object(network_guard, "_load_libseccomp") as loader:
            report = network_guard.enforce_network_guard()
        loader.assert_not_called()
        self.assertFalse(report["os_enforced"])
        self.assertEqual(report["reason"], "macos_requires_separate_network_isolation")

    def test_worker_stops_before_analyzer_when_guard_fails(self):
        from cashwatch import worker
        with patch.object(worker, "enforce_network_guard", side_effect=network_guard.NetworkIsolationError("test")), \
             patch.object(worker, "LocalVideoAnalyzer") as analyzer:
            with self.assertRaises(network_guard.NetworkIsolationError):
                worker.run({})
        analyzer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
