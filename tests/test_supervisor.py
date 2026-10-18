import os
import subprocess
import sys
import time

from cashwatch.jobs import kill_process_tree


def test_hard_stop_terminates_running_worker():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=(os.name == "posix"))
    started = time.monotonic()
    kill_process_tree(process)
    assert process.poll() is not None
    assert time.monotonic() - started < 5

