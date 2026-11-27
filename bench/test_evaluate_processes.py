import os
from pathlib import Path
import signal
import tempfile
import time
import unittest

from evaluate import run


class EvaluationProcessTest(unittest.TestCase):
    def test_timeout_stops_child_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "child.pid"
            source = f'''
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with open({str(pidfile)!r}, "w") as f:
    f.write(str(child.pid))
time.sleep(60)
'''
            pid = None
            try:
                self.assertEqual(run(source), (False, "timeout"))
                pid = int(pidfile.read_text())
                for _ in range(100):
                    try:
                        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                    except (FileNotFoundError, ProcessLookupError):  # the process can vanish while its stat file is being read
                        state = "gone"
                    if state in ("gone", "Z"):
                        break
                    time.sleep(0.01)
                self.assertIn(state, ("gone", "Z"), "timed-out evaluation left a running child")
            finally:
                if pid is None and pidfile.exists():
                    pid = int(pidfile.read_text())
                if pid is not None:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_successful_evaluation_still_passes(self):
        self.assertEqual(run("assert 2 + 2 == 4"), (True, ""))

    def test_failure_keeps_the_last_error_line(self):
        self.assertEqual(run("raise ValueError('example')"), (False, "ValueError: example"))


if __name__ == "__main__":
    unittest.main()
