import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import unittest


class QuickBenchmarkTest(unittest.TestCase):
    def run_quick(self, failure, plain=False, spec=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bench").mkdir()
            (root / "commands").mkdir()
            (root / "home/bin").mkdir(parents=True)
            shutil.copy(Path(__file__).with_name("quick.sh"), root / "bench/quick.sh")

            def script(path, text):
                path.write_text("#!/bin/bash\n" + text)
                path.chmod(0o755)

            script(root / "home/bin/llama-server", "exit 0\n")
            script(root / "commands/curl", 'if [ ! -e "$TEST_ROOT/probed" ]; then touch "$TEST_ROOT/probed"; exit 1; fi\nwhile [ ! -s "$TEST_ROOT/server.pid" ]; do sleep 0.01; done\nexit 0\n')
            # The measurement step retires the pid of the server it used, so the readiness probe of the next
            # arm waits for that arm's own server. With a stale pid the probe returned at once and quick.sh
            # could signal a server that had not finished starting, which bash can lose.
            script(root / "commands/python3", 'cat "$TEST_ROOT/server.pid" >> "$TEST_ROOT/server.pids"\nrm "$TEST_ROOT/server.pid"\n'
                   + ("exit 42\n" if failure else "echo '| synthetic result |'\n"))
            script(root / "commands/lscpu", "printf 'Core(s) per socket: 1\\nSocket(s): 1\\n'\n")
            script(root / "serve", 'echo $$ > "$TEST_ROOT/server.pid.new"\nmv "$TEST_ROOT/server.pid.new" "$TEST_ROOT/server.pid"\necho "serving on test (synthetic mode)"\nexec sleep 60\n')
            env = dict(os.environ, TEST_ROOT=str(root), TMPDIR=str(root), OPENSET_LLM_HOME=str(root / "home"),
                       PATH=str(root / "commands") + os.pathsep + os.environ["PATH"])
            proc = subprocess.Popen(["bash", str(root / "bench/quick.sh")] + (["--plain"] if plain else []) + (["--spec"] if spec else []),
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, start_new_session=True)
            try:
                stdout, stderr = proc.communicate(timeout=10)
                self.assertTrue((root / "server.pids").exists(), stderr)
                pids = [int(line) for line in (root / "server.pids").read_text().split()]
                self.assertEqual(len(pids), 1 + plain + spec, stderr)
                for pid in pids:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                return proc.returncode, stdout, stderr
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()

    def test_failed_measurement_fails_without_a_success_report(self):
        code, stdout, _ = self.run_quick(True)
        self.assertNotEqual(code, 0)
        self.assertNotIn("openset-llm quick check", stdout)

    def test_two_successful_arms_keep_both_rows(self):
        code, stdout, stderr = self.run_quick(False, plain=True)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stdout.count("| synthetic result |"), 2)

    def test_speculation_arm_adds_a_third_row(self):
        code, stdout, stderr = self.run_quick(False, plain=True, spec=True)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stdout.count("| synthetic result |"), 3)


if __name__ == "__main__":
    unittest.main()
