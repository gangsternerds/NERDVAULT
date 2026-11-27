import io
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import bench2


class ServerProcessTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.children = []
        self.leaders = []
        self.addCleanup(self.cleanup_processes)
        self.wrapper = self.root / 'wrapper.py'
        self.wrapper.write_text('''import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(120)
''')

    def cleanup_processes(self):
        for pid in self.children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for proc in self.leaders:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def launch(self, health_error=None):
        pidfile = self.root / 'child.pid'
        popen = subprocess.Popen

        def spawn(*args, **kwargs):
            proc = popen(*args, **kwargs)
            self.leaders.append(proc)
            for _ in range(200):
                if pidfile.exists():
                    self.children.append(int(pidfile.read_text()))
                    return proc
                time.sleep(.005)
            self.fail('wrapper did not start')

        response = io.BytesIO()
        response.status = 200
        with patch.object(bench2, 'free_port', return_value=12345), \
             patch.object(bench2.subprocess, 'Popen', side_effect=spawn), \
             patch.object(bench2.urllib.request, 'urlopen', side_effect=health_error, return_value=response):
            cmd = shlex.join([sys.executable, str(self.wrapper), str(pidfile)])
            return bench2.start('fixture', {'cmd': cmd}, str(self.root), 'test')

    def assert_stopped(self, pid):
        for _ in range(200):
            try:
                state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):  # the process can vanish while its stat file is being read
                return
            if state == 'Z':
                return
            time.sleep(.005)
        self.fail(f'owned child {pid} is still running')

    def test_stop_reaps_wrapper_and_stops_only_its_children(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],
                                     start_new_session=True)
        self.leaders.append(unrelated)
        _, proc, _ = self.launch()
        bench2.stop(proc)
        self.assertIsNotNone(proc.returncode)
        self.assert_stopped(self.children[0])
        self.assertIsNone(unrelated.poll())

    def test_readiness_interruption_stops_group_and_reaps_leader(self):
        with self.assertRaises(KeyboardInterrupt):
            self.launch(KeyboardInterrupt())
        self.assertIsNotNone(self.leaders[0].returncode)
        self.assert_stopped(self.children[0])

    def test_readiness_timeout_stops_group_and_reaps_leader(self):
        with patch.object(bench2.time, 'time', side_effect=[0, 1201]):
            with self.assertRaisesRegex(SystemExit, 'not healthy'):
                self.launch()
        self.assertIsNotNone(self.leaders[0].returncode)
        self.assert_stopped(self.children[0])


if __name__ == '__main__':
    unittest.main()
