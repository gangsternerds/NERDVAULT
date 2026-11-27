import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest


BENCH = Path(__file__).with_name('bench2.py')

WRAPPER = r'''
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('root')
parser.add_argument('--ignore-term', action='store_true')
parser.add_argument('--keep-leader', action='store_true')
parser.add_argument('--host')
parser.add_argument('--port')
args = parser.parse_args()
root = Path(args.root)
number = len(list(root.glob('child.*.pid')))
if number:
    pid = int((root / 'child.0.pid').read_text())
    for _ in range(200):
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            running = stat[0] != 'Z'
            rss_kb = int(stat[21]) * os.sysconf('SC_PAGE_SIZE') // 1024
        except (FileNotFoundError, ProcessLookupError):
            running, rss_kb = False, 0
        if not running:
            break
        time.sleep(.005)
    (root / 'replacement.json').write_text(json.dumps({
        'previous_running': running, 'previous_rss_kb': rss_kb,
    }))
child = subprocess.Popen([
    sys.executable, str(root / 'server.py'), args.port,
    str(os.getpid()), str(number), str(int(args.ignore_term)),
    str(int(args.keep_leader)),
])
(root / f'child.{number}.pid').write_text(str(child.pid))
(root / f'leader.{number}.pid').write_text(str(os.getpid()))
child.wait()
'''

SERVER = r'''
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import signal
import sys
import time

port, leader, number, ignore_term, keep_leader = map(int, sys.argv[1:])
if ignore_term:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

class Handler(BaseHTTPRequestHandler):
    measured_requests = 0

    def log_message(self, *args):
        pass

    def reply(self, body):
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        self.reply({'status': 'ok'})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/tokenize':
            self.reply({'tokens': [1]})
            return
        measured = body['messages'][0]['content'] != 'Say hi.'
        first_request = type(self).measured_requests == 0
        if measured:
            type(self).measured_requests += 1
        if number == 0 and measured and first_request:
            # The launcher dies after readiness; the actual server survives.
            if not keep_leader:
                os.kill(leader, signal.SIGKILL)
                for _ in range(200):
                    try:
                        state = Path(f'/proc/{leader}/stat').read_text().rsplit(')', 1)[1].split()[0]
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    if state == 'Z':
                        break
                    time.sleep(.005)
            self.send_error(503, 'launcher exited during request')
            return
        self.reply({
            'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}],
            'timings': {'predicted_n': 1, 'predicted_per_second': 10},
        })

HTTPServer(('127.0.0.1', port), Handler).serve_forever()
'''


@unittest.skipUnless(sys.platform.startswith('linux'), 'uses /proc process state')
class ServerRestartTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.addCleanup(self.cleanup_processes)
        (self.root / 'wrapper.py').write_text(WRAPPER)
        (self.root / 'server.py').write_text(SERVER)

    def cleanup_processes(self):
        for path in self.root.glob('leader.*.pid'):
            try:
                os.killpg(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def assert_stopped(self, pid):
        for _ in range(200):
            try:
                state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):
                return
            if state == 'Z':
                return
            time.sleep(.005)
        self.fail(f'owned server {pid} is still running')

    def run_restart(self, ignore_term=False, keep_leader=False):
        command = [sys.executable, str(self.root / 'wrapper.py'), str(self.root)]
        if ignore_term:
            command.append('--ignore-term')
        if keep_leader:
            command.append('--keep-leader')
        arms = self.root / 'arms.json'
        arms.write_text(json.dumps({'fixture': {'cmd': shlex.join(command)}}))
        prompts = self.root / 'prompts.jsonl'
        prompts.write_text(''.join(json.dumps({
            'id': f'prompt-{i}', 'task': 'test',
            'messages': [{'role': 'user', 'content': f'prompt {i}'}],
        }) + '\n' for i in range(2)))
        output = self.root / 'results.jsonl'
        run = subprocess.run([
            sys.executable, str(BENCH), '--arms', str(arms),
            '--prompts', str(prompts), '--out', str(output), '--block', '2',
        ], capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('BENCH_DONE', run.stderr)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertIn('503', rows[0]['error'])
        self.assertNotIn('error', rows[1])
        self.assertEqual(rows[1]['predicted_n'], 1)
        self.assertEqual(len(list(self.root.glob('child.*.pid'))), 1 if keep_leader else 2)
        if not keep_leader:
            replacement = json.loads((self.root / 'replacement.json').read_text())
            self.assertFalse(replacement['previous_running'], replacement)
        for path in self.root.glob('child.*.pid'):
            self.assert_stopped(int(path.read_text()))

    def test_restart_stops_previous_server_before_launch(self):
        self.run_restart()

    def test_restart_kills_previous_server_ignoring_term(self):
        self.run_restart(ignore_term=True)

    def test_request_error_keeps_live_launcher_for_next_request(self):
        self.run_restart(keep_leader=True)


if __name__ == '__main__':
    unittest.main()
