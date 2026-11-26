import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = 'fpu sse sse2 ssse3 sse4_1 sse4_2 popcnt avx'
AVX2 = BASE + ' fma f16c avx2 bmi2'
CPUS = {
    'icelake': AVX2 + ' avx512f avx512dq avx512bw avx512vl avx512vbmi avx512_vnni',
    'cascadelake': AVX2 + ' avx512f avx512dq avx512bw avx512vl avx512_vnni',
    'skylakex': AVX2 + ' avx512f avx512dq avx512bw avx512vl',
    'alderlake': AVX2 + ' avx_vnni',
    'zen3': AVX2,
    'avx2-no-fma': BASE + ' avx2',
    'sandybridge': BASE,
}


class CpuPaths(unittest.TestCase):
    """install.sh --check and serve read the CPU flags from /proc/cpuinfo; a grep in front of the real one answers
    for that file, so every CPU class can be checked on any machine."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.tools = self.work / 'tools'
        self.tools.mkdir()
        real = shutil.which('grep')
        grep = self.tools / 'grep'
        grep.write_text('#!/bin/bash\n'
                        'for last; do :; done\n'
                        'if [ "$last" = /proc/cpuinfo ]; then\n'
                        '  case " $* " in *"model name"*) echo "model name : Fixture CPU" ;; *) echo "flags : $FIXTURE_FLAGS" ;; esac\n'
                        '  exit 0\n'
                        'fi\n'
                        f'exec {real} "$@"\n')
        grep.chmod(0o755)
        uname = self.tools / 'uname'
        uname.write_text('#!/bin/sh\ncase "$1" in -m) echo x86_64 ;; -s) echo Linux ;; *) echo Linux ;; esac\n')
        uname.chmod(0o755)
        self.prefix = self.work / 'home'
        (self.prefix / 'bin').mkdir(parents=True)
        (self.prefix / 'models').mkdir()
        (self.prefix / 'models/model.gguf').touch()
        server = self.prefix / 'bin/llama-server'
        server.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        server.chmod(0o755)

    def env(self, cpu):
        return {**os.environ, 'OPENSET_LLM_HOME': str(self.prefix), 'FIXTURE_FLAGS': CPUS[cpu],
                'PATH': str(self.tools) + os.pathsep + os.environ['PATH']}

    def check(self, cpu):
        result = subprocess.run(['bash', str(ROOT / 'install.sh'), '--check'], env=self.env(cpu),
                                text=True, capture_output=True, timeout=15)
        return result.stdout + result.stderr

    def serve(self, cpu, *args):
        return subprocess.run(['bash', str(ROOT / 'serve'), '--no-resume', *args], env=self.env(cpu),
                              text=True, capture_output=True, timeout=15)

    def test_check_names_the_path_of_every_cpu_class(self):
        self.assertIn('full fast path', self.check('icelake'))
        self.assertIn('partial fast path', self.check('cascadelake'))
        for cpu in ('alderlake', 'zen3', 'skylakex'):
            out = self.check(cpu)
            self.assertIn('256-bit kernels', out, cpu)
            self.assertNotIn("fork's own", out, cpu)
        self.assertIn('AVX-VNNI', self.check('alderlake'))
        self.assertNotIn('AVX-VNNI', self.check('zen3'))
        self.assertIn('without VNNI', self.check('skylakex'))
        for cpu in ('avx2-no-fma', 'sandybridge'):
            out = self.check(cpu)
            self.assertNotIn('256-bit kernels', out, cpu)
            self.assertIn('slowly', out, cpu)

    def test_serve_keeps_speculation_on_only_with_avx512_vnni(self):
        for cpu, on in (('icelake', True), ('cascadelake', True), ('alderlake', False), ('zen3', False), ('skylakex', False)):
            result = self.serve(cpu)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertEqual('--spec-type' in args, on, cpu)
            if not on:
                self.assertIn('speculation off', result.stderr, cpu)
                self.assertIn('--spec', result.stderr, cpu)

    def draft_depth(self, cpu, *args):
        result = self.serve(cpu, *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertIn('--spec-type', argv)
        return argv[argv.index('--spec-draft-n-max') + 1]

    def test_serve_spec_flag_forces_speculation_without_avx512(self):
        self.assertEqual(self.draft_depth('zen3', '--spec'), '2')
        self.assertEqual(self.draft_depth('alderlake', '--spec'), '2')
        self.assertEqual(self.draft_depth('zen3', '--spec', '--spec-depth', '3'), '3')

    def test_draft_depth_with_avx512_is_unchanged(self):
        self.assertEqual(self.draft_depth('icelake'), '3')
        self.assertEqual(self.draft_depth('icelake', '--spec'), '3')

    def test_compact_attention_caches_still_need_avx512(self):
        for flag in ('--kv6', '--kv5'):
            result = self.serve('alderlake', flag)
            self.assertEqual(result.returncode, 2, flag)
            self.assertIn('--kv8', result.stderr)
        self.assertEqual(self.serve('alderlake', '--kv8').returncode, 0)


if __name__ == '__main__':
    unittest.main()
