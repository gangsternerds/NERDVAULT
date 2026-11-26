from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ['-O2', '-mavx2', '-mfma', '-mf16c']
# name -> (extra compiler flags, CPU flags the machine needs to run the result)
VARIANTS = {
    'avx2': ([], ()),                                   # the vpmaddubsw dot: any AVX2 CPU
    'avx2-forced': (['-DPTQ1A_FORCE_MADD'], ()),        # same dot with the VNNI detection overridden
    'avx-vnni': (['-mavxvnni'], ('avx_vnni',)),         # VEX vpdpbusd: Intel Core 12th gen and newer, Zen 5
    'avx512-vnni-256': (['-mavx512f', '-mavx512bw', '-mavx512vl', '-mavx512vnni'],
                        ('avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni')),   # the same kernels in an AVX-512 build
}


def cpu_flags():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('flags'):
                return set(line.split(':', 1)[1].split())
    except OSError:
        pass
    return set()


class Avx2Kernel(unittest.TestCase):
    def test_kernels_match_the_scalar_reference(self):
        compiler = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
        flags = cpu_flags()
        if platform.machine() not in ('x86_64', 'AMD64') or not compiler:
            self.skipTest('needs an x86-64 C compiler')
        if not {'avx2', 'fma', 'f16c'} <= flags:
            self.skipTest('this CPU has no AVX2 + FMA + F16C')
        ran = []
        with tempfile.TemporaryDirectory() as tmp:
            for name, (extra, needs) in VARIANTS.items():
                if not set(needs) <= flags:
                    continue
                exe = Path(tmp) / f'kernel-{name}'
                build = subprocess.run([compiler, *BASE, *extra, '-I', str(ROOT / 'kernels'),
                                        str(ROOT / 'tests/kernel_avx2.c'), '-o', str(exe), '-lm'],
                                       capture_output=True, text=True)
                self.assertEqual(build.returncode, 0, f'{name}: {build.stderr}')
                run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300)
                self.assertEqual(run.returncode, 0, f'{name}: {run.stdout}{run.stderr}')
                self.assertIn('failures=0', run.stdout, name)
                want = 'vpmaddubsw' if name.startswith('avx2') else 'vpdpbusd'
                self.assertIn(f'dot={want}', run.stdout, name)
                ran.append(name)
        self.assertIn('avx2', ran)


if __name__ == '__main__':
    unittest.main()
