from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def table(path, name):
    """The rows of `static const uint8_t <name>[R][C] ... = { {..}, .. };` in a kernel source."""
    text = (ROOT / path).read_text()
    match = re.search(r'static const uint8_t ' + name + r'\[(\d+)\]\[(\d+)\][^=]*=\s*\{(.*?)\n\};', text, re.S)
    assert match, f'{name} not found in {path}'
    rows = [[int(v) for v in re.findall(r'\d+', row)] for row in re.findall(r'\{([^{}]*)\}', match.group(3))]
    assert len(rows) == int(match.group(1)) and all(len(r) == int(match.group(2)) for r in rows)
    return rows


class KernelTables(unittest.TestCase):
    def test_avx512_decode_tables(self):
        rows = table('kernels/ptq1-vbmi.inc', 'ptq1v_tab')
        for i in range(64):
            j = i % 8
            want = [5 * j, 40 + 5 * j if j < 4 else 60, 60 if j < 4 else 40 + 5 * (j - 4)]
            want += [((i & 31) // p) % 3 if (i & 31) < 27 else 0 for p in (1, 3, 9)]
            want += [((i & 15) // p) % 3 if (i & 15) < 9 else 0 for p in (1, 3)]
            want += [8 * j + 4 if i < 8 else 64 + 8 * j, 64 + 8 * j + 4, 64 + 8 * j]
            self.assertEqual([row[i] for row in rows], want, i)

    def test_avx2_decode_tables(self):
        rows = table('kernels/ptq1-avx2.inc', 'ptq1a_tab')
        for i in range(32):
            v = i & 15
            self.assertEqual([rows[0][i], rows[1][i]], [v % 3, v // 3] if v < 9 else [0, 0], i)

    def test_no_code_runs_when_a_cpu_variant_is_loaded(self):
        # --portable: the loader opens every libggml-cpu-<variant>.so to ask for its score. A constructor in a variant
        # built for AVX-512 runs AVX-512 code right there, and the process dies on a CPU without it.
        for path in list((ROOT / 'kernels').iterdir()) + list((ROOT / 'patches').glob('*.patch')):
            self.assertNotIn('__attribute__((constructor', path.read_text(), path.name)


if __name__ == '__main__':
    unittest.main()
