import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ApplyFailure(unittest.TestCase):
    def test_first_failed_patch_stops_the_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches, kernels, source = root / 'patches', root / 'kernels', root / 'engine'
            for path in (patches, kernels, source / 'ggml/src/ggml-cpu'):
                path.mkdir(parents=True)
            shutil.copyfile(ROOT / 'patches/apply.sh', patches / 'apply.sh')
            for name in ('ptq1-common.h', 'ptq1-drop.h', 'ptq1-fast.inc', 'ptq1-panel.inc',
                         'ptq1-vbmi.inc', 'ptq1-avx2.inc', 'x1-q4k.inc', 'attn-fast.inc', 'kvq.h'):
                (kernels / name).touch()
            paths = ('ggml/src/ggml-cpu/ggml-cpu.c', 'ggml/src/ggml-cpu/ops.cpp',
                     'src/models/qwen35.cpp', 'src/llama-context.cpp', 'tools/server/server-context.cpp',
                     'common/chat.h', 'common/common.h', 'common/arg.cpp', 'ggml/include/ggml.h', 'ggml/src/ggml.c')
            for name in paths:
                file = source / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('original\n')
            subprocess.run(['git', 'init', '-q', str(source)], check=True)
            subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture', '-c',
                            'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            (patches / '0001-invalid.patch').write_text('not a patch\n')
            (patches / '0002-next.patch').write_text('diff --git a/common/chat.h b/common/chat.h\n'
                '--- a/common/chat.h\n+++ b/common/chat.h\n@@ -1 +1 @@\n-original\n+changed\n')
            result = subprocess.run(['sh', str(patches / 'apply.sh'), str(source)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual((source / 'common/chat.h').read_text(), 'original\n')
            self.assertNotIn('applied: 0002', result.stdout)


if __name__ == '__main__':
    unittest.main()
