import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstallRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.prefix = self.work / "installation"
        self.tools = self.work / "tools"
        self.tools.mkdir()
        (self.prefix / "bin").mkdir(parents=True)
        (self.prefix / "models").mkdir()
        self.command(self.prefix / "bin/llama-server", "#!/bin/sh\nexit 0\n")
        for name in ("dirname", "uname", "grep", "cut", "sed", "awk", "cat", "mkdir", "id",
                     "mv", "rm", "touch", "tar", "gzip", "head", "mktemp", "basename", "flock"):
            executable = shutil.which(name)
            if executable:
                (self.tools / name).symlink_to(executable)
        self.command(self.tools / "df", "#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted\nfixture 999999999 0 999999999 0%% /\n'\n")
        self.command(self.tools / "du", "#!/bin/sh\nprintf '0 fixture\n'\n")
        self.command(self.tools / "python3", "#!/bin/sh\nexit 0\n")
        self.command(self.tools / "curl", "#!/bin/sh\nprintf download >> \"$DOWNLOAD_LOG\"\nwhile [ \"$#\" -gt 0 ]; do if [ \"$1\" = -o ]; then shift; printf fixture > \"$1\"; fi; shift; done\n")
        self.env = {**os.environ, "OPENSET_LLM_HOME": str(self.prefix),
                    "PATH": str(self.tools), "DOWNLOAD_LOG": str(self.work / "downloads")}
        self.env.pop("OPENSET_LLM_BIN_URL", None)
        self.env.pop("HF_TOKEN", None)

    def command(self, path, content):
        path.write_text(content)
        path.chmod(0o755)

    def install(self, *args):
        return subprocess.run(["/bin/bash", str(ROOT / "install.sh"), *args],
                              env=self.env, text=True, capture_output=True, timeout=15)

    def test_successful_graft_clears_stale_plain_model_marker(self):
        (self.prefix / "models/no-mtp").touch()
        (self.prefix / "models/base.gguf").write_bytes(b"base")
        (self.prefix / "models/mtp-sidecar.gguf").write_bytes(b"sidecar")
        (self.prefix / "llama.cpp/gguf-py").mkdir(parents=True)
        self.command(self.tools / "python3", "#!/bin/sh\nif [ \"$1\" = -I ] && [ \"$2\" != -c ]; then for output; do :; done; printf grafted > \"$output\"; fi\n")
        result = self.install("--model-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.prefix / "models/model.gguf").read_bytes(), b"grafted")
        self.assertFalse((self.prefix / "models/no-mtp").exists())

    def test_failed_graft_preserves_marker(self):
        (self.prefix / "models/no-mtp").touch()
        (self.prefix / "models/base.gguf").write_bytes(b"base")
        (self.prefix / "models/mtp-sidecar.gguf").write_bytes(b"sidecar")
        (self.prefix / "llama.cpp/gguf-py").mkdir(parents=True)
        self.command(self.tools / "python3", "#!/bin/sh\n[ \"$1\" != -I ] || [ \"$2\" = -c ]\n")
        result = self.install("--model-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.prefix / "models/no-mtp").exists())

    def test_failed_model_publication_preserves_marker_and_exit_status(self):
        (self.prefix / "models/no-mtp").touch()
        (self.prefix / "models/base.gguf").write_bytes(b"base")
        (self.prefix / "models/mtp-sidecar.gguf").write_bytes(b"sidecar")
        (self.prefix / "llama.cpp/gguf-py").mkdir(parents=True)
        self.command(self.tools / "python3", "#!/bin/sh\nif [ \"$1\" = -I ] && [ \"$2\" != -c ]; then for output; do :; done; printf grafted > \"$output\"; fi\n")
        real_mv = shutil.which("mv")
        (self.tools / "mv").unlink()
        self.command(self.tools / "mv", '#!/bin/sh\n'
            'case "$1" in */model.gguf.part) exit 31 ;; esac\n'
            f'exec "{real_mv}" "$@"\n')
        result = self.install("--model-only")
        self.assertEqual(result.returncode, 31, result.stderr)
        self.assertTrue((self.prefix / "models/no-mtp").exists())
        self.assertFalse((self.prefix / "models/model.gguf").exists())
        self.assertEqual((self.prefix / "models/base.gguf").read_bytes(), b"base")
        self.assertEqual((self.prefix / "models/mtp-sidecar.gguf").read_bytes(), b"sidecar")


if __name__ == "__main__":
    unittest.main()
