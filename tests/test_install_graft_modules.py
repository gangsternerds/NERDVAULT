import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstallRegression(unittest.TestCase):
    """install.sh runs the MTP graft with `python3 -I`, which sees neither the user's site-packages (pip install
    --user) nor PYTHONPATH. The check that decides whether a venv is needed has to look in the same mode."""

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
        self.command(self.tools / "curl", "#!/bin/sh\nprintf download >> \"$DOWNLOAD_LOG\"\nwhile [ \"$#\" -gt 0 ]; do if [ \"$1\" = -o ]; then shift; printf fixture > \"$1\"; fi; shift; done\n")
        # what `python3 -m venv` leaves behind: a pip that records its arguments and a python that "grafts"
        self.venv = self.work / "venv-fixture"
        (self.venv / "bin").mkdir(parents=True)
        self.command(self.venv / "bin/pip", "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$PIP_LOG\"\n")
        self.command(self.venv / "bin/python", "#!/bin/sh\nfor output; do :; done; printf venv-grafted > \"$output\"\n")
        self.env = {**os.environ, "OPENSET_LLM_HOME": str(self.prefix),
                    "PATH": str(self.tools), "DOWNLOAD_LOG": str(self.work / "downloads"),
                    "VENV_FIXTURE": str(self.venv), "PIP_LOG": str(self.work / "pip")}
        self.env.pop("OPENSET_LLM_BIN_URL", None)
        self.env.pop("HF_TOKEN", None)
        (self.prefix / "models/base.gguf").write_bytes(b"base")
        (self.prefix / "models/mtp-sidecar.gguf").write_bytes(b"sidecar")
        (self.prefix / "llama.cpp/gguf-py").mkdir(parents=True)

    def command(self, path, content):
        path.write_text(content)
        path.chmod(0o755)

    def install(self, *args):
        return subprocess.run(["/bin/bash", str(ROOT / "install.sh"), *args],
                              env=self.env, text=True, capture_output=True, timeout=15)

    def test_modules_only_in_the_user_site_get_a_venv_for_the_graft(self):
        # numpy + pyyaml installed with `pip install --user`: a plain python3 imports them, python3 -I does not
        self.command(self.tools / "python3", "#!/bin/sh\n"
            "case \"$1 $2\" in\n"
            "  '-m venv') mv \"$VENV_FIXTURE\" \"$3\" ;;\n"
            "  -I*) echo \"ModuleNotFoundError: No module named 'numpy'\" >&2; exit 1 ;;\n"
            "esac\n")
        result = self.install("--model-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.prefix / "models/model.gguf").read_bytes(), b"venv-grafted")
        self.assertIn("numpy pyyaml", (self.work / "pip").read_text())

    def test_modules_the_isolated_python_can_import_need_no_venv(self):
        # numpy + pyyaml from the distribution (python3-numpy, python3-yaml): python3 -I imports them too
        self.command(self.tools / "python3", "#!/bin/sh\n"
            "case \"$1 $2\" in\n"
            "  '-m venv') mv \"$VENV_FIXTURE\" \"$3\" ;;\n"
            "  '-I -c') exit 0 ;;\n"
            "  -I*) for output; do :; done; printf grafted > \"$output\" ;;\n"
            "esac\n")
        result = self.install("--model-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.prefix / "models/model.gguf").read_bytes(), b"grafted")
        self.assertFalse((self.prefix / "venv").exists())
        self.assertFalse((self.work / "pip").exists())


if __name__ == "__main__":
    unittest.main()
