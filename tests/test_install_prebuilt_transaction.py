import os
import io
import tarfile
from pathlib import Path
import shutil
import signal
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
        # A test runner started as a background job has SIGINT ignored, and bash cannot trap a signal that
        # was ignored when it started. Give the installer the default so its INT trap can be tested there too.
        return subprocess.run(["/bin/bash", str(ROOT / "install.sh"), *args],
                              env=self.env, text=True, capture_output=True, timeout=15,
                              preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))

    def archive(self, truncate=False, executable=True):
        payload = b'#!/bin/sh\nexit 0\n# replacement engine\n' if executable else b'#!/bin/sh\nexit 12\n'
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            entry = tarfile.TarInfo('openset-llm/llama-server')
            entry.mode = 0o755
            entry.size = len(payload)
            tar.addfile(entry, io.BytesIO(payload))
        data = stream.getvalue()
        archive = self.work / 'engine.tar.gz'
        archive.write_bytes(data[:-8] if truncate else data)
        self.env['ARCHIVE'] = str(archive)
        self.command(self.tools / 'curl', '#!/bin/sh\ncat "$ARCHIVE"\n')
        (self.prefix / 'models/model.gguf').write_bytes(b'cached')
        return payload

    def test_truncated_archive_preserves_the_working_engine(self):
        before = (self.prefix / 'bin/llama-server').read_bytes()
        self.archive(truncate=True)
        result = self.install('--bin-url', 'https://example.invalid/engine.tar.gz', '--no-mtp')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.prefix / 'bin/llama-server').read_bytes(), before)
        self.assertEqual(list(self.prefix.glob('.bin.*')), [])

    def test_unusable_replacement_preserves_the_working_engine(self):
        before = (self.prefix / 'bin/llama-server').read_bytes()
        self.archive(executable=False)
        result = self.install('--bin-url', 'https://example.invalid/engine.tar.gz', '--no-mtp')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.prefix / 'bin/llama-server').read_bytes(), before)

    def test_successful_replacement_removes_obsolete_libraries(self):
        (self.prefix / 'bin/obsolete.so').write_bytes(b'old backend')
        expected = self.archive()
        result = self.install('--bin-url', 'https://example.invalid/engine.tar.gz', '--no-mtp')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.prefix / 'bin/llama-server').read_bytes(), expected)
        self.assertFalse((self.prefix / 'bin/obsolete.so').exists())

    def test_interruption_between_renames_restores_the_previous_engine(self):
        real_mv = shutil.which('mv')
        before = (self.prefix / 'bin/llama-server').read_bytes()
        for signal_name, exit_code in (('TERM', 143), ('INT', 130)):
            with self.subTest(signal=signal_name):
                (self.prefix / 'bin').mkdir(exist_ok=True)
                (self.prefix / 'bin/llama-server').write_bytes(before)
                self.archive()
                (self.tools / 'mv').unlink()
                self.command(self.tools / 'mv', '#!/bin/sh\n'
                    f'"{real_mv}" "$@" || exit $?\n'
                    'if [ "$1" = "$OPENSET_LLM_HOME/bin" ]; then\n'
                    f'  kill -{signal_name} "$PPID"\n'
                    'fi\n')
                result = self.install('--bin-url', 'https://example.invalid/engine.tar.gz', '--no-mtp')
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual((self.prefix / 'bin/llama-server').read_bytes(), before)
                self.assertEqual(list(self.prefix.glob('.bin.*')), [])

    def test_failed_publish_restores_engine_and_preserves_error_status(self):
        real_mv = shutil.which('mv')
        before = (self.prefix / 'bin/llama-server').read_bytes()
        self.archive()
        (self.tools / 'mv').unlink()
        self.command(self.tools / 'mv', '#!/bin/sh\n'
            'case "$1" in\n'
            '  "$OPENSET_LLM_HOME"/.bin.*.previous) ;;\n'
            '  "$OPENSET_LLM_HOME"/.bin.*) exit 37 ;;\n'
            'esac\n'
            f'exec "{real_mv}" "$@"\n')
        result = self.install('--bin-url', 'https://example.invalid/engine.tar.gz', '--no-mtp')
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual((self.prefix / 'bin/llama-server').read_bytes(), before)
        self.assertEqual(list(self.prefix.glob('.bin.*')), [])


if __name__ == '__main__':
    unittest.main()
