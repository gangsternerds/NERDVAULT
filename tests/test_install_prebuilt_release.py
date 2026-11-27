import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARBALL = 'openset-llm-linux-x86_64.tar.gz'


class PrebuiltRelease(unittest.TestCase):
    """./install.sh --prebuilt: the release tarball is installed only if its checksum matches SHA256SUMS and it was
    built from the same engine sources as this checkout. Otherwise the working engine stays as it was."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.prefix = self.work / 'installation'
        self.tools = self.work / 'tools'
        self.tools.mkdir()
        (self.prefix / 'bin').mkdir(parents=True)
        (self.prefix / 'models').mkdir()
        (self.prefix / 'models/model.gguf').write_bytes(b'cached')
        self.command(self.prefix / 'bin/llama-server', '#!/bin/sh\nexit 0\n')
        self.before = (self.prefix / 'bin/llama-server').read_bytes()
        for name in ('dirname', 'uname', 'grep', 'cut', 'sed', 'awk', 'cat', 'mkdir', 'id', 'mv', 'rm', 'touch',
                     'tar', 'gzip', 'head', 'mktemp', 'basename', 'flock', 'sha256sum', 'cksum', 'cp'):
            executable = shutil.which(name)
            if executable:
                (self.tools / name).symlink_to(executable)
        self.command(self.tools / 'df', "#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted\nfixture 999999999 0 999999999 0%% /\n'\n")
        self.command(self.tools / 'du', "#!/bin/sh\nprintf '0 fixture\n'\n")
        self.command(self.tools / 'python3', '#!/bin/sh\nexit 0\n')
        # the release: -o FILE URL writes the tarball, a URL ending in SHA256SUMS prints the checksum file
        self.command(self.tools / 'curl', '#!/bin/sh\n'
            'out=; url=\n'
            'while [ "$#" -gt 0 ]; do case $1 in -o) out=$2; shift ;; --retry) shift ;; -*) ;; *) url=$1 ;; esac; shift; done\n'
            'printf \'%s\\n\' "$url" >> "$URL_LOG"\n'
            'case $url in\n'
            '  */SHA256SUMS) [ -f "$SUMS" ] || exit 22; cat "$SUMS" ;;\n'
            '  *) [ -f "$ARCHIVE" ] || exit 22; if [ -n "$out" ]; then cp "$ARCHIVE" "$out"; else cat "$ARCHIVE"; fi ;;\n'
            'esac\n')
        self.archive = self.work / TARBALL
        self.sums = self.work / 'SHA256SUMS'
        self.env = {**os.environ, 'OPENSET_LLM_HOME': str(self.prefix), 'PATH': str(self.tools),
                    'OPENSET_LLM_RELEASE_URL': 'https://example.invalid/release',
                    'ARCHIVE': str(self.archive), 'SUMS': str(self.sums), 'URL_LOG': str(self.work / 'urls')}
        self.env.pop('OPENSET_LLM_BIN_URL', None)
        self.env.pop('HF_TOKEN', None)
        self.engine_id = subprocess.run(['/bin/bash', str(ROOT / 'install.sh'), '--engine-id'],
                                        text=True, capture_output=True, check=True).stdout

    def command(self, path, content):
        path.write_text(content)
        path.chmod(0o755)

    def release(self, engine_id=None, checksum=None, listed=True):
        payload = b'#!/bin/sh\nexit 0\n# release engine\n'
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            members = [('openset-llm/llama-server', payload, 0o755)]
            if engine_id is not None:
                members.append(('openset-llm/.engine-id', engine_id.encode(), 0o644))
            for name, data, mode in members:
                entry = tarfile.TarInfo(name)
                entry.mode = mode
                entry.size = len(data)
                tar.addfile(entry, io.BytesIO(data))
        self.archive.write_bytes(stream.getvalue())
        digest = checksum or hashlib.sha256(stream.getvalue()).hexdigest()
        self.sums.write_text(f'{digest}  {TARBALL if listed else "another-file.tar.gz"}\n')
        return payload

    def install(self, *args):
        return subprocess.run(['/bin/bash', str(ROOT / 'install.sh'), *args, '--no-mtp'],
                              env=self.env, text=True, capture_output=True, timeout=15)

    def assert_untouched(self, result, message):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)
        self.assertEqual((self.prefix / 'bin/llama-server').read_bytes(), self.before)
        self.assertEqual(list(self.prefix.glob('.bin.*')), [])

    def test_matching_release_is_installed(self):
        payload = self.release(engine_id=self.engine_id)
        result = self.install('--prebuilt')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.prefix / 'bin/llama-server').read_bytes(), payload)
        self.assertFalse((self.prefix / 'bin/.download').exists())
        self.assertEqual(list(self.prefix.glob('.bin.*')), [])
        self.assertEqual((self.work / 'urls').read_text().split(),
                         [f'https://example.invalid/release/{TARBALL}', 'https://example.invalid/release/SHA256SUMS'])

    def test_wrong_checksum_installs_nothing(self):
        self.release(engine_id=self.engine_id, checksum='0' * 64)
        self.assert_untouched(self.install('--prebuilt'), 'checksum mismatch')

    def test_tarball_missing_from_the_checksum_file_installs_nothing(self):
        self.release(engine_id=self.engine_id, listed=False)
        self.assert_untouched(self.install('--prebuilt'), 'checksum mismatch')

    def test_missing_checksum_file_installs_nothing(self):
        self.release(engine_id=self.engine_id)
        self.sums.unlink()
        self.assert_untouched(self.install('--prebuilt'), 'SHA256SUMS')

    def test_engine_from_other_sources_installs_nothing(self):
        self.release(engine_id='0000000000000000000000000000000000000000 1\n')
        self.assert_untouched(self.install('--prebuilt'), 'different engine sources')

    def test_engine_without_an_id_installs_nothing(self):
        self.release(engine_id=None)
        self.assert_untouched(self.install('--prebuilt'), 'different engine sources')

    def test_release_without_a_prebuilt_engine_says_how_to_build(self):
        result = self.install('--prebuilt')   # no tarball on the release
        self.assert_untouched(result, 'Run plain ./install.sh')

    def test_prebuilt_excludes_bin_url_and_portable(self):
        for other in (('--bin-url', 'https://example.invalid/engine.tar.gz'), ('--portable',)):
            with self.subTest(other=other[0]):
                result = self.install('--prebuilt', *other)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('cannot be used together', result.stderr)
                self.assertEqual((self.prefix / 'bin/llama-server').read_bytes(), self.before)

    def test_engine_id_names_the_fork_commit_and_ignores_the_installer(self):
        fork = [line.split('=', 1)[1] for line in (ROOT / 'install.sh').read_text().splitlines()
                if line.startswith('FORK_SHA=')][0]
        self.assertEqual(self.engine_id.split()[0], fork)
        self.assertEqual(len(self.engine_id.split()), 2)


if __name__ == '__main__':
    unittest.main()
