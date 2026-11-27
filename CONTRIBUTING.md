# Contributing to openset-llm

Thanks for looking. This is a small project; the most useful contributions are, in order:

1. **A measurement on your machine.** Speed depends on the CPU and the memory. Open an issue with the CPU model, the
   output of `./install.sh --check`, the "serving on ..." line `./serve` prints at start, and what you measured and how.
2. **A bug report** that someone else can reproduce: the exact commands, what you expected, what happened, and the
   same machine details as above. Use the bug template.
3. **A patch.** Small and focused is best. Open an issue first for anything that changes behaviour or a number in the
   README.

## Working on the code

- `patches/` and `kernels/` are applied to a pinned commit of PrismML's llama.cpp fork by `patches/apply.sh`; change them there, not in a checkout.
- Before sending a pull request, run `./install.sh` on a clean machine, then `./serve` and one request (the `curl` in the README); for anything that touches speed, `bench/README.md`.
- Shell scripts must pass `bash -n`; Python must compile. CI checks both.
- Run the unit tests before sending a pull request; CI runs the same two commands. They need no model and take
  under a minute:
  ```sh
  python3 -m pip install gguf numpy          # in a virtualenv; a few tests are skipped without these
  python3 -m unittest discover -s tests -p 'test_*.py'
  (cd bench && python3 -m unittest discover -s . -p 'test_*.py')
  ```
- A bug fix should come with a test that fails without the fix.
- With a server running, `bash scripts/smoke-test.sh` checks the three endpoints most clients use.
- Text a user sees (README, `--help`, messages) should say what a thing does in plain words.

## Numbers in the README

Every number in the README was measured, and most can be reproduced with `bench/` (its README lists the few inputs
we cannot publish). A pull request that changes a number needs the run that produced it: the machine, the commands
and the raw output.

## Licence

By contributing you agree that your contribution is licensed under the MIT licence of this repository (`LICENSE`).
