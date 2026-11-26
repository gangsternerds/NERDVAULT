# Changelog

All notable changes to openset-llm. Dates are YYYY-MM-DD.

## Unreleased

## 0.2.0 (2026-11-26)

Added

- `./serve --multi N`: a system prompt the streams share is read once and loaded from disk afterwards (the prefix part
  of agent-loop resume; it was off with `--multi`). Same replies as `--no-resume`, no extra memory. `--multi` now
  writes prefix states to `~/.openset-llm/state` like the single-stream server does. (#17)
- `bench/quick.sh`: a 3-minute speed check that prints a block to paste into a "Measurement on my machine" issue. (#7)
- `docker-compose.yml`: `docker compose up` builds the image, keeps the model in a named volume and waits on
  `/health`. (#8)
- `./install.sh --check` is mentioned in the quick start. (#6)
- Unit tests for the installer, `serve`, `start`, the model graft and the benchmark harness (`tests/`,
  `bench/test_*.py`), run by CI.

Fixed: engine and install

- A prefix state file that was cut short on disk (for example by a full disk) made the server abort when a request
  tried to load it. Such a file is now detected, removed and written again. (#17)
- `patches/apply.sh` reported success and carried on after a patch failed to apply. It now stops at the first
  failure. (#36)
- `install.sh --bin-url`: a broken or unusable prebuilt archive no longer replaces a working engine; the new binaries
  are unpacked and checked in a staging folder first. (#72)
- `install.sh`: a download from a mirror that ignores resume requests restarts from the beginning. (#74)
- `install.sh`: checks for git before downloading weights when the speculation head has to be attached (#58), clears
  the plain-model marker after the head is attached (#60), and skips the Python setup when the model is already
  there (#62).
- `scripts/graft_mtp.py`: refuses an output path that is one of its inputs (#68) and keeps the tensor alignment the
  model file declares (#70).
- `./serve`: numbers with leading zeros (`--ctx 08192`) are read as decimal. (#64)
- `./start`: honours `--host` when it waits for the server and opens the page, and does not open a page with
  `--no-webui`. (#66)

Fixed: benchmark harness (no published number changes)

- `bench/quick.sh` exits with an error, and without a report, when a measurement fails. (#54)
- `bench/bench2.py`, `bench/multi.py`: memory samplers stop on every exit path (#38, #56); a failed stream is
  reported; `--rounds` above 3 keeps four prompts per round (#44); an interrupted last result line is repaired on
  resume (#52).
- `bench/rrun.py`: starts and stops only its own server (#46) and fails when the server never becomes ready (#48).
- `bench/evaluate.py`: keeps decorators when grading code (#40) and stops child processes of a timed-out
  program (#50).
- `bench/analyze.py`: load-only probes are listed separately from throughput rows. (#42)

Thanks to Afloat16 for #36–#74, revaldianggara for #6 and recranvvi5c for #8.

## 0.1.0 (2026-11-25)

First public release.

- A 27B LLM (Qwen3.8-27B as PrismML's ternary Bonsai-2-27B) at 17–20 tokens per second on a CPU-only machine with
  8 cores and 8 GB of RAM: new AVX-512 kernels for the ternary weights, speculative decoding with the model's own
  multi-token-prediction head, a fused attention kernel, several requests in one pass (`--multi`), agent-loop resume.
- One-command install and start (`./start`), a chat page, an OpenAI-compatible server (`./serve`), a Dockerfile, and
  the benchmark harness behind every number in the README (`bench/`).
- `./serve --kv6` and `--kv5` (experimental): 6- and 5-bit attention caches. On an 8 GB machine speculation stays on
  to 12k / 16k context and plain decoding reaches 32k / 64k. Not exact; needs AVX-512; see the README's memory section.
- `./serve` refuses two cache flags together and contexts that cannot fit with `--kv6` / `--kv5` under 12 GB.
- LoRA adapters load through `./serve -- --lora <file>` (documented with measured cost; #1, #2).

Known limits: Linux x86-64 only; the fast path needs AVX-512 VNNI + VBMI; one model; measured on cloud VMs
(Intel Ice Lake, AMD Zen 4 / Zen 5). See the README's Limits section.
