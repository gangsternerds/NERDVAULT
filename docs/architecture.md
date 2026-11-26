# Architecture Map (`patches/`, `kernels/`, `scripts/`)

`openset-llm` is structured as a modular set of SIMD kernels and focused runtime patches applied cleanly on top of a pinned commit of PrismML's `llama.cpp` fork (`patches/apply.sh`).

## 1. Runtime Patch Stack (`patches/`)

Applied in order by `patches/apply.sh` during `./install.sh` (or `Dockerfile` build):

| Patch | Role |
|---|---|
| `0001-ptq1-fast-path-and-rowsplit.patch` | Ternary (`PTQ1`) fast-path matmul dispatch and row-splitting across CPU threads |
| `0002-k6-decode-state-fusions.patch` | Decode-loop state fusions to minimize memory traffic per token |
| `0003-x1-speculation.patch` | `X1` / MTP speculative decoding integration (`--spec` / `--no-spec`) |
| `0004-x2-multistream.patch` | `X2` multi-stream batching (`--multi N`) so concurrent requests share weight passes |
| `0005-adaptive-speculation.patch` | Experimental adaptive speculative draft depth (`--spec-adapt`, `--spec-depth`) |
| `0006-tool-turn-checkpoints.patch` | Checkpoint saving across multi-turn tool / agent conversations |
| `0007-resume.patch` | Agent-loop prefix state persistence in `$OPENSET_LLM_HOME/state` |
| `0008-attention-kernel.patch` | Custom CPU attention kernel integration |
| `0009-kvq-types.patch` | Experimental quantized KV cache types (`--kv8`, `--kv6`, `--kv5`) |
| `0010-multi-resume.patch` | Shared system-prompt disk resume for `--multi N` plus truncated-state recovery |

## 2. Custom CPU Kernels (`kernels/`)

| File | Purpose |
|---|---|
| `ptq1-common.h` / `ptq1-drop.h` | Shared ternary (`PTQ1`) unpacking types and helpers |
| `ptq1-fast.inc` / `ptq1-panel.inc` | Fast-path and panelized AVX-512 ternary matmul loops |
| `ptq1-vbmi.inc` | AVX-512 VNNI + VBMI kernel path (full fast path on Ice Lake+ and Zen 4+) |
| `x1-q4k.inc` | Quantized draft-head (`X1` / MTP) kernel path |
| `attn-fast.inc` | Fast CPU attention inner loops |
| `kvq.h` | Quantized attention KV cache definitions (`--kv8` / `--kv6` / `--kv5`) |

## 3. Packaging, Model Prep & Verification (`scripts/`)

| Script | Purpose |
|---|---|
| `scripts/graft_mtp.py` | Grafts the MTP speculative decoding sidecar onto the base ternary GGUF during install |
| `scripts/verify-dispatch.sh` | Verifies dynamic CPU backend variant dispatch on portable builds |
| `scripts/package.sh` | Packs a `--portable` build into a redistributable tarball for `./install.sh --bin-url` |
| `scripts/cross-aarch64.sh` | Cross-compilation helper targeting Linux `aarch64` |

## Where to Contribute

See `CONTRIBUTING.md`: change code inside `patches/` and `kernels/` (not a build checkout), keep shell scripts passing `bash -n` and Python compiling (`python3 -m py_compile`), and attach `./bench/quick.sh` or `bench/` output for any speed-related patch.
