#!/bin/sh
# Apply this repo's changes to a PrismML llama.cpp checkout at e3de715. Idempotent (resets the patched files first).
#   sh patches/apply.sh <llama.cpp checkout>
# 1. kernels/*  -> ggml/src/ggml-cpu/  (the fast PTQ1_0 ternary matmul: AVX-512 kernels, and 256-bit kernels for CPUs
#    without AVX-512; the draft-layer kernel)
# 2. 0001-*.patch: wires it into ggml_compute_forward_mul_mat; splits single-row GET_ROWS/CONCAT across threads.
# 3. 0002-*.patch: K6, fuses the decode graph's recurrent-state and norm/GLU/sign chains (bit-identical data moves).
# 4. 0003-*.patch: X1, faster MTP speculation cycle (GGML_X1 + X1_* env; all unset = 0002's behaviour).
# 5. 0004-*.patch: X2, several streams per weight pass (GGML_X2 + LLAMA_X2_* env; all unset = 0003's behaviour).
# 6. 0005-*.patch: adaptive draft depth (LLAMA_SPEC_ADAPT=1; unset = 0004's behaviour).
# 7. 0006-*.patch: tool results count as user turns for context checkpoints (common/chat.h, 2 lines).
# 8. 0007-*.patch: resume where the next request will resume (LLAMA_RESUME=0 = 0006's behaviour; LLAMA_RESUME_DIR
#    = prefix states on disk). See README.md "What is inside".
# 9. 0008-*.patch: our attention kernel (kernels/attn-fast.inc) for FLASH_ATTN_EXT (GGML_ATTN=0 = the fork's paths), and
#    q8_K as a KV cache type (int8 KV, opt-in). GGML_ATTN=0 selects the fork's paths.
# 10. 0009-*.patch: kv5 / kv6 KV cache types (kernels/kvq.h: 5 / 6-bit rows read by attn-fast.inc; opt-in,
#    `serve --kv6` / `--kv5`).
# 11. 0010-*.patch: resume for `serve --multi`: LLAMA_RESUME_PREFIX_ONLY=1 with --ctx-checkpoints 0 uses the prefix states
#    on disk without a checkpoint in RAM (unset = 0007's resume logic); and state files are sealed with a trailer, so a
#    file cut short on disk is written again instead of aborting the server.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
src=${1:?usage: apply.sh <llama.cpp checkout>}
cp "$here"/../kernels/ptq1-common.h "$here"/../kernels/ptq1-drop.h "$here"/../kernels/ptq1-fast.inc "$here"/../kernels/ptq1-panel.inc \
   "$here"/../kernels/ptq1-vbmi.inc "$here"/../kernels/ptq1-avx2.inc "$here"/../kernels/x1-q4k.inc "$here"/../kernels/attn-fast.inc "$here"/../kernels/kvq.h "$src/ggml/src/ggml-cpu/"
git -C "$src" checkout -q -- ggml/src/ggml-cpu/ggml-cpu.c ggml/src/ggml-cpu/ops.cpp src/models/qwen35.cpp \
   src/llama-context.cpp tools/server/server-context.cpp common/chat.h common/common.h common/arg.cpp ggml/include/ggml.h ggml/src/ggml.c
for p in "$here"/0*.patch; do
  git -C "$src" apply "$p"
  echo "applied: $(basename "$p")"
done
