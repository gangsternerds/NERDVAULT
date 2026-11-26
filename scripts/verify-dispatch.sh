#!/usr/bin/env bash
# Check that every CPU path gives the same answers, on one AVX-512 VNNI+VBMI machine.
# Needs a portable build (./install.sh --portable): it holds one libggml-cpu-<variant>.so per x86 level, and the
# loader picks the best one the CPU supports. Hiding the better variants forces the slower paths on this machine.
#   bash scripts/verify-dispatch.sh [n_predict]      -> prints per-arm tok/s, which kernel loaded, output identity
# Arms: fast (icelake variant, defaults) | k6-off (GGML_K6=0) | fast-off (GGML_PTQ1_FAST=0) | avx512-novbmi (cascadelake) |
#       avx2 (haswell: the 256-bit kernels) | avx2-fork (haswell, GGML_PTQ1_AVX2=0: the fork's own kernel) |
#       avx512-novnni (skylakex: the 256-bit kernels in an AVX-512 build without VNNI) |
#       fast-256 (icelake, GGML_PTQ1_AVX2=1: the 256-bit kernels inside the AVX-512 build; must equal fast) |
#       fast+MTP-3 (speculation vs plain) | X1 (serve's default env) vs MTP-3 on the fast, no-VBMI and AVX2
#       paths (X1 must not change any path's output)
# The alderlake variant (AVX-VNNI) only loads on a CPU that has AVX-VNNI, so it is not an arm here.
set -uo pipefail
PREFIX=${OPENSET_LLM_HOME:-$HOME/.openset-llm}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
X1ENV="GGML_X1=109 LLAMA_ARG_REPACK=0 X1_MTP_ANCHOR_ONLY=1 X1_DRAFT_VOCAB=65536 X1_DRAFT_IDS=$HERE/data/x1-draft-ids.txt X1_Q6_AS_Q4=1 X1_DRAFT_FALLBACK=0.2"
SRCBIN=$PREFIX/llama.cpp/build/bin
MODEL=$PREFIX/models/model.gguf
N=${1:-64}
W=$(mktemp -d)
PROMPTS=(
  "Write a Python function that returns the n-th Fibonacci number iteratively."
  "Q: A train travels 120 km in 1.5 hours. What is its average speed in km/h? A:"
  "Translate to French: The weather is lovely today and we are going to the beach."
  "The three main causes of the French Revolution were"
  '{"name": "Alice", "age": 30, "skills": ['
  "Explain in two sentences why the sky is blue."
)
mkbin() {   # variant -> dir holding only that CPU variant
  local d=$W/bin-$1; mkdir -p "$d"; cp -a "$SRCBIN"/* "$d"/
  for f in "$d"/libggml-cpu-*.so; do [ "$f" = "$d/libggml-cpu-$1.so" ] || rm -f "$f"; done
  echo "$d"
}
run_arm() {   # name bindir env spec
  local name=$1 dir=$2 envs=$3 spec=$4 port=8099
  local sargs=(); [ "$spec" = 1 ] && sargs=(--spec-type draft-mtp --spec-draft-n-max 3)
  env $envs LD_LIBRARY_PATH="$dir" "$dir/llama-server" -m "$MODEL" -t "$(nproc)" -c 4096 -ub 256 -np 1 \
    --cache-ram 0 --ctx-checkpoints 1 --no-webui --port $port "${sargs[@]}" > "$W/srv-$name.log" 2>&1 &
  local pid=$!
  for _ in $(seq 300); do curl -sf localhost:$port/health >/dev/null 2>&1 && break; sleep 2; done
  local kern; kern=$(grep -o 'libggml-cpu-[a-z0-9_.]*\.so' /proc/$pid/maps 2>/dev/null | head -1)   # the variant loaded
  local i=0
  : > "$W/out-$name.txt"; : > "$W/tps-$name.txt"
  for p in "${PROMPTS[@]}"; do
    body=$(python3 -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"n_predict":int(sys.argv[2]),"temperature":0,"cache_prompt":False}))' "$p" "$N")
    curl -s localhost:$port/completion -d "$body" > "$W/r.json"
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r["content"]))' "$W/r.json" >> "$W/out-$name.txt"
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["timings"]["predicted_per_second"])' "$W/r.json" >> "$W/tps-$name.txt"
    i=$((i+1))
  done
  kill $pid; wait $pid 2>/dev/null
  local tps; tps=$(sort -g "$W/tps-$name.txt" | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  printf '%-14s kernel=%-26s median decode %6.2f tok/s\n' "$name" "${kern:-?}" "$tps"
}
FAST=$(mkbin icelake); NOVBMI=$(mkbin cascadelake); AVX2=$(mkbin haswell); NOVNNI=$(mkbin skylakex)
run_arm fast "$FAST" "" 0
run_arm k6-off "$FAST" "GGML_K6=0" 0
run_arm fast-off "$FAST" "GGML_PTQ1_FAST=0" 0
run_arm avx512-novbmi "$NOVBMI" "" 0
run_arm avx2 "$AVX2" "" 0
run_arm avx2-fork "$AVX2" "GGML_PTQ1_AVX2=0" 0
run_arm avx512-novnni "$NOVNNI" "" 0
run_arm fast-256 "$FAST" "GGML_PTQ1_AVX2=1" 0
run_arm fast-mtp3 "$FAST" "" 1
run_arm fast-x1 "$FAST" "$X1ENV" 1
run_arm novbmi-mtp3 "$NOVBMI" "" 1
run_arm novbmi-x1 "$NOVBMI" "$X1ENV" 1
run_arm avx2-mtp3 "$AVX2" "" 1
run_arm avx2-x1 "$AVX2" "$X1ENV" 1
same() { paste -d'\n' "$W/out-$1.txt" "$W/out-$2.txt" | paste - - | awk -F'\t' '$1==$2' | wc -l; }
echo "== greedy identity vs 'fast' (${#PROMPTS[@]} prompts x $N tokens)"
for a in k6-off fast-off avx512-novbmi fast-256 avx2 avx2-fork avx512-novnni fast-mtp3; do
  printf '%-14s %d/%d identical\n' "$a" "$(same fast $a)" "${#PROMPTS[@]}"
done
echo "== 256-bit kernels: fast-256 must be identical to fast (same kernel arithmetic). The avx2 arms are a different"
echo "   build (every other operation runs at 256 bits), so a few replies can part at a near-tie; they must agree with"
echo "   each other about as often as with fast"
printf '%-14s %d/%d identical to avx2-fork\n' avx2 "$(same avx2 avx2-fork)" "${#PROMPTS[@]}"
echo "== X1 vs MTP-3 on the same path (must be identical)"
for p in fast novbmi avx2; do
  printf '%-14s %d/%d identical\n' "$p-x1" "$(same $p-mtp3 $p-x1)" "${#PROMPTS[@]}"
done
echo "outputs: $W"
