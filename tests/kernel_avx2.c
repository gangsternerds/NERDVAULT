// kernel_avx2.c: the 256-bit ternary kernels (kernels/ptq1-avx2.inc) against a scalar reference. Run by
// tests/test_kernel_avx2.py, which compiles it once per dot-product variant the CPU can execute:
//   cc -O2 -mavx2 -mfma -mf16c [-mavxvnni | -mavx512f -mavx512bw -mavx512vl -mavx512vnni | -DPTQ1A_FORCE_MADD] -I kernels tests/kernel_avx2.c -lm
// Checks, all bit-exact (max |diff| must be 0):
//   1. the 16-bit word decode, every one of the 59049 values, with the two multiplies emulated in scalar code;
//   2. the AVX2 activation quantizer = the scalar Q8_0 quantizer of ptq1-common.h, byte for byte;
//   3. PTQ1_0 blocks -> panels -> M = 1 kernel (panel pairs and a single last panel, a partial last panel);
//   4. the same through the M >= 2 kernels for 2..9 and 64 tokens (the verify and prompt paths).
// The reference decodes the PTQ1_0 blocks with ptq1f_unpack_block and sums in the order the kernels promise:
// exact int32 per 32-k block, one fma per block in block order.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
typedef uint16_t ggml_half;
typedef struct { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; } block_ptq1_0;
#define GGML_CPU_FP16_TO_FP32(h) _cvtsh_ss(h)
#define GGML_CPU_FP32_TO_FP16(f) _cvtss_sh((f), 0)
#define PTQ1F_NO_GGML
#define PTQ1F_MIN(a, b) ((a) < (b) ? (a) : (b))
#include "ptq1-common.h"
#include "ptq1-avx2.inc"
#if !PTQ1A_OK
#error "needs -mavx2 -mfma -mf16c"
#endif

static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t) (rng_s >> 32); }
static float rndf(void) { return (float) (rnd() & 0xffffff) / (float) 0x800000 - 1.0f; }   // [-1, 1)

static int check_words(void) {
    int bad = 0;
    for (uint32_t q = 0; q < 59049; ++q) {
        uint32_t w = (uint32_t) (((uint64_t) q * 65536 + 59048) / 59049), rem = q;
        if (w > 0xffff) { ++bad; continue; }
        for (int s = 0; s < 5; ++s) {
            const uint32_t d = (w * 9) >> 16; w = (w * 9) & 0xffff;
            const uint32_t want = rem / 6561; rem = rem % 6561 * 9;
            if (d != want) { ++bad; break; }
        }
    }
    return bad;
}

static void random_block(block_ptq1_0 * b) {
    uint8_t c5[5];
    for (int m = 0; m < 24; ++m) { for (int n = 0; n < 5; ++n) c5[n] = (uint8_t) (rnd() % 3); b->qs[m] = ptq1f_pack5(c5); }
    for (int h = 0; h < 2; ++h) { for (int n = 0; n < 4; ++n) c5[n] = (uint8_t) (rnd() % 3); c5[4] = 0; b->qh[h] = ptq1f_pack5(c5); }
    b->d = GGML_CPU_FP32_TO_FP16(0.005f + 0.02f * (float) (rnd() % 1000) / 1000.0f);
}

// one output of the reference: row r of the weights x one quantized column (as[] holds -sum(q) per block)
static float ref_row(const block_ptq1_0 * w, int64_t k, const uint8_t * acts, int64_t kpad) {
    const int8_t * q = (const int8_t *) acts; const float * dy = (const float *) (acts + kpad);
    const int32_t * as = (const int32_t *) (dy + kpad / 32);
    uint8_t codes[128]; float accf = 0;
    for (int64_t b = 0; b < kpad / 32; ++b) {
        int32_t acc = 0; float dx = 0;
        if (b * 32 < k) {
            ptq1f_unpack_block(w + b / 4, codes);
            for (int j = 0; j < 32; ++j) acc += (int32_t) codes[(b % 4) * 32 + j] * q[b * 32 + j];
            dx = GGML_CPU_FP16_TO_FP32(w[b / 4].d);
        }
        accf = fmaf((float) (acc + as[b]), dx * dy[b], accf);
    }
    return accf;
}

static int run_case(int64_t rows, int64_t k, int64_t m) {
    const int64_t nb = k / 128;
    block_ptq1_0 * w = malloc((size_t) (rows * nb) * sizeof(*w));
    for (int64_t i = 0; i < rows * nb; ++i) random_block(w + i);
    ptq1p_entry e; memset(&e, 0, sizeof e);
    ptq1a_init_entry(&e, w, rows, k);
    e.packed = aligned_alloc(64, (size_t) ((e.panel_stride * e.npanel + 63) / 64 * 64 + 64));
    ptq1a_repack_panels(&e, 0, e.npanel);
    const size_t cb = ptq1p_col_bytes(k);
    uint8_t * acts = aligned_alloc(64, (cb * (size_t) m + 63) / 64 * 64), * acts_ref = aligned_alloc(64, (cb + 63) / 64 * 64);
    float * x = malloc((size_t) (k * m) * sizeof(float));
    int bad = 0;
    for (int64_t t = 0; t < m; ++t) {
        const float amp = t % 3 == 0 ? 1.0f : t % 3 == 1 ? 37.5f : 1e-3f;
        for (int64_t i = 0; i < k; ++i) x[t * k + i] = rndf() * amp;
        if (t == 1) for (int64_t i = 0; i < 64 && i < k; ++i) x[t * k + i] = 0.0f;            // all-zero blocks: d = 0
        if (t == 2) for (int64_t i = 0; i < k; ++i) x[t * k + i] = (float) ((i * 7) % 255 - 127) * 0.5f;   // exact .5 ties after scaling
        ptq1a_quantize_part(x + t * k, k, e.kpad, acts + t * cb, 0, k / 32, 1);
        memset(acts_ref, 0xAA, cb);
        ptq1p_quantize_col_part_kp(x + t * k, k, e.kpad, acts_ref, 0, k / 32, 1);
        if (memcmp(acts + t * cb, acts_ref, (size_t) (e.kpad + (e.kpad / 32) * 8)) != 0) { printf("FAIL quantizer rows=%lld k=%lld token %lld\n", (long long) rows, (long long) k, (long long) t); ++bad; }
    }
    float * ref = malloc((size_t) (rows * m) * sizeof(float)), * out = malloc((size_t) (rows * m + 8) * sizeof(float));
    for (int64_t t = 0; t < m; ++t) for (int64_t r = 0; r < rows; ++r) ref[t * rows + r] = ref_row(w + r * nb, k, acts + t * cb, e.kpad);
    // M = 1 kernel on every token's column, in two splits (pairs + single last panel; one panel at a time)
    for (int split = 0; split < 2; ++split) {
        for (int64_t i = 0; i < rows * m + 8; ++i) out[i] = NAN;
        for (int64_t t = 0; t < m; ++t) {
            if (split == 0) ptq1a_panels(&e, acts + t * cb, out + t * rows, 0, e.npanel);
            else for (int64_t p = 0; p < e.npanel; ++p) ptq1a_panels(&e, acts + t * cb, out + t * rows, p, p + 1);
        }
        if (memcmp(out, ref, (size_t) (rows * m) * sizeof(float)) != 0) { printf("FAIL M=1 kernel rows=%lld k=%lld split %d\n", (long long) rows, (long long) k, split); ++bad; }
        if (!isnan(out[rows * m])) { printf("FAIL M=1 kernel wrote past the last row\n"); ++bad; }
    }
    // M >= 2 kernel, panels in chunks of 2 (as the scheduler hands them out) and all at once
    if (m >= 2) for (int split = 0; split < 2; ++split) {
        for (int64_t i = 0; i < rows * m + 8; ++i) out[i] = NAN;
        if (split == 0) ptq1a_gemm(&e, acts, cb, m, 0, e.npanel, out, rows);
        else for (int64_t p = 0; p < e.npanel; p += 2) ptq1a_gemm(&e, acts, cb, m, p, PTQ1F_MIN(e.npanel, p + 2), out, rows);
        if (memcmp(out, ref, (size_t) (rows * m) * sizeof(float)) != 0) {
            float md = 0; for (int64_t i = 0; i < rows * m; ++i) md = fmaxf(md, fabsf(out[i] - ref[i]));
            printf("FAIL M=%lld kernel rows=%lld k=%lld split %d max|diff| %g\n", (long long) m, (long long) rows, (long long) k, split, md); ++bad;
        }
        if (!isnan(out[rows * m])) { printf("FAIL M>=2 kernel wrote past the last row\n"); ++bad; }
    }
    // the in-register kernel for small batches (the scheduler uses it for 2..4 tokens; any M is valid)
    if (m >= 2) {
        for (int64_t i = 0; i < rows * m + 8; ++i) out[i] = NAN;
        for (int64_t p = 0; p < e.npanel; p += 3) ptq1a_panels_m(&e, acts, cb, m, p, PTQ1F_MIN(e.npanel, p + 3), out, rows);
        if (memcmp(out, ref, (size_t) (rows * m) * sizeof(float)) != 0) { printf("FAIL small-batch kernel M=%lld rows=%lld k=%lld\n", (long long) m, (long long) rows, (long long) k); ++bad; }
        if (!isnan(out[rows * m])) { printf("FAIL small-batch kernel wrote past the last row\n"); ++bad; }
    }
    free(w); free(e.packed); free(acts); free(acts_ref); free(x); free(ref); free(out);
    return bad;
}

int main(void) {
    int bad = check_words();
    if (bad) printf("FAIL word decode: %d of 59049 values\n", bad);
    static const int64_t rows_k[][2] = { {8, 128}, {16, 640}, {21, 1152}, {40, 5120}, {3, 384}, {17, 2560} };
    static const int64_t ms[] = { 1, 2, 3, 4, 5, 8, 9, 64 };
    int cases = 0;
    for (size_t i = 0; i < sizeof rows_k / sizeof rows_k[0]; ++i)
        for (size_t j = 0; j < sizeof ms / sizeof ms[0]; ++j) { bad += run_case(rows_k[i][0], rows_k[i][1], ms[j]); ++cases; }
    printf("dot=%s cases=%d failures=%d\n", PTQ1A_VNNI ? "vpdpbusd" : "vpmaddubsw", cases, bad);
    return bad ? 1 : 0;
}
