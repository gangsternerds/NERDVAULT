// ptq1-fast: CPU-native execution of PrismML PTQ1_0 ternary weights (Bonsai-2 / Qwen3.8-27B).
// #included in ggml-cpu.c; on by default (mode 7), GGML_PTQ1_FAST=0 turns it off, 1..7 pick a mode.
//
// Why: the stock path calls an SSE vec_dot once per (weight row, token), re-decoding the
// base-3 packing for every token, and gathers activations 16 bytes at a time because
// Q8_0 interleaves scales with data. Here:
//   1. On first use each weight tensor is re-laid out ("made CPU-friendly"): rows become
//      64-byte chunks of 320 trits where decode step n of byte j yields element 64n+j,
//      so 64 decoded codes always line up with 64 contiguous activations. 1.725 bpw,
//      slightly smaller than PTQ1_0's 1.75.
//   2. Activations are quantized with exactly Q8_0's int8 math (so every integer dot is
//      bit-identical to the reference) but stored planar, with per-32 scales pre-expanded
//      to one 16-float vector per 64 elements and the "-1" trit offset folded into a
//      per-128-group constant.
//   3. Each 64-trit step is decoded ONCE and applied to up to 8 tokens, so prefill and
//      speculative verification amortize the decode instead of repeating it.
#include "ptq1-common.h"
#include "ptq1-drop.h"
#ifndef PTQ1F_MIN
#define PTQ1F_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#include "ptq1-avx2.inc"   // the 256-bit kernels: the default on a build without AVX-512, GGML_PTQ1_AVX2=1 on one with it

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#include <immintrin.h>
#include <pthread.h>
#include <time.h>
#include <sys/mman.h>

#define PTQ1F_CHUNK 320          // trits per 64-byte chunk
#define PTQ1F_COLS  8            // tokens per decode pass

typedef struct ptq1f_entry {
    const void * src;            // original tensor data
    uint8_t    * packed;         // rows * row_stride bytes
    int64_t      rows, k, row_stride, n_chunks;
    struct ptq1f_entry * next;
} ptq1f_entry;

static ptq1f_entry * ptq1f_list = NULL;
static pthread_mutex_t ptq1f_mu = PTHREAD_MUTEX_INITIALIZER;
static int ptq1f_enabled_cache = -1;

#ifndef PTQ1F_MIN
#define PTQ1F_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static bool ptq1f_enabled(void) {
    if (ptq1f_enabled_cache < 0) {
        const char * e = getenv("GGML_PTQ1_FAST");
        ptq1f_enabled_cache = e ? atoi(e) : 7;   // default 7 (fastest; degrades to what the build supports); 0 = off, 1 = exact Q8_0 numerics, 2 = per-128 activation scales,
                                                   // 3 = row panels, 4 = row panels + unrolled M = 1 path,
                                                   // 5 = 4 + bit-aligned trit fields decoded with VBMI (ptq1-vbmi.inc),
                                                   // 6 = 5 + shared quantize, dynamic chunks and 1 chain/token for M = 2..8
                                                   // 7 = 6 + two panels interleaved per core in the M = 1 decode (K5)
    }
    return ptq1f_enabled_cache >= 1;
}

static int ptq1f_mode(void) { return ptq1f_enabled_cache; }

#ifndef PTQ1F_NO_GGML
static bool ptq1f_applies(const struct ggml_tensor * src0, const struct ggml_tensor * src1) {
    return src0->type == GGML_TYPE_PTQ1_0 && src1->type == GGML_TYPE_F32 &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && src0->ne[0] % 128 == 0 &&
           ptq1f_enabled();
}

#endif

// planar activation buffer per token: int8[k_pad] | f32 dyv[k_pad/4] | f32 A[k/128]
static size_t ptq1f_col_bytes(int64_t k) {
    const int64_t k_pad = (k + PTQ1F_CHUNK - 1) / PTQ1F_CHUNK * PTQ1F_CHUNK;
    return (size_t) k_pad + (size_t) k_pad + (size_t) (k / 128) * 4 + 128;
}

#ifndef PTQ1F_NO_GGML
static size_t ptq1p_col_bytes(int64_t k);
#if defined(__AVX512VBMI__)
static size_t ptq1v_col_bytes(int64_t k);
#endif
static size_t ptq1f_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1) {
#if PTQ1A_OK
    if (ptq1a_on()) return ptq1a_wsize(src0, src1);
#endif
#if defined(__AVX512VBMI__)
    if (ptq1f_mode() >= 5) return ptq1v_col_bytes(src0->ne[0]) * (size_t) src1->ne[1] + 64;
#endif
    if (ptq1f_mode() >= 3) return ptq1p_col_bytes(src0->ne[0]) * (size_t) src1->ne[1] + 64;
    return ptq1f_col_bytes(src0->ne[0]) * (size_t) src1->ne[1] + 64;
}
#endif

static void ptq1f_repack_rows(ptq1f_entry * e, int64_t r0, int64_t r1) {
    const int64_t nb = e->k / 128;
    uint8_t * codes = malloc((size_t) e->n_chunks * PTQ1F_CHUNK);
    for (int64_t r = r0; r < r1; ++r) {
        const block_ptq1_0 * src = (const block_ptq1_0 *) e->src + r * nb;
        uint8_t * dst = e->packed + r * e->row_stride;
        memset(codes, 1, (size_t) e->n_chunks * PTQ1F_CHUNK);   // padding trit 1 == weight 0
        for (int64_t b = 0; b < nb; ++b) ptq1f_unpack_block(src + b, codes + b * 128);
        for (int64_t c = 0; c < e->n_chunks; ++c) {
            for (int j = 0; j < 64; ++j) {
                uint8_t c5[5];
                for (int n = 0; n < 5; ++n) c5[n] = codes[c * PTQ1F_CHUNK + 64 * n + j];
                dst[c * 64 + j] = ptq1f_pack5(c5);
            }
        }
        ggml_half * sc = (ggml_half *) (dst + e->n_chunks * 64);
        for (int64_t b = 0; b < nb; ++b) sc[b] = src[b].d;
    }
    free(codes);
}

static inline __m512i ptq1f_decode(__m512i * x) {
    const __mmask64 ge1 = _mm512_cmpge_epu8_mask(*x, _mm512_set1_epi8((char) 86));
    const __mmask64 ge2 = _mm512_cmpge_epu8_mask(*x, _mm512_set1_epi8((char) 171));
    const __m512i one = _mm512_set1_epi8(1);
    __m512i c = _mm512_maskz_mov_epi8(ge1, one);
    c = _mm512_mask_add_epi8(c, ge2, c, one);
    *x = _mm512_add_epi8(*x, _mm512_add_epi8(*x, *x));
    return c;
}

// Q8_0-identical quantization of one token into the planar layout
static void ptq1f_quantize_col(const float * x, int64_t k, uint8_t * out) {
    const int64_t k_pad = (k + PTQ1F_CHUNK - 1) / PTQ1F_CHUNK * PTQ1F_CHUNK;
    int8_t * q = (int8_t *) out;
    float * dyv = (float *) (out + k_pad);
    float * A = (float *) (out + 2 * k_pad);
    memset(q, 0, (size_t) k_pad);
    memset(dyv, 0, (size_t) k_pad);
    for (int64_t g = 0; g < k / 128; ++g) A[g] = 0.0f;
    for (int64_t b = 0; b < k / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        const float dh = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(d));   // Q8_0 stores d as fp16
        int sum = 0;
        for (int j = 0; j < 32; ++j) {
            const int v = (int) roundf(x[b * 32 + j] * id);
            q[b * 32 + j] = (int8_t) v;
            sum += v;
        }
        // dyv: per 64-element step, lanes 0..7 <- first 32, lanes 8..15 <- second 32
        float * lane = dyv + (b / 2) * 16 + (b % 2) * 8;
        for (int l = 0; l < 8; ++l) lane[l] = dh;
        A[b / 4] += dh * (float) sum;
    }
}

// rows [r0,r1) x tokens [0,m): dst[t*ld + r]
static void ptq1f_gemm(const ptq1f_entry * e, const uint8_t * acts, size_t col_bytes, int64_t m,
                       int64_t r0, int64_t r1, float * dst, int64_t ld) {
    const int64_t k = e->k, k_pad = e->n_chunks * PTQ1F_CHUNK, ng = k / 128;
    for (int64_t t0 = 0; t0 < m; t0 += PTQ1F_COLS) {
        const int nt = (int) (m - t0 < PTQ1F_COLS ? m - t0 : PTQ1F_COLS);
        const int8_t * q[PTQ1F_COLS]; const float * dyv[PTQ1F_COLS]; const float * A[PTQ1F_COLS];
        for (int t = 0; t < nt; ++t) {
            const uint8_t * base = acts + (size_t) (t0 + t) * col_bytes;
            q[t] = (const int8_t *) base; dyv[t] = (const float *) (base + k_pad);
            A[t] = (const float *) (base + 2 * k_pad);
        }
        for (int64_t r = r0; r < r1; ++r) {
            const uint8_t * row = e->packed + r * e->row_stride;
            const ggml_half * sc = (const ggml_half *) (row + e->n_chunks * 64);
            __m512 acc[PTQ1F_COLS];
            for (int t = 0; t < nt; ++t) acc[t] = _mm512_setzero_ps();
            for (int64_t c = 0; c < e->n_chunks; ++c) {
                __m512i x = _mm512_loadu_si512((const void *) (row + c * 64));
                const int64_t e0 = c * PTQ1F_CHUNK;
                const int steps = (int) ((k - e0) >= PTQ1F_CHUNK ? 5 : (k - e0) / 64);
                for (int n = 0; n < steps; ++n) {
                    const int64_t el = e0 + 64 * n;
                    const __m512i codes = ptq1f_decode(&x);
                    const __m512 dx = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(sc[el / 128]));
                    for (int t = 0; t < nt; ++t) {
                        const __m512i a = _mm512_loadu_si512((const void *) (q[t] + el));
                        const __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes, a);
                        const __m512 s = _mm512_mul_ps(_mm512_cvtepi32_ps(d), _mm512_loadu_ps(dyv[t] + el / 4));
                        acc[t] = _mm512_fmadd_ps(s, dx, acc[t]);
                    }
                }
            }
            for (int t = 0; t < nt; ++t) {
                float corr = 0.0f;
                for (int64_t g = 0; g < ng; ++g) corr += GGML_CPU_FP16_TO_FP32(sc[g]) * A[t][g];
                dst[(t0 + t) * ld + r] = _mm512_reduce_add_ps(acc[t]) - corr;
            }
        }
    }
    (void) k_pad;
}

// Multi-token path with K tiling: a row block of <=16 rows walks K in slices of
// PTQ1F_SLICE chunks so the slice of every token's activations (+ expanded scales) stays in
// L1 while it is reused by all 16 rows. Without this, 8 tokens' activations (2 x 17 KB each
// on FFN rows) stream from L2 for every row and the per-token cost dominates.
#define PTQ1F_SLICE 6
#define PTQ1F_RBLK 16
static void ptq1f_gemm_tiled(const ptq1f_entry * e, const uint8_t * acts, size_t col_bytes, int64_t m,
                             int64_t r0, int64_t r1, float * dst, int64_t ld) {
    const int64_t k = e->k, k_pad = e->n_chunks * PTQ1F_CHUNK, ng = k / 128;
    __m512 accbuf[PTQ1F_RBLK][PTQ1F_COLS];
    for (int64_t rb = r0; rb < r1; rb += PTQ1F_RBLK) {
        const int64_t rn = (r1 - rb) < PTQ1F_RBLK ? (r1 - rb) : PTQ1F_RBLK;
        for (int64_t t0 = 0; t0 < m; t0 += PTQ1F_COLS) {
            const int nt = (int) (m - t0 < PTQ1F_COLS ? m - t0 : PTQ1F_COLS);
            const int8_t * q[PTQ1F_COLS]; const float * dyv[PTQ1F_COLS]; const float * A[PTQ1F_COLS];
            for (int t = 0; t < nt; ++t) {
                const uint8_t * base = acts + (size_t) (t0 + t) * col_bytes;
                q[t] = (const int8_t *) base; dyv[t] = (const float *) (base + k_pad);
                A[t] = (const float *) (base + 2 * k_pad);
            }
            for (int64_t i = 0; i < rn; ++i) for (int t = 0; t < nt; ++t) accbuf[i][t] = _mm512_setzero_ps();
            for (int64_t c0 = 0; c0 < e->n_chunks; c0 += PTQ1F_SLICE) {
                const int64_t c1 = (c0 + PTQ1F_SLICE) < e->n_chunks ? (c0 + PTQ1F_SLICE) : e->n_chunks;
                for (int64_t i = 0; i < rn; ++i) {
                    const uint8_t * row = e->packed + (rb + i) * e->row_stride;
                    const ggml_half * sc = (const ggml_half *) (row + e->n_chunks * 64);
                    __m512 acc[PTQ1F_COLS];
                    for (int t = 0; t < nt; ++t) acc[t] = accbuf[i][t];
                    for (int64_t c = c0; c < c1; ++c) {
                        __m512i x = _mm512_loadu_si512((const void *) (row + c * 64));
                        const int64_t e0 = c * PTQ1F_CHUNK;
                        const int steps = (int) ((k - e0) >= PTQ1F_CHUNK ? 5 : (k - e0) / 64);
                        for (int n = 0; n < steps; ++n) {
                            const int64_t el = e0 + 64 * n;
                            const __m512i codes = ptq1f_decode(&x);
                            const __m512 dx = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(sc[el / 128]));
                            for (int t = 0; t < nt; ++t) {
                                const __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), codes,
                                                                      _mm512_loadu_si512((const void *) (q[t] + el)));
                                const __m512 sv = _mm512_mul_ps(_mm512_cvtepi32_ps(d), _mm512_loadu_ps(dyv[t] + el / 4));
                                acc[t] = _mm512_fmadd_ps(sv, dx, acc[t]);
                            }
                        }
                    }
                    for (int t = 0; t < nt; ++t) accbuf[i][t] = acc[t];
                }
            }
            for (int64_t i = 0; i < rn; ++i) {
                const ggml_half * sc = (const ggml_half *) (e->packed + (rb + i) * e->row_stride + e->n_chunks * 64);
                for (int t = 0; t < nt; ++t) {
                    float corr = 0.0f;
                    for (int64_t g = 0; g < ng; ++g) corr += GGML_CPU_FP16_TO_FP32(sc[g]) * A[t][g];
                    dst[(t0 + t) * ld + rb + i] = _mm512_reduce_add_ps(accbuf[i][t]) - corr;
                }
            }
        }
    }
}

// Mode 2: one int8 scale per 128 activations (the weight group size), so both 64-steps of
// a group accumulate in int32 and the float work drops to one fma per group per token.
// Numerics differ from Q8_0 (coarser activation scale): admitted only after a quality check.
// layout per token: int8[k_pad] | f32 dy[k/128] | f32 A[k/128]
static void ptq1f_quantize_col128(const float * x, int64_t k, uint8_t * out) {
    const int64_t k_pad = (k + PTQ1F_CHUNK - 1) / PTQ1F_CHUNK * PTQ1F_CHUNK;
    int8_t * q = (int8_t *) out;
    float * dy = (float *) (out + k_pad);
    float * A = dy + k / 128;
    memset(q, 0, (size_t) k_pad);
    for (int64_t g = 0; g < k / 128; ++g) {
        float amax = 0.0f;
        for (int j = 0; j < 128; ++j) amax = fmaxf(amax, fabsf(x[g * 128 + j]));
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        int sum = 0;
        for (int j = 0; j < 128; ++j) { const int v = (int) roundf(x[g * 128 + j] * id); q[g * 128 + j] = (int8_t) v; sum += v; }
        dy[g] = d; A[g] = d * (float) sum;
    }
}

static void ptq1f_gemm128(const ptq1f_entry * e, const uint8_t * acts, size_t col_bytes, int64_t m,
                          int64_t r0, int64_t r1, float * dst, int64_t ld) {
    const int64_t k = e->k, k_pad = e->n_chunks * PTQ1F_CHUNK, ng = k / 128;
    for (int64_t t0 = 0; t0 < m; t0 += PTQ1F_COLS) {
        const int nt = (int) (m - t0 < PTQ1F_COLS ? m - t0 : PTQ1F_COLS);
        const int8_t * q[PTQ1F_COLS]; const float * dy[PTQ1F_COLS]; const float * A[PTQ1F_COLS];
        for (int t = 0; t < nt; ++t) {
            const uint8_t * base = acts + (size_t) (t0 + t) * col_bytes;
            q[t] = (const int8_t *) base; dy[t] = (const float *) (base + k_pad); A[t] = dy[t] + ng;
        }
        for (int64_t r = r0; r < r1; ++r) {
            const uint8_t * row = e->packed + r * e->row_stride;
            const ggml_half * sc = (const ggml_half *) (row + e->n_chunks * 64);
            __m512 acc[PTQ1F_COLS]; __m512i ai[PTQ1F_COLS];
            for (int t = 0; t < nt; ++t) { acc[t] = _mm512_setzero_ps(); ai[t] = _mm512_setzero_si512(); }
            for (int64_t c = 0; c < e->n_chunks; ++c) {
                __m512i x = _mm512_loadu_si512((const void *) (row + c * 64));
                const int64_t e0 = c * PTQ1F_CHUNK;
                const int steps = (int) ((k - e0) >= PTQ1F_CHUNK ? 5 : (k - e0) / 64);
                for (int n = 0; n < steps; ++n) {
                    const int64_t el = e0 + 64 * n;
                    const __m512i codes = ptq1f_decode(&x);
                    for (int t = 0; t < nt; ++t)
                        ai[t] = _mm512_dpbusd_epi32(ai[t], codes, _mm512_loadu_si512((const void *) (q[t] + el)));
                    if ((el >> 6) & 1) {                       // second half of a 128-group: flush
                        const int64_t g = el >> 7;
                        const float dx = GGML_CPU_FP16_TO_FP32(sc[g]);
                        for (int t = 0; t < nt; ++t) {
                            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ai[t]), _mm512_set1_ps(dx * dy[t][g]), acc[t]);
                            ai[t] = _mm512_setzero_si512();
                        }
                    }
                }
            }
            for (int t = 0; t < nt; ++t) {
                float corr = 0.0f;
                for (int64_t g = 0; g < ng; ++g) corr += GGML_CPU_FP16_TO_FP32(sc[g]) * A[t][g];
                dst[(t0 + t) * ld + r] = _mm512_reduce_add_ps(acc[t]) - corr;
            }
        }
    }
}

#include "ptq1-panel.inc"

#ifndef PTQ1F_NO_GGML
// returns true if handled
static bool ptq1f_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    if (!ptq1f_applies(src0, src1)) return false;
#if PTQ1A_OK
    if (ptq1a_on()) return ptq1a_mul_mat(params, dst);
#endif
    if (ptq1f_mode() >= 3) return ptq1p_mul_mat(params, dst);
    const int ith = params->ith, nth = params->nth;
    const int64_t k = src0->ne[0], rows = src0->ne[1], m = src1->ne[1];

    // all threads read the registry before anyone can write it (writes happen after a barrier)
    ptq1f_entry * e = NULL;
    pthread_mutex_lock(&ptq1f_mu);
    for (ptq1f_entry * it = ptq1f_list; it; it = it->next) if (it->src == src0->data) { e = it; break; }
    pthread_mutex_unlock(&ptq1f_mu);
    if (!e) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            ptq1f_entry * ne = calloc(1, sizeof(*ne));
            ne->src = src0->data; ne->rows = rows; ne->k = k;
            ne->n_chunks = (k + PTQ1F_CHUNK - 1) / PTQ1F_CHUNK;
            ne->row_stride = (ne->n_chunks * 64 + (k / 128) * 2 + 63) / 64 * 64;
            ne->packed = aligned_alloc(64, (size_t) (ne->row_stride * rows));
            pthread_mutex_lock(&ptq1f_mu); ne->next = ptq1f_list; ptq1f_list = ne; pthread_mutex_unlock(&ptq1f_mu);
        }
        ggml_barrier(params->threadpool);
        pthread_mutex_lock(&ptq1f_mu);
        for (ptq1f_entry * it = ptq1f_list; it; it = it->next) if (it->src == src0->data) { e = it; break; }
        pthread_mutex_unlock(&ptq1f_mu);
        const int64_t per = (rows + nth - 1) / nth;
        ptq1f_repack_rows(e, PTQ1F_MIN(rows, ith * per), PTQ1F_MIN(rows, (ith + 1) * per));
    }

    const size_t col_bytes = ptq1f_col_bytes(k);
    uint8_t * acts = (uint8_t *) (((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);
    for (int64_t t = ith; t < m; t += nth)
        (ptq1f_mode() == 2 ? ptq1f_quantize_col128 : ptq1f_quantize_col)(
            (const float *) ((const char *) src1->data + t * src1->nb[1]), k, acts + t * col_bytes);
    ggml_barrier(params->threadpool);

    // split rows into 16-row chunks handed out round-robin
    const int64_t blk = 16, nblk = (rows + blk - 1) / blk;
    for (int64_t b = ith; b < nblk; b += nth)
        (ptq1f_mode() == 2 ? ptq1f_gemm128 : (m > 1 ? ptq1f_gemm_tiled : ptq1f_gemm))(e, acts, col_bytes, m, b * blk, PTQ1F_MIN(rows, (b + 1) * blk),
                   (float *) dst->data, dst->nb[1] / sizeof(float));
    return true;
}
#endif
#else
#if PTQ1A_OK && !defined(PTQ1F_NO_GGML)
// no AVX-512 kernels in this build: the 256-bit kernels of ptq1-avx2.inc (AVX2, with AVX-VNNI when the build has it).
// GGML_PTQ1_AVX2=0 or GGML_PTQ1_FAST=0 leaves the matmul to the fork's own kernel.
static bool ptq1f_applies(const struct ggml_tensor * src0, const struct ggml_tensor * src1) {
    return src0->type == GGML_TYPE_PTQ1_0 && src1->type == GGML_TYPE_F32 &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && src0->ne[0] % 128 == 0 && ptq1a_on();
}
static bool ptq1f_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (!ptq1f_applies(dst->src[0], dst->src[1])) return false;
    return ptq1a_mul_mat(params, dst);
}
static size_t ptq1f_wsize(const struct ggml_tensor * a, const struct ggml_tensor * b) { return ptq1a_wsize(a, b); }
#else
static bool ptq1f_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) { (void) params; (void) dst; return false; }
static size_t ptq1f_wsize(const struct ggml_tensor * a, const struct ggml_tensor * b) { (void) a; (void) b; return 0; }
static bool ptq1f_applies(const struct ggml_tensor * a, const struct ggml_tensor * b) { (void) a; (void) b; return false; }
#endif
static inline int ptq1p_k7(void) { return 0; }
static const int ptq1p_k7_la = 0;
static void ptq1p_k7_spin(void) { }
static inline void ptq1p_k7_drain(int ith) { (void) ith; }
#endif
