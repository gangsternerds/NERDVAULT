// ptq1-avx2: the PTQ1_0 ternary matmul for CPUs WITHOUT AVX-512. AVX2 + FMA + F16C only, plus AVX-VNNI (VEX vpdpbusd,
// Intel Core 12th gen and newer, Zen 5) when the build has it.
// A build without the AVX-512 kernels of ptq1-fast.inc runs this by default (GGML_PTQ1_AVX2=0 or GGML_PTQ1_FAST=0 = the
// fork's own kernel). A build with them keeps them, and GGML_PTQ1_AVX2=1 selects this family instead (same-machine A/B).
//
// Decode without vpermb / vpmultishiftqb. A 16-bit word holds 10 trits as a base-3 number q < 3^10 = 59049, stored as
// w = ceil(q * 65536 / 59049). Then (w * 9) >> 16 is the top base-9 digit (= two trits, 0..8) and w * 9 mod 65536 is the
// encoding of the rest, so one decode stage is one vpmulhuw + one vpmullw per 16 words, both plain AVX2 (exact for every
// q, checked exhaustively in tests/kernel_avx2.c). Two stages' words are packed to 32 bytes of digits with vpackuswb,
// and two vpshufb (v % 3, v / 3) split each digit into its two trits, each landing as a full 32-byte code vector laid
// out exactly as vpdpbusd wants it: lane l = row l, 4 consecutive k per lane. Per 64 trits: 2 vpmulhuw + 2 vpmullw +
// 1 vpackuswb + 2 vpshufb = 7 uops on three ymm ports (Ice Lake and later run ymm ops on p0/p1/p5, zmm on two), vs the
// 6-uop ptq1f_decode per 64 trits on two ports. Density 10 trits / 16 bits = 1.6 bpw, the same bytes as mode 4's
// base-243 layout and 5 % fewer than the VBMI layout (1.684), so the repacked 27B is ~5.6 GB.
//
// Layout: panels of 8 rows (one ymm of int32 sums = 8 rows x 4 k). A panel is nck 64-byte chunks (32 words) then the
// fp16 scales, 8 per 128-k group. Chunk c covers k in [40c, 40c + 40): word i (after the vpackuswb lane shuffle, see
// ptq1a_wi) belongs to row i/4, j = i%4, and its stage s digit is trit(k = 40c + 8s + j) + 3 trit(k = 40c + 8s + 4 + j).
// So stage s yields A = codes for k = 40c + 8s .. +3 and B = codes for 40c + 8s + 4 .. +7. k is padded to 160
// (= lcm(40, 32)): a superblock of 4 chunks = 20 stages = 5 Q8_0 blocks, the flush points fixed at compile time.
//
// Dot product: vpdpbusd ymm (AVX-VNNI, or the EVEX form on an AVX-512 build) when available; otherwise vpmaddubsw +
// vpaddw into an int16 accumulator per block (|pair sum| <= 508, |block sum| <= 8 x 508: no saturation) and one
// vpmaddwd per block to widen. Either way every 32-k block sum is an exact int32, and the per-block float flush
// (cvt, dx * dy, fma, in block order from 0) is the one ptq1v_panel / ptq1k4_tile do, so the outputs are bit-identical
// to the AVX-512 paths (tests/kernel_avx2.c checks every path against a scalar reference).
//
// Paths (16 ymm registers, not 32, so the tiles are re-cut):
//   M = 1    ptq1a_panels: PTQ1A_NP panels interleaved per call (two weight streams per core, K5), 2 int chains each,
//            decode in registers, software prefetch PTQ1A_PF bytes ahead per stream.
//   M = 2..8 ptq1a_panels_m: one panel x 4 tokens per pass, decode and accumulators in registers (a speculation verify
//            batch, or the streams of --multi). GGML_PTQ1_AVX2_SMALLM=n moves the limit (0 = always ptq1a_gemm).
//   M > 8    ptq1a_gemm: K4's structure -- decode a k-chunk of PTQ1A_KC superblocks of 2 panels into an L1
//            buffer of code vectors once, then token tiles of PTQ1A_NT against it (per k-step: 2 code loads + NT
//            broadcasts + 2 NT dots, no table registers); the float accumulator of (token, 8 rows) is carried in dst
//            between chunks. The thread scheduling (shared quantize for M <= 8, dynamic panel chunks) is ptq1p_mul_mat's.
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define PTQ1A_OK 1
#include <immintrin.h>
#include <pthread.h>
#if defined(PTQ1A_FORCE_MADD)             // test builds: the vpmaddubsw dot on a VNNI machine (identity of the two dots)
#define PTQ1A_VNNI 0
#elif defined(__AVXVNNI__)
#define PTQ1A_VNNI 1
#define PTQ1A_DPBUSD(acc, u, s) _mm256_dpbusd_avx_epi32((acc), (u), (s))
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define PTQ1A_VNNI 1
#define PTQ1A_DPBUSD(acc, u, s) _mm256_dpbusd_epi32((acc), (u), (s))
#else
#define PTQ1A_VNNI 0
#endif

#define PTQ1A_ROWS 8
#define PTQ1A_KC_CHUNK 40            // k per 64-byte chunk
#define PTQ1A_SB 4                   // chunks per superblock (160 k = 5 Q8_0 blocks)
#ifndef PTQ1A_NP
#define PTQ1A_NP 2                   // panels interleaved per M = 1 kernel call
#endif
#ifndef PTQ1A_PF
#define PTQ1A_PF 1024                // M = 1 prefetch distance (bytes ahead per stream); 0 = none
#endif
#ifndef PTQ1A_NT
#define PTQ1A_NT 4                   // tokens per tile in ptq1a_gemm (2 panels x NT int accumulators + 2 codes + 1 bcast + 2 dxv <= 16 ymm)
#endif
#ifndef PTQ1A_KC
#define PTQ1A_KC 8                   // superblocks per k-chunk in ptq1a_gemm (2 panels x KC x 1280 B of codes in L1)
#endif
#ifndef PTQ1A_MB
#define PTQ1A_MB 512                 // tokens per token block (bounds the L2 activation chunk: MB x 160 KC bytes)
#endif

// the default: on where the AVX-512 kernels are not compiled in (the condition is ptq1-fast.inc's), off where they are
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#define PTQ1A_DEFAULT 0
#else
#define PTQ1A_DEFAULT 1
#endif
static int ptq1a_on_cache = -1;
static inline int ptq1a_on(void) {
    if (ptq1a_on_cache < 0) {
        const char * s = getenv("GGML_PTQ1_AVX2");
        int on = s ? atoi(s) != 0 : PTQ1A_DEFAULT;
#if PTQ1A_DEFAULT
        const char * f = getenv("GGML_PTQ1_FAST");          // GGML_PTQ1_FAST=0 means the fork's own kernel on every build
        if (f && atoi(f) == 0) on = 0;
#endif
        ptq1a_on_cache = on;
    }
    return ptq1a_on_cache;
}

// word index of output byte i after vpackuswb(lo = words 0..15, hi = words 16..31)
static inline int ptq1a_wi(int i) { return i < 8 ? i : i < 16 ? i + 8 : i < 24 ? i - 8 : i; }

static void ptq1a_init_entry(ptq1p_entry * e, const void * src, int64_t rows, int64_t k) {
    e->src = src; e->rows = rows; e->k = k; e->nsb = 0; e->n36 = 0;
    e->kpad = (k + PTQ1P_KPAD - 1) / PTQ1P_KPAD * PTQ1P_KPAD;
    e->npanel = (rows + PTQ1A_ROWS - 1) / PTQ1A_ROWS;
    e->nck = e->kpad / PTQ1A_KC_CHUNK;
    e->ngp = (e->kpad + 127) / 128;
    e->panel_stride = (e->nck * 64 + e->ngp * PTQ1A_ROWS * 2 + 63) / 64 * 64;
}
static inline int64_t ptq1a_data_bytes(const ptq1p_entry * e) { return e->nck * 64; }

// codes: PTQ1A_ROWS rows of kpad trits {0,1,2} (row stride kpad) -> the panel's chunks at dst
static void ptq1a_pack_codes(const uint8_t * codes, int64_t kpad, uint8_t * dst) {
    for (int64_t c = 0; c < kpad / PTQ1A_KC_CHUNK; ++c)
        for (int i = 0; i < 32; ++i) {
            const int r = i / 4, j = i % 4;
            uint32_t q = 0;
            for (int s = 0; s < 5; ++s) {
                const uint8_t * row = codes + r * kpad + c * PTQ1A_KC_CHUNK + 8 * s + j;
                q = q * 9 + row[0] + 3 * row[4];
            }
            const uint16_t w = (uint16_t) (((uint64_t) q * 65536 + 59048) / 59049);
            memcpy(dst + c * 64 + 2 * ptq1a_wi(i), &w, 2);
        }
}

static void ptq1a_repack_panels(ptq1p_entry * e, int64_t p0, int64_t p1) {
    const int64_t nb = e->k / 128;
    uint8_t * codes = malloc((size_t) (PTQ1A_ROWS * e->kpad));
    for (int64_t p = p0; p < p1; ++p) {
        memset(codes, 1, (size_t) (PTQ1A_ROWS * e->kpad));             // trit 1 == weight 0
        uint8_t * dst = e->packed + p * e->panel_stride;
        ggml_half * sc = (ggml_half *) (dst + ptq1a_data_bytes(e));
        memset(sc, 0, (size_t) (e->ngp * PTQ1A_ROWS * 2));
        for (int r = 0; r < PTQ1A_ROWS; ++r) {
            const int64_t row = p * PTQ1A_ROWS + r;
            if (row >= e->rows) continue;
            const block_ptq1_0 * src = (const block_ptq1_0 *) e->src + row * nb;
            for (int64_t b = 0; b < nb; ++b) {
                ptq1f_unpack_block(src + b, codes + r * e->kpad + b * 128);
                sc[b * PTQ1A_ROWS + r] = src[b].d;
            }
        }
        ptq1a_pack_codes(codes, e->kpad, dst);
    }
    free(codes);
}

// vpshufb tables, per 128-bit lane: digit v (0..8) -> v % 3 and v / 3. Constants, not a load-time constructor (a
// --portable build loads every CPU variant of this library before it knows which one the CPU can run).
static const uint8_t ptq1a_tab[2][32] __attribute__((aligned(32))) = {
    {   0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   0,   0,   0,   0,   0,   0,
        0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   0,   0,   0,   0,   0,   0 },
    {   0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   0,   0,   0,   0 }
};

// The Q8_0 quantizer of ptq1p_quantize_col_part_kp in AVX2: same q, same -sum(q), same fp16-rounded d per block.
// roundf (half away from zero) is reproduced as trunc, then +-1 where |v - trunc(v)| >= 0.5 (exact in float).
static void ptq1a_quantize_part(const float * x, int64_t k, int64_t kpad, uint8_t * out, int64_t b0, int64_t b1, int neg_asum) {
    int8_t * q = (int8_t *) out;
    float * dy = (float *) (out + kpad);
    int32_t * asum = (int32_t *) (dy + kpad / 32);
    if (b0 == 0) {
        memset(q + k, 0, (size_t) (kpad - k));
        for (int64_t b = k / 32; b < kpad / 32; ++b) { dy[b] = 0.0f; asum[b] = 0; }
    }
    const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f), mone = _mm256_set1_ps(-1.0f), zero = _mm256_setzero_ps();
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i lanes = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t b = b0; b < b1; ++b) {
        __m256 xv[4], am = zero;
        for (int h = 0; h < 4; ++h) { xv[h] = _mm256_loadu_ps(x + b * 32 + 8 * h); am = _mm256_max_ps(am, _mm256_and_ps(xv[h], absm)); }
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4)); m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float amax = _mm_cvtss_f32(m4);
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        const __m256 idv = _mm256_set1_ps(id);
        __m256i qi[4], sv = _mm256_setzero_si256();
        for (int h = 0; h < 4; ++h) {
            const __m256 v = _mm256_mul_ps(xv[h], idv);
            const __m256 r = _mm256_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            const __m256 away = _mm256_cmp_ps(_mm256_and_ps(_mm256_sub_ps(v, r), absm), half, _CMP_GE_OQ);
            const __m256 sg = _mm256_blendv_ps(one, mone, _mm256_cmp_ps(v, zero, _CMP_LT_OQ));
            qi[h] = _mm256_cvttps_epi32(_mm256_add_ps(r, _mm256_and_ps(away, sg)));
            sv = _mm256_add_epi32(sv, qi[h]);
        }
        const __m256i p16a = _mm256_packs_epi32(qi[0], qi[1]), p16b = _mm256_packs_epi32(qi[2], qi[3]);
        const __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), lanes);
        _mm256_storeu_si256((__m256i *) (q + b * 32), p8);
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sv), _mm256_extracti128_si256(sv, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4e)); s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xb1));
        const int s = _mm_cvtsi128_si32(s4);
        dy[b] = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(d));
        asum[b] = neg_asum ? -s : s;
    }
}

// ---- the dot-product step and the block flush, shared by the kernels below ---------------------------------------
#if PTQ1A_VNNI
#define PTQ1A_ACC_INIT(nas)        (nas)                                        // chain 0 starts at -sum(q)
#define PTQ1A_DOT(acc, codes, bq)  (acc) = PTQ1A_DPBUSD((acc), (codes), (bq))
#define PTQ1A_SUM(a0, a1, nas)     _mm256_add_epi32((a0), (a1))
#else
#define PTQ1A_ACC_INIT(nas)        _mm256_setzero_si256()
#define PTQ1A_DOT(acc, codes, bq)  (acc) = _mm256_add_epi16((acc), _mm256_maddubs_epi16((codes), (bq)))
#define PTQ1A_SUM(a0, a1, nas)     _mm256_add_epi32(_mm256_add_epi32(_mm256_madd_epi16((a0), ones), _mm256_madd_epi16((a1), ones)), (nas))
#endif
#if PTQ1A_VNNI
#define PTQ1A_SUM1(a, nas) (a)
#else
#define PTQ1A_SUM1(a, nas) _mm256_add_epi32(_mm256_madd_epi16((a), ones), (nas))
#endif
#define PTQ1A_FLUSH(accf, a0, a1, nas, dxv, dyb) (accf) = _mm256_fmadd_ps(_mm256_cvtepi32_ps(PTQ1A_SUM(a0, a1, nas)), _mm256_mul_ps((dxv), (dyb)), (accf))

// one chunk's two word vectors at stage s: digits (packed), then the A and B code vectors
#define PTQ1A_STAGE(w0, w1, s, A, B) do {                                                                        \
        const __m256i h0_ = _mm256_mulhi_epu16((w0), nine), h1_ = _mm256_mulhi_epu16((w1), nine);                \
        if ((s) < 4) { (w0) = _mm256_mullo_epi16((w0), nine); (w1) = _mm256_mullo_epi16((w1), nine); }           \
        const __m256i d_ = _mm256_packus_epi16(h0_, h1_);                                                        \
        (A) = _mm256_shuffle_epi8(t0, d_); (B) = _mm256_shuffle_epi8(t1, d_);                                     \
    } while (0)
#define PTQ1A_STAGE_DECL

// M = 1: NP panels (panel, panel + stride, ...) x one token, chunk by chunk, each panel 2 int chains (A and B).
// Per panel per 32-k block: 4 stages x (7 decode + 2 dot) + 3 flush uops, int sums exact, flush in block order.
static inline __attribute__((always_inline)) void ptq1a_panel_np(
        const ptq1p_entry * e, const uint8_t * panel, const uint8_t * acts, const int NP, const int PF, float * dst, int64_t row0) {
    const int64_t kpad = e->kpad;
    const int8_t * q = (const int8_t *) acts; const float * dy = (const float *) (acts + kpad); const int32_t * as = (const int32_t *) (dy + kpad / 32);
    const __m256i nine = _mm256_set1_epi16(9), t0 = _mm256_load_si256((const __m256i *) ptq1a_tab[0]), t1 = _mm256_load_si256((const __m256i *) ptq1a_tab[1]);
#if !PTQ1A_VNNI
    const __m256i ones = _mm256_set1_epi16(1);
#endif
    const uint8_t * P[PTQ1A_NP]; const ggml_half * sc[PTQ1A_NP];
    __m256 accf[PTQ1A_NP], dxv[PTQ1A_NP]; __m256i a0[PTQ1A_NP], a1[PTQ1A_NP];
    for (int p = 0; p < NP; ++p) {
        P[p] = panel + p * e->panel_stride; sc[p] = (const ggml_half *) (P[p] + ptq1a_data_bytes(e));
        accf[p] = _mm256_setzero_ps(); dxv[p] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) sc[p]));
        a0[p] = PTQ1A_ACC_INIT(_mm256_set1_epi32(as[0])); a1[p] = _mm256_setzero_si256();
    }
    for (int64_t sb = 0; sb < e->nck / PTQ1A_SB; ++sb) {
        const int64_t c0 = sb * PTQ1A_SB, b0 = sb * 5;
#pragma GCC unroll 4
        for (int c = 0; c < PTQ1A_SB; ++c) {
            for (int p = 0; p < NP; ++p) {
                const uint8_t * ch = P[p] + (c0 + c) * 64;
                if (PF) _mm_prefetch((const char *) ch + PF, _MM_HINT_T0);
                __m256i w0 = _mm256_loadu_si256((const __m256i *) ch), w1 = _mm256_loadu_si256((const __m256i *) (ch + 32));
                PTQ1A_STAGE_DECL
#pragma GCC unroll 5
                for (int s = 0; s < 5; ++s) {
                    __m256i A, B;
                    PTQ1A_STAGE(w0, w1, s, A, B);
                    const int64_t k0 = (c0 + c) * PTQ1A_KC_CHUNK + 8 * s;
                    PTQ1A_DOT(a0[p], A, _mm256_set1_epi32(*(const int32_t *) (q + k0)));
                    PTQ1A_DOT(a1[p], B, _mm256_set1_epi32(*(const int32_t *) (q + k0 + 4)));
                    if ((c * 5 + s) % 4 == 3) {                                   // end of a 32-k block
                        const int64_t b = b0 + (c * 5 + s) / 4;
                        const __m256i nas = _mm256_set1_epi32(as[b]), nas1 = _mm256_set1_epi32(as[b + 1]);
                        PTQ1A_FLUSH(accf[p], a0[p], a1[p], nas, dxv[p], _mm256_set1_ps(dy[b]));
                        if (((b + 1) & 3) == 0) dxv[p] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (sc[p] + ((b + 1) >> 2) * PTQ1A_ROWS)));
                        a0[p] = PTQ1A_ACC_INIT(nas1); a1[p] = _mm256_setzero_si256();
                        (void) nas; (void) nas1;
                    }
                }
            }
        }
    }
    for (int p = 0; p < NP; ++p) {
        const int64_t r0 = row0 + p * PTQ1A_ROWS;
        const int64_t nr = (e->rows - r0) < PTQ1A_ROWS ? (e->rows - r0) : PTQ1A_ROWS;
        if (nr <= 0) continue;
        if (nr == PTQ1A_ROWS) _mm256_storeu_ps(dst + r0, accf[p]);
        else { float tmp[8]; _mm256_storeu_ps(tmp, accf[p]); memcpy(dst + r0, tmp, (size_t) nr * 4); }
    }
}
// panels [p0, p1) in groups of PTQ1A_NP, the rest one at a time
static void ptq1a_panels(const ptq1p_entry * e, const uint8_t * acts, float * dst, int64_t p0, int64_t p1) {
    int64_t p = p0;
    for (; p + PTQ1A_NP <= p1; p += PTQ1A_NP) ptq1a_panel_np(e, e->packed + p * e->panel_stride, acts, PTQ1A_NP, PTQ1A_PF, dst, p * PTQ1A_ROWS);
    for (; p < p1; ++p) ptq1a_panel_np(e, e->packed + p * e->panel_stride, acts, 1, PTQ1A_PF, dst, p * PTQ1A_ROWS);
}

// M = 2..8 (a speculation verify batch, the streams of --multi): one panel x NT <= 4 tokens, everything in registers. Each stage is decoded once
// and every token adds its two dots to ONE integer accumulator (int16 form: a 32-k block is 8 pair sums of at most 508
// per lane, no saturation), so a 4-token pass costs about twice a 1-token pass (through ptq1a_gemm it costs 4 times). Same exact int32
// block sums and the same per-block flush as the kernels above and below: bit-identical to both.
static inline __attribute__((always_inline)) void ptq1a_panel_nt(
        const ptq1p_entry * e, const uint8_t * panel, const uint8_t * acts, size_t cb, const int NT, const int PF, float * dst, int64_t ld, int64_t row0) {
    const int64_t kpad = e->kpad;
    const int8_t * q[PTQ1A_NT]; const float * dy[PTQ1A_NT]; const int32_t * as[PTQ1A_NT];
    for (int t = 0; t < NT; ++t) {
        const uint8_t * base = acts + (size_t) t * cb;
        q[t] = (const int8_t *) base; dy[t] = (const float *) (base + kpad); as[t] = (const int32_t *) (dy[t] + kpad / 32);
    }
    const __m256i nine = _mm256_set1_epi16(9), t0 = _mm256_load_si256((const __m256i *) ptq1a_tab[0]), t1 = _mm256_load_si256((const __m256i *) ptq1a_tab[1]);
#if !PTQ1A_VNNI
    const __m256i ones = _mm256_set1_epi16(1);
#endif
    const ggml_half * sc = (const ggml_half *) (panel + ptq1a_data_bytes(e));
    __m256 accf[PTQ1A_NT]; __m256i acc[PTQ1A_NT];
    __m256 dxv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) sc));
    for (int t = 0; t < NT; ++t) { accf[t] = _mm256_setzero_ps(); acc[t] = PTQ1A_ACC_INIT(_mm256_set1_epi32(as[t][0])); }
    for (int64_t sb = 0; sb < e->nck / PTQ1A_SB; ++sb) {
        const int64_t c0 = sb * PTQ1A_SB, b0 = sb * 5;
#pragma GCC unroll 4
        for (int c = 0; c < PTQ1A_SB; ++c) {
            const uint8_t * ch = panel + (c0 + c) * 64;
            if (PF) _mm_prefetch((const char *) ch + PF, _MM_HINT_T0);
            __m256i w0 = _mm256_loadu_si256((const __m256i *) ch), w1 = _mm256_loadu_si256((const __m256i *) (ch + 32));
#pragma GCC unroll 5
            for (int s = 0; s < 5; ++s) {
                __m256i A, B;
                PTQ1A_STAGE(w0, w1, s, A, B);
                const int64_t k0 = (c0 + c) * PTQ1A_KC_CHUNK + 8 * s;
                for (int t = 0; t < NT; ++t) {
                    PTQ1A_DOT(acc[t], A, _mm256_set1_epi32(*(const int32_t *) (q[t] + k0)));
                    PTQ1A_DOT(acc[t], B, _mm256_set1_epi32(*(const int32_t *) (q[t] + k0 + 4)));
                }
                if ((c * 5 + s) % 4 == 3) {                                   // end of a 32-k block
                    const int64_t b = b0 + (c * 5 + s) / 4;
                    for (int t = 0; t < NT; ++t) {
                        const __m256i nas = _mm256_set1_epi32(as[t][b]);
                        accf[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(PTQ1A_SUM1(acc[t], nas)), _mm256_mul_ps(dxv, _mm256_set1_ps(dy[t][b])), accf[t]);
                        acc[t] = PTQ1A_ACC_INIT(_mm256_set1_epi32(as[t][b + 1]));
                        (void) nas;
                    }
                    if (((b + 1) & 3) == 0) dxv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (sc + ((b + 1) >> 2) * PTQ1A_ROWS)));
                }
            }
        }
    }
    const int64_t nr = (e->rows - row0) < PTQ1A_ROWS ? (e->rows - row0) : PTQ1A_ROWS;
    for (int t = 0; t < NT; ++t) {
        if (nr == PTQ1A_ROWS) _mm256_storeu_ps(dst + t * ld + row0, accf[t]);
        else { float tmp[8]; _mm256_storeu_ps(tmp, accf[t]); memcpy(dst + t * ld + row0, tmp, (size_t) nr * 4); }
    }
}
// panels [p0, p1) x m tokens in tiles of up to PTQ1A_NT (m <= 4 is one pass over each panel, 5..8 two)
static void ptq1a_panels_m(const ptq1p_entry * e, const uint8_t * acts, size_t cb, int64_t m, int64_t p0, int64_t p1, float * dst, int64_t ld) {
    for (int64_t p = p0; p < p1; ++p) {
        const uint8_t * panel = e->packed + p * e->panel_stride;
        for (int64_t t0 = 0; t0 < m; t0 += PTQ1A_NT) {
            const uint8_t * a = acts + (size_t) t0 * cb; float * d = dst + t0 * ld;
            switch (m - t0 < PTQ1A_NT ? (int) (m - t0) : PTQ1A_NT) {
                case 1: ptq1a_panel_nt(e, panel, a, cb, 1, PTQ1A_PF, d, ld, p * PTQ1A_ROWS); break;
                case 2: ptq1a_panel_nt(e, panel, a, cb, 2, PTQ1A_PF, d, ld, p * PTQ1A_ROWS); break;
                case 3: ptq1a_panel_nt(e, panel, a, cb, 3, PTQ1A_PF, d, ld, p * PTQ1A_ROWS); break;
                default: ptq1a_panel_nt(e, panel, a, cb, PTQ1A_NT, PTQ1A_PF, d, ld, p * PTQ1A_ROWS); break;
            }
        }
    }
}

// one chunk (40 k) -> 10 code vectors: out[2s] = A (k = 40c + 8s..), out[2s + 1] = B
static inline __attribute__((always_inline)) void ptq1a_decode_chunk(const uint8_t * ch, __m256i * out) {
    const __m256i nine = _mm256_set1_epi16(9), t0 = _mm256_load_si256((const __m256i *) ptq1a_tab[0]), t1 = _mm256_load_si256((const __m256i *) ptq1a_tab[1]);
    __m256i w0 = _mm256_loadu_si256((const __m256i *) ch), w1 = _mm256_loadu_si256((const __m256i *) (ch + 32));
    PTQ1A_STAGE_DECL
#pragma GCC unroll 5
    for (int s = 0; s < 5; ++s) PTQ1A_STAGE(w0, w1, s, out[2 * s], out[2 * s + 1]);
}

// 2 panels x NT tokens x nblk blocks of decoded codes (codes[0..] panel 0, codes[cstride..] panel 1; 8 vectors per
// block). q[t] points at the tokens' activations pre-offset to the chunk's first k; the float accumulator of (panel,
// token) lives in dst[p] + t * ld (zeroed by first). msk[p]: row mask of a partial last panel (NULL = all 8 rows).
static inline __attribute__((always_inline)) void ptq1a_tile(
        const __m256i * codes, int cstride, int nblk, const ggml_half * const * sc, int64_t b0, const int np,
        const int8_t * const * q, const float * const * dy, const int32_t * const * as,
        float * const * dst, int64_t ld, const __m256i * const * msk, int first, const int NT) {
#if !PTQ1A_VNNI
    const __m256i ones = _mm256_set1_epi16(1);
#endif
    __m256i acc[2][PTQ1A_NT]; __m256 dxv[2];
    for (int p = 0; p < 2; ++p) {
        for (int t = 0; t < NT; ++t) acc[p][t] = PTQ1A_ACC_INIT(_mm256_set1_epi32(as[t][b0]));
        dxv[p] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (sc[p < np ? p : 0] + (b0 >> 2) * PTQ1A_ROWS)));
    }
    for (int b = 0; b < nblk; ++b) {
#pragma GCC unroll 8
        for (int s = 0; s < 8; ++s) {
            const __m256i c0 = codes[8 * b + s], c1 = codes[cstride + 8 * b + s];
            for (int t = 0; t < NT; ++t) {
                const __m256i a = _mm256_set1_epi32(*(const int32_t *) (q[t] + 32 * b + 4 * s));
                PTQ1A_DOT(acc[0][t], c0, a);
                PTQ1A_DOT(acc[1][t], c1, a);
            }
        }
        const int64_t gb = b0 + b;
        for (int t = 0; t < NT; ++t) {
            const __m256 dyv = _mm256_set1_ps(dy[t][gb]);
            const __m256i nas = _mm256_set1_epi32(as[t][gb]), nas1 = _mm256_set1_epi32(as[t][gb + 1]);
            for (int p = 0; p < np; ++p) {
                float * d = dst[p] + t * ld;
                __m256 accf = (first && b == 0) ? _mm256_setzero_ps() : (msk[p] ? _mm256_maskload_ps(d, *msk[p]) : _mm256_loadu_ps(d));
                accf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(PTQ1A_SUM1(acc[p][t], nas)), _mm256_mul_ps(dxv[p], dyv), accf);
                if (msk[p]) _mm256_maskstore_ps(d, *msk[p], accf); else _mm256_storeu_ps(d, accf);
            }
            for (int p = 0; p < 2; ++p) acc[p][t] = PTQ1A_ACC_INIT(nas1);
            (void) nas; (void) nas1;
        }
        if (((gb + 1) & 3) == 0) for (int p = 0; p < 2; ++p) dxv[p] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (sc[p < np ? p : 0] + ((gb + 1) >> 2) * PTQ1A_ROWS)));
    }
}

// panels [p0, p1) x all m tokens (M >= 2): per token block, per k-chunk of KC superblocks, per panel pair: decode the
// pair's chunk into L1 once, prefetch the next pair's chunk, then token tiles of NT against it.
static void ptq1a_gemm(const ptq1p_entry * e, const uint8_t * acts, size_t cb, int64_t m, int64_t p0, int64_t p1, float * dst, int64_t ld) {
    const int KC = PTQ1A_KC, cstride = KC * PTQ1A_SB * 10;
    __m256i codes[2 * PTQ1A_KC * PTQ1A_SB * 10] __attribute__((aligned(64)));
    const int64_t kpad = e->kpad, nsbt = e->nck / PTQ1A_SB;
    for (int64_t m0 = 0; m0 < m; m0 += PTQ1A_MB) {
        const int64_t m1 = m0 + PTQ1A_MB < m ? m0 + PTQ1A_MB : m;
        for (int64_t sb = 0; sb < nsbt; sb += KC) {
            const int64_t sb1 = sb + KC < nsbt ? sb + KC : nsbt;
            const int nblk = (int) ((sb1 - sb) * 5), nck = (int) ((sb1 - sb) * PTQ1A_SB);
            const int64_t choff = sb * PTQ1A_SB * 64, s0 = sb * PTQ1A_SB * PTQ1A_KC_CHUNK, b0 = sb * 5;
            for (int64_t p = p0; p < p1; p += 2) {
                const int np = (int) (p1 - p < 2 ? p1 - p : 2);
                const ggml_half * sc[2]; float * d[2]; __m256i mk[2]; const __m256i * msk[2];
                for (int i = 0; i < np; ++i) {
                    const uint8_t * panel = e->packed + (p + i) * e->panel_stride;
                    for (int c = 0; c < nck; ++c) ptq1a_decode_chunk(panel + choff + c * 64, codes + i * cstride + c * 10);
                    sc[i] = (const ggml_half *) (panel + ptq1a_data_bytes(e));
                    const int64_t row0 = (p + i) * PTQ1A_ROWS;
                    const int nr = (int) ((e->rows - row0) < PTQ1A_ROWS ? (e->rows - row0) : PTQ1A_ROWS);
                    mk[i] = _mm256_setr_epi32(nr > 0 ? -1 : 0, nr > 1 ? -1 : 0, nr > 2 ? -1 : 0, nr > 3 ? -1 : 0, nr > 4 ? -1 : 0, nr > 5 ? -1 : 0, nr > 6 ? -1 : 0, nr > 7 ? -1 : 0);
                    msk[i] = nr == PTQ1A_ROWS ? NULL : &mk[i];
                    d[i] = dst + m0 * ld + row0;
                }
                for (int i = 0; i < np && p + np + i < p1; ++i) {   // next pair's chunk: a cold DRAM/L3 read otherwise
                    const uint8_t * nxt = e->packed + (p + np + i) * e->panel_stride + choff;
                    for (int o = 0; o < nck * 64; o += 64) _mm_prefetch((const char *) nxt + o, _MM_HINT_T1);
                }
                for (int64_t t0 = m0; t0 < m1; t0 += PTQ1A_NT) {
                    const int nt = (int) (m1 - t0 < PTQ1A_NT ? m1 - t0 : PTQ1A_NT);
                    const int8_t * q[PTQ1A_NT]; const float * dy[PTQ1A_NT]; const int32_t * as[PTQ1A_NT]; float * dd[2];
                    for (int t = 0; t < nt; ++t) {
                        const uint8_t * base = acts + (size_t) (t0 + t) * cb;
                        q[t] = (const int8_t *) base + s0; dy[t] = (const float *) (base + kpad); as[t] = (const int32_t *) (dy[t] + kpad / 32);
                    }
                    for (int i = 0; i < np; ++i) dd[i] = d[i] + (t0 - m0) * ld;
                    switch (nt) {
                        case 1: ptq1a_tile(codes, cstride, nblk, sc, b0, np, q, dy, as, dd, ld, msk, sb == 0, 1); break;
                        case 2: ptq1a_tile(codes, cstride, nblk, sc, b0, np, q, dy, as, dd, ld, msk, sb == 0, 2); break;
                        case 3: ptq1a_tile(codes, cstride, nblk, sc, b0, np, q, dy, as, dd, ld, msk, sb == 0, 3); break;
                        default: ptq1a_tile(codes, cstride, nblk, sc, b0, np, q, dy, as, dd, ld, msk, sb == 0, PTQ1A_NT); break;
                    }
                }
            }
        }
    }
}

#ifndef PTQ1F_NO_GGML
static ptq1p_entry * ptq1a_list = NULL;
static pthread_mutex_t ptq1a_mu = PTHREAD_MUTEX_INITIALIZER;
static int ptq1a_smallm_cache = -1;
static inline int ptq1a_smallm(void) {   // tokens per call up to which the in-register kernel runs
    if (ptq1a_smallm_cache < 0) { const char * s = getenv("GGML_PTQ1_AVX2_SMALLM"); ptq1a_smallm_cache = s ? atoi(s) : 8; }
    return ptq1a_smallm_cache;
}
static size_t ptq1a_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1) { return ptq1p_col_bytes(src0->ne[0]) * (size_t) src1->ne[1] + 64; }

// ptq1p_mul_mat's scheduling on the AVX2 kernels: lock-free entry lookup, first-use repack split across threads,
// every thread quantizes a slice of every token's blocks for M <= 8 (one token per thread past that), one barrier,
// then dynamic panel chunks (ggml_threadpool_chunk_add) for every M.
static bool ptq1a_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int ith = params->ith, nth = params->nth;
    const int64_t k = src0->ne[0], rows = src0->ne[1], m = src1->ne[1];
    ptq1p_entry * e = NULL;
    for (ptq1p_entry * it = __atomic_load_n(&ptq1a_list, __ATOMIC_ACQUIRE); it; it = it->next) if (it->src == src0->data) { e = it; break; }
    if (!e) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            ptq1p_entry * ne = calloc(1, sizeof(*ne));
            ptq1a_init_entry(ne, src0->data, rows, k);
            // + 64: the kernels load the scale vector of the group after the last one (and never use it)
            ne->packed = aligned_alloc(64, (size_t) ((ne->panel_stride * ne->npanel + 63) / 64 * 64 + 64));
            pthread_mutex_lock(&ptq1a_mu); ne->next = ptq1a_list; __atomic_store_n(&ptq1a_list, ne, __ATOMIC_RELEASE); pthread_mutex_unlock(&ptq1a_mu);
        }
        ggml_barrier(params->threadpool);
        for (ptq1p_entry * it = __atomic_load_n(&ptq1a_list, __ATOMIC_ACQUIRE); it; it = it->next) if (it->src == src0->data) { e = it; break; }
        const int64_t per = (e->npanel + nth - 1) / nth;
        ptq1a_repack_panels(e, PTQ1F_MIN(e->npanel, ith * per), PTQ1F_MIN(e->npanel, (ith + 1) * per));
        // GGML_PTQ1_DROP (ptq1-drop.h): the last thread to finish its panels gives the GGUF pages back
        if (__atomic_add_fetch(&e->ndone, 1, __ATOMIC_ACQ_REL) == nth)
            ptq1d_drop(src0->data, ggml_nbytes(src0), strcmp(src0->name, "x1_draft_head") == 0);
    }
    const size_t cb = ptq1p_col_bytes(k);
    uint8_t * acts = (uint8_t *) (((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);
    ggml_threadpool_chunk_set(params->threadpool, nth);
    if (m <= PTQ1P_TB) {
        const int64_t nb = k / 32, per = (nb + nth - 1) / nth;
        for (int64_t t = 0; t < m; ++t)
            ptq1a_quantize_part((const float *) ((const char *) src1->data + t * src1->nb[1]), k, e->kpad, acts + t * cb, PTQ1F_MIN(nb, ith * per), PTQ1F_MIN(nb, (ith + 1) * per), 1);
    } else {
        for (int64_t t = ith; t < m; t += nth)
            ptq1a_quantize_part((const float *) ((const char *) src1->data + t * src1->nb[1]), k, e->kpad, acts + t * cb, 0, k / 32, 1);
    }
    ggml_barrier(params->threadpool);
    float * out = (float *) dst->data; const int64_t ld = dst->nb[1] / sizeof(float);
    if (m == 1) {
        int64_t ch = PTQ1F_MIN(16, (e->npanel + 8 * nth - 1) / (8 * nth)); ch = (ch + 1) / 2 * 2;   // whole panel pairs
        const int64_t nch = (e->npanel + ch - 1) / ch;
        for (int64_t c = ith; c < nch; c = ggml_threadpool_chunk_add(params->threadpool, 1))
            ptq1a_panels(e, acts, out, c * ch, PTQ1F_MIN(e->npanel, (c + 1) * ch));
    } else if (m <= ptq1a_smallm()) {
        const int64_t ch = PTQ1F_MIN(8, (e->npanel + 8 * nth - 1) / (8 * nth));
        const int64_t nch = (e->npanel + ch - 1) / ch;
        for (int64_t c = ith; c < nch; c = ggml_threadpool_chunk_add(params->threadpool, 1))
            ptq1a_panels_m(e, acts, cb, m, c * ch, PTQ1F_MIN(e->npanel, (c + 1) * ch), out, ld);
    } else {
        const int64_t ch = PTQ1F_MIN(16, ((e->npanel + 8 * nth - 1) / (8 * nth) + 1) / 2 * 2);   // whole panel pairs
        const int64_t nch = (e->npanel + ch - 1) / ch;
        for (int64_t c = ith; c < nch; c = ggml_threadpool_chunk_add(params->threadpool, 1))
            ptq1a_gemm(e, acts, cb, m, c * ch, PTQ1F_MIN(e->npanel, (c + 1) * ch), out, ld);
    }
    return true;
}
#endif
#else
#define PTQ1A_OK 0
#endif
