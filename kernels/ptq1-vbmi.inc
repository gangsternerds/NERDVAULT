// ptq1-vbmi: GGML_PTQ1_FAST=5 -- the 16-row panels of ptq1-panel, but the trits are stored as bit-aligned
// fields so one decode step is a vpmultishiftqb (shared by 3 steps) + one vpermb, instead of the 6-uop base-3
// digit extraction of ptq1f_decode (2 compares, 2 masked byte ops, x*3 as 2 adds).
//
// Layout per panel: nsb "V" superblocks of 2048 bytes (608 k each), then n36 "W" superblocks of 1024 bytes
// (288 k each), then the fp16 scales exactly as ptq1-panel stores them (16 per 128-k group). Both superblock
// kinds are runs of 64-byte chunks = 8 qwords. Every qword holds 12 five-bit fields of 3 trits (bits 5f..5f+4,
// value t0 + 3 t1 + 9 t2 < 27). In a V superblock the 4 spare bits (60..63) hold 2 more trits (value t0 + 3 t1
// < 9): 38 trits per 64 bits = 1.684 bits/weight; a W superblock leaves them empty (36 trits, 1.78 bpw) and only
// exists to finish k without padding: k = 608 nsb + 288 n36 whenever that needs fewer bytes than padding k up to
// a multiple of 608 (k = 5120: 7 V + 3 W, 0 padding; k = 17408: 29 V, 1.3 % padding).
//
// Decode of one 512-byte unit of a V superblock = 38 vpdpbusd steps of 64 trits (a W superblock is 4 x 9 steps):
//   per chunk pair (c0, c1):  A = multishift(c0; 0,5,..,35)                 -> fields 0-7 of c0's 8 qwords
//                             B = multishift(c1; same)
//                             C = multishift(c0; 40,45,50,55,60,60,60,60)   D = multishift(c1; 60,60,60,60,40,45,50,55)
//                             E = blend_dword(C, D)                         -> fields 8-11 of c0 and of c1
//                             3 vpermb per vector (one 64-entry table per trit index) -> 9 steps
//   the 4-bit fields of the 8 chunks sit in C/D too; 7 vpermt2b gather them into one vector -> 2 vpermb -> 2 steps
// = 16 multishift + 4 blend + 7 vpermt2b + 38 vpermb + 38 vpdpbusd = 103 vector uops per 2432 trits = 2.7 per 64,
// vs ptq1f_decode's 6 + 1 vpdpbusd = 7 per 64. The flush per 32-k block is 3 uops (cvt, mul, fma): the block's
// int32 accumulator starts at -sum(q) (the quantizer stores it negated), so there is no vpsubd.
// Port view (Ice Lake, 2 vector ports for zmm): vpmultishiftqb/vpermb/vpermt2b are port 5 only (61 of ~117 uops
// per unit), vpdpbusd and vcvtdq2ps port 0, the rest either: ~60 cycles per 512 bytes against ~19 per 64 today.
// A multishift byte carries the next field's low bit above the 5-bit field (bits 4-5 above the 4-bit one), so
// every table is replicated to make those index bits don't-care.
//
// Step s of a panel covers k = 4s..4s+3 for the 16 rows, lane l = row l, exactly like ptq1-panel's decode steps;
// the int32 sums per 32-k block and the flush order are those of ptq1p_panel, so results are bit-identical.
// The activation column is laid out as in ptq1-panel but padded to this kpad (ptq1v_col_bytes) with -sum(q).
#if defined(__AVX512VBMI__)
#define PTQ1V_OK 1
#define PTQ1V_KSB   608          // k per V superblock
#define PTQ1V_SBB   2048         // bytes per V superblock
#define PTQ1V_STEPS 152          // decode steps per V superblock (= 19 Q8_0 blocks)
#define PTQ1W_KSB   288          // k per W superblock
#define PTQ1W_SBB   1024
#define PTQ1W_STEPS 72           // = 9 Q8_0 blocks
#ifndef PTQ1V_NCH
#define PTQ1V_NCH 2              // interleaved int32 accumulator chains per token (summed before each flush: exact)
#endif
#ifndef PTQ1V_PF
#define PTQ1V_PF 512             // software prefetch distance in bytes ahead of the chunk pair being decoded (0 = none)
#endif

// (nsb, n36) with the fewest bytes: pad k up to a multiple of 608, or finish exactly with 288-k superblocks
static void ptq1v_split(int64_t k, int64_t * nsb, int64_t * n36) {
    int64_t a = (k + PTQ1V_KSB - 1) / PTQ1V_KSB, d = 0, best = a * PTQ1V_SBB;
    for (int64_t x = k / PTQ1V_KSB; x >= 0; --x) {
        const int64_t r = k - x * PTQ1V_KSB;
        if (r % PTQ1W_KSB) continue;
        const int64_t bytes = x * PTQ1V_SBB + (r / PTQ1W_KSB) * PTQ1W_SBB;
        if (bytes < best) { best = bytes; a = x; d = r / PTQ1W_KSB; }
    }
    *nsb = a; *n36 = d;
}
static inline int64_t ptq1v_kpad(int64_t k) { int64_t a, d; ptq1v_split(k, &a, &d); return a * PTQ1V_KSB + d * PTQ1W_KSB; }

static void ptq1v_init_entry(ptq1p_entry * e, const void * src, int64_t rows, int64_t k) {
    e->src = src; e->rows = rows; e->k = k;
    ptq1v_split(k, &e->nsb, &e->n36);
    e->kpad = e->nsb * PTQ1V_KSB + e->n36 * PTQ1W_KSB;
    e->npanel = (rows + PTQ1P_ROWS - 1) / PTQ1P_ROWS;
    e->nck = 0;
    e->ngp = (e->kpad + 127) / 128;
    e->panel_stride = (e->nsb * PTQ1V_SBB + e->n36 * PTQ1W_SBB + e->ngp * PTQ1P_ROWS * 2 + 63) / 64 * 64;
}
static inline int64_t ptq1v_data_bytes(const ptq1p_entry * e) { return e->nsb * PTQ1V_SBB + e->n36 * PTQ1W_SBB; }

// per token: int8 q[kpad] | f32 dy[kpad/32] | i32 -asum[kpad/32] | 64 slack bytes (the kernel preloads asum[b + 1])
static size_t ptq1v_col_bytes(int64_t k) {
    const int64_t kpad = ptq1v_kpad(k);
    return (size_t) (kpad + (kpad / 32) * 8 + 64);
}

// Where the trit of (step s, byte position i) of a superblock lives: byte offset of its qword within the
// superblock, bit offset of the field in that qword, field width (5 or 4) and the trit's index n in the field
// (field value = sum of trit_n * 3^n). Position i is row i/4, k offset i%4, as vpdpbusd sees it.
// s < 152 addresses a V superblock; for a W superblock pass s + 1000 (s < 72).
static inline void ptq1v_slot(int s, int i, int * qoff, int * bit, int * width, int * n) {
    const int w = s >= 1000;
    if (w) s -= 1000;
    const int u = w ? s / 72 : s / 38, sl = w ? s : s % 38;
    int chunk, qw, field;
    if (sl < 36 || w) {
        const int p = sl / 9, v = (sl % 9) / 3, j = i % 8;
        qw = i / 8; *n = sl % 3; *width = 5;
        if (v == 0)      { chunk = 2 * p;            field = j; }
        else if (v == 1) { chunk = 2 * p + 1;        field = j; }
        else             { chunk = 2 * p + (j >= 4); field = 8 + (j & 3); }
        *bit = 5 * field;
    } else {
        chunk = i / 8; qw = i % 8; *n = sl - 36; *width = 4; *bit = 60;
    }
    *qoff = u * 512 + chunk * 64 + qw * 8;
}

// codes: PTQ1P_ROWS rows of kpad trits {0,1,2} (row stride kpad) -> the panel's superblocks at dst
static void ptq1v_pack_codes(const uint8_t * codes, int64_t kpad, uint8_t * dst, int64_t nsb, int64_t n36) {
    static const uint64_t pow3[3] = {1, 3, 9};
    memset(dst, 0, (size_t) (nsb * PTQ1V_SBB + n36 * PTQ1W_SBB));
    for (int64_t sb = 0; sb < nsb + n36; ++sb) {
        const int w = sb >= nsb;
        uint8_t * sdst = w ? dst + nsb * PTQ1V_SBB + (sb - nsb) * PTQ1W_SBB : dst + sb * PTQ1V_SBB;
        const int64_t k0 = w ? nsb * PTQ1V_KSB + (sb - nsb) * PTQ1W_KSB : sb * PTQ1V_KSB;
        const int steps = w ? PTQ1W_STEPS : PTQ1V_STEPS;
        for (int s = 0; s < steps; ++s)
            for (int i = 0; i < 64; ++i) {
                int qoff, bit, width, n;
                ptq1v_slot(w ? s + 1000 : s, i, &qoff, &bit, &width, &n);
                const uint64_t c = codes[(i / 4) * kpad + k0 + 4 * s + (i % 4)];
                uint8_t * p = sdst + qoff;
                uint64_t x; memcpy(&x, p, 8); x += (c * pow3[n]) << bit; memcpy(p, &x, 8);
            }
    }
}

static void ptq1v_repack_panels(ptq1p_entry * e, int64_t p0, int64_t p1) {
    const int64_t nb = e->k / 128;
    uint8_t * codes = malloc((size_t) (PTQ1P_ROWS * e->kpad));
    for (int64_t p = p0; p < p1; ++p) {
        memset(codes, 1, (size_t) (PTQ1P_ROWS * e->kpad));             // trit 1 == weight 0
        uint8_t * dst = e->packed + p * e->panel_stride;
        ggml_half * sc = (ggml_half *) (dst + ptq1v_data_bytes(e));
        memset(sc, 0, (size_t) (e->ngp * PTQ1P_ROWS * 2));
        for (int r = 0; r < PTQ1P_ROWS; ++r) {
            const int64_t row = p * PTQ1P_ROWS + r;
            if (row >= e->rows) continue;
            const block_ptq1_0 * src = (const block_ptq1_0 *) e->src + row * nb;
            for (int64_t b = 0; b < nb; ++b) {
                ptq1f_unpack_block(src + b, codes + r * e->kpad + b * 128);
                sc[b * PTQ1P_ROWS + r] = src[b].d;
            }
        }
        ptq1v_pack_codes(codes, e->kpad, dst, e->nsb, e->n36);
    }
    free(codes);
}

// constant vectors: 0 ctrlA, 1 ctrlC, 2 ctrlD, 3-5 trit-n tables for 5-bit fields, 6-7 for 4-bit fields,
// 8 gather index for (C0, D0), 9 gather-append index from a C, 10 from a D
// Written out as constants (not filled in by a load-time constructor): a --portable build loads every CPU variant of
// this library to ask whether it can run, and code that runs at load time in an AVX-512 variant is an illegal
// instruction on a CPU without AVX-512. The formulas: i = byte, j = i % 8;
//   [0] 5j   [1] j < 4 ? 40 + 5j : 60   [2] j < 4 ? 60 : 40 + 5(j - 4)
//   [3 + n] trit n of (i & 31) if < 27 else 0   [6 + n] trit n of (i & 15) if < 9 else 0
//   [8] i < 8 ? 8j + 4 : 64 + 8j   [9] 64 + 8j + 4   [10] 64 + 8j      (tests/test_kernel_tables.py recomputes them)
static const uint8_t ptq1v_tab[11][64] __attribute__((aligned(64))) = {
    {   0,   5,  10,  15,  20,  25,  30,  35,   0,   5,  10,  15,  20,  25,  30,  35,
        0,   5,  10,  15,  20,  25,  30,  35,   0,   5,  10,  15,  20,  25,  30,  35,
        0,   5,  10,  15,  20,  25,  30,  35,   0,   5,  10,  15,  20,  25,  30,  35,
        0,   5,  10,  15,  20,  25,  30,  35,   0,   5,  10,  15,  20,  25,  30,  35 },
    {  40,  45,  50,  55,  60,  60,  60,  60,  40,  45,  50,  55,  60,  60,  60,  60,
       40,  45,  50,  55,  60,  60,  60,  60,  40,  45,  50,  55,  60,  60,  60,  60,
       40,  45,  50,  55,  60,  60,  60,  60,  40,  45,  50,  55,  60,  60,  60,  60,
       40,  45,  50,  55,  60,  60,  60,  60,  40,  45,  50,  55,  60,  60,  60,  60 },
    {  60,  60,  60,  60,  40,  45,  50,  55,  60,  60,  60,  60,  40,  45,  50,  55,
       60,  60,  60,  60,  40,  45,  50,  55,  60,  60,  60,  60,  40,  45,  50,  55,
       60,  60,  60,  60,  40,  45,  50,  55,  60,  60,  60,  60,  40,  45,  50,  55,
       60,  60,  60,  60,  40,  45,  50,  55,  60,  60,  60,  60,  40,  45,  50,  55 },
    {   0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   1,   2,   0,
        1,   2,   0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   0,   0,   0,   0,
        0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   1,   2,   0,
        1,   2,   0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   0,   0,   0,   0 },
    {   0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   1,   1,   1,   2,
        2,   2,   0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   0,   0,
        0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   1,   1,   1,   2,
        2,   2,   0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   0,   0 },
    {   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,
        1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,
        1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   0,   0,   0,   0,   0 },
    {   0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   0,   0,   0,   0,   0,   0,
        0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   0,   0,   0,   0,   0,   0,
        0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   0,   0,   0,   0,   0,   0,
        0,   1,   2,   0,   1,   2,   0,   1,   2,   0,   0,   0,   0,   0,   0,   0 },
    {   0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   1,   1,   1,   2,   2,   2,   0,   0,   0,   0,   0,   0,   0 },
    {   4,  12,  20,  28,  36,  44,  52,  60,  64,  72,  80,  88,  96, 104, 112, 120,
       64,  72,  80,  88,  96, 104, 112, 120,  64,  72,  80,  88,  96, 104, 112, 120,
       64,  72,  80,  88,  96, 104, 112, 120,  64,  72,  80,  88,  96, 104, 112, 120,
       64,  72,  80,  88,  96, 104, 112, 120,  64,  72,  80,  88,  96, 104, 112, 120 },
    {  68,  76,  84,  92, 100, 108, 116, 124,  68,  76,  84,  92, 100, 108, 116, 124,
       68,  76,  84,  92, 100, 108, 116, 124,  68,  76,  84,  92, 100, 108, 116, 124,
       68,  76,  84,  92, 100, 108, 116, 124,  68,  76,  84,  92, 100, 108, 116, 124,
       68,  76,  84,  92, 100, 108, 116, 124,  68,  76,  84,  92, 100, 108, 116, 124 },
    {  64,  72,  80,  88,  96, 104, 112, 120,  64,  72,  80,  88,  96, 104, 112, 120,
       64,  72,  80,  88,  96, 104, 112, 120,  64,  72,  80,  88,  96, 104, 112, 120,
       64,  72,  80,  88,  96, 104, 112, 120,  64,  72,  80,  88,  96, 104, 112, 120,
       64,  72,  80,  88,  96, 104, 112, 120,  64,  72,  80,  88,  96, 104, 112, 120 }
};

// NT tokens x one panel; same accumulation and flush as ptq1p_panel. Fully unrolled over the steps of a
// superblock so every flush point is a compile-time constant.
static inline __attribute__((always_inline)) void ptq1v_panel_fl(
        const ptq1p_entry * e, const uint8_t * panel, const uint8_t * acts, size_t cb, const int NT, const int NCH, const int PF, const int FL,
        float * dst, int64_t ld, int64_t row0, int64_t t0) {
    const int64_t kpad = e->kpad;
    const int64_t blast = kpad / 32 - 1;   // FL128: the last block flushes whatever its group position
    const ggml_half * sc = (const ggml_half *) (panel + ptq1v_data_bytes(e));
    const int8_t * q[PTQ1P_TB]; const float * dy[PTQ1P_TB]; const int32_t * as[PTQ1P_TB];
    for (int t = 0; t < NT; ++t) {
        const uint8_t * base = acts + (size_t) (t0 + t) * cb;
        q[t] = (const int8_t *) base; dy[t] = (const float *) (base + kpad); as[t] = (const int32_t *) (dy[t] + (FL == 1 ? kpad / 32 : (kpad + 127) / 128));
    }
    const __m512i ctrlA = _mm512_load_si512(ptq1v_tab[0]), ctrlC = _mm512_load_si512(ptq1v_tab[1]), ctrlD = _mm512_load_si512(ptq1v_tab[2]);
    const __m512i l30 = _mm512_load_si512(ptq1v_tab[3]), l31 = _mm512_load_si512(ptq1v_tab[4]), l32 = _mm512_load_si512(ptq1v_tab[5]);
    const __m512i l20 = _mm512_load_si512(ptq1v_tab[6]), l21 = _mm512_load_si512(ptq1v_tab[7]);
    const __m512i g0 = _mm512_load_si512(ptq1v_tab[8]), gC = _mm512_load_si512(ptq1v_tab[9]), gD = _mm512_load_si512(ptq1v_tab[10]);
    __m512 accf[PTQ1P_TB]; __m512i acci[PTQ1P_TB][NCH];
    for (int t = 0; t < NT; ++t) {
        accf[t] = _mm512_setzero_ps();
        acci[t][0] = _mm512_set1_epi32(as[t][0]);                     // block 0 starts at -sum(q)
        for (int c = 1; c < NCH; ++c) acci[t][c] = _mm512_setzero_si512();
    }
    __m512 dxv = _mm512_setzero_ps();
    int64_t b0 = 0, s0 = 0;        // first block / step of the current superblock
    const uint8_t * P = panel;
#define PTQ1V_STEP(s, codes) do {                                                                             \
        const __m512i cv_ = (codes);                                                                          \
        for (int t = 0; t < NT; ++t)                                                                          \
            acci[t][(s) % NCH] = _mm512_dpbusd_epi32(acci[t][(s) % NCH], cv_,                     \
                                                           _mm512_set1_epi32(*(const int32_t *) (q[t] + 4 * (s0 + (s))))); \
        if ((s) % 8 == 7) {                                                                                   \
            const int64_t b = b0 + (s) / 8;                                                                   \
            if ((b & 3) == 0) dxv = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (sc + (b >> 2) * PTQ1P_ROWS))); \
            if (FL == 1 || (b & 3) == 3 || b == blast) {                                                      \
                const int64_t gi_ = FL == 1 ? b : (b >> 2);                                                   \
                for (int t = 0; t < NT; ++t) {                                                                \
                    __m512i sum_ = acci[t][0];                                                                \
                    for (int c = 1; c < NCH; ++c) sum_ = _mm512_add_epi32(sum_, acci[t][c]);            \
                    accf[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum_), _mm512_mul_ps(dxv, _mm512_set1_ps(dy[t][gi_])), accf[t]); \
                    acci[t][0] = _mm512_set1_epi32(as[t][gi_ + 1]);                                           \
                    for (int c = 1; c < NCH; ++c) acci[t][c] = _mm512_setzero_si512();                  \
                }                                                                                             \
            }                                                                                                 \
        } } while (0)
    // one chunk pair at byte offset off, 9 steps from step sl; tail: gather its 4-bit fields into T (V only)
#define PTQ1V_PAIR(off, sl, p, TAIL) do {                                                                     \
        if (PF) { _mm_prefetch((const char *) (P + (off) + PF), _MM_HINT_T0); _mm_prefetch((const char *) (P + (off) + PF + 64), _MM_HINT_T0); } \
        const __m512i c0 = _mm512_loadu_si512((const void *) (P + (off)));                                    \
        const __m512i c1 = _mm512_loadu_si512((const void *) (P + (off) + 64));                               \
        const __m512i A = _mm512_multishift_epi64_epi8(ctrlA, c0), B = _mm512_multishift_epi64_epi8(ctrlA, c1); \
        const __m512i C = _mm512_multishift_epi64_epi8(ctrlC, c0), D = _mm512_multishift_epi64_epi8(ctrlD, c1); \
        const __m512i E = _mm512_mask_blend_epi32((__mmask16) 0xAAAA, C, D);                                  \
        if (TAIL) {                                                                                           \
            if ((p) == 0) T = _mm512_permutex2var_epi8(C, g0, D);                                             \
            else {                                                                                            \
                T = _mm512_mask_permutex2var_epi8(T, (__mmask64) 0xFFu << (16 * (p)), gC, C);                 \
                T = _mm512_mask_permutex2var_epi8(T, (__mmask64) 0xFFu << (16 * (p) + 8), gD, D);             \
            }                                                                                                 \
        }                                                                                                     \
        PTQ1V_STEP((sl) + 0, _mm512_permutexvar_epi8(A, l30));                                                \
        PTQ1V_STEP((sl) + 1, _mm512_permutexvar_epi8(A, l31));                                                \
        PTQ1V_STEP((sl) + 2, _mm512_permutexvar_epi8(A, l32));                                                \
        PTQ1V_STEP((sl) + 3, _mm512_permutexvar_epi8(B, l30));                                                \
        PTQ1V_STEP((sl) + 4, _mm512_permutexvar_epi8(B, l31));                                                \
        PTQ1V_STEP((sl) + 5, _mm512_permutexvar_epi8(B, l32));                                                \
        PTQ1V_STEP((sl) + 6, _mm512_permutexvar_epi8(E, l30));                                                \
        PTQ1V_STEP((sl) + 7, _mm512_permutexvar_epi8(E, l31));                                                \
        PTQ1V_STEP((sl) + 8, _mm512_permutexvar_epi8(E, l32));                                                \
        } while (0)
    for (int64_t sb = 0; sb < e->nsb; ++sb, P += PTQ1V_SBB, b0 += PTQ1V_STEPS / 8, s0 += PTQ1V_STEPS) {
#pragma GCC unroll 4
        for (int u = 0; u < 4; ++u) {
            __m512i T = _mm512_setzero_si512();
#pragma GCC unroll 4
            for (int p = 0; p < 4; ++p) PTQ1V_PAIR(u * 512 + p * 128, u * 38 + p * 9, p, 1);
            PTQ1V_STEP(u * 38 + 36, _mm512_permutexvar_epi8(T, l20));
            PTQ1V_STEP(u * 38 + 37, _mm512_permutexvar_epi8(T, l21));
        }
    }
    for (int64_t sb = 0; sb < e->n36; ++sb, P += PTQ1W_SBB, b0 += PTQ1W_STEPS / 8, s0 += PTQ1W_STEPS) {
        __m512i T = _mm512_setzero_si512(); (void) T;
#pragma GCC unroll 8
        for (int p = 0; p < 8; ++p) PTQ1V_PAIR(p * 128, p * 9, p, 0);
    }
#undef PTQ1V_PAIR
#undef PTQ1V_STEP
    const int64_t nr = (e->rows - row0) < PTQ1P_ROWS ? (e->rows - row0) : PTQ1P_ROWS;
    const __mmask16 m = (__mmask16) ((1u << nr) - 1);
    for (int t = 0; t < NT; ++t) _mm512_mask_storeu_ps(dst + (t0 + t) * ld + row0, m, accf[t]);
}

static inline __attribute__((always_inline)) void ptq1v_panel(
        const ptq1p_entry * e, const uint8_t * panel, const uint8_t * acts, size_t cb, const int NT, const int NCH, const int PF,
        float * dst, int64_t ld, int64_t row0, int64_t t0) {
    ptq1v_panel_fl(e, panel, acts, cb, NT, NCH, PF, 1, dst, ld, row0, t0);
}

// M = 1 (decode) path
static void ptq1v_panel1(const ptq1p_entry * e, const uint8_t * panel, const uint8_t * acts, float * dst, int64_t row0) {
    ptq1v_panel(e, panel, acts, 0, 1, PTQ1V_NCH, PTQ1V_PF, dst, 0, row0, 0);
}

// GGML_PTQ1_FAST=7 (K5, 2026-10-27): the M = 1 decode walks TWO panels at once. One panel is one
// weight stream per core, and at ~15 uops per 64-byte line the OoO window keeps too few misses in flight: mode 6
// streams at 54-65 GB/s against a 70-108 GB/s read floor, worst on k = 5120 (19 KB panels). Two interleaved panels
// (2 streams, the decode uops unchanged) are 1.09-1.26x on 3 boxes x 4 (k, call size) cells; 4 panels gain no more and
// lose on small matrices (256 KB tail units). Chunk size, next-chunk prefetch and the chain count were measured flat.
// NP consecutive panels (panel, panel + stride, ...) x one token, interleaved step by step. Same per-panel accumulation
// and flush order as ptq1v_panel (NT = 1), so every panel's 16 outputs are bit-identical to ptq1v_panel1 (k5.c checks).
#define PTQ1V_MAXNP 4
static inline __attribute__((always_inline)) void ptq1v_panel_np(
        const ptq1p_entry * e, const uint8_t * panel, const uint8_t * acts, const int NP, const int NCH, const int PF,
        float * dst, int64_t row0) {
    const int64_t kpad = e->kpad;
    const int8_t * q = (const int8_t *) acts; const float * dy = (const float *) (acts + kpad); const int32_t * as = (const int32_t *) (dy + kpad / 32);
    const ggml_half * sc[PTQ1V_MAXNP]; const uint8_t * P[PTQ1V_MAXNP];
    for (int p = 0; p < NP; ++p) { P[p] = panel + p * e->panel_stride; sc[p] = (const ggml_half *) (P[p] + ptq1v_data_bytes(e)); }
    const __m512i ctrlA = _mm512_load_si512(ptq1v_tab[0]), ctrlC = _mm512_load_si512(ptq1v_tab[1]), ctrlD = _mm512_load_si512(ptq1v_tab[2]);
    const __m512i l30 = _mm512_load_si512(ptq1v_tab[3]), l31 = _mm512_load_si512(ptq1v_tab[4]), l32 = _mm512_load_si512(ptq1v_tab[5]);
    const __m512i l20 = _mm512_load_si512(ptq1v_tab[6]), l21 = _mm512_load_si512(ptq1v_tab[7]);
    const __m512i g0 = _mm512_load_si512(ptq1v_tab[8]), gC = _mm512_load_si512(ptq1v_tab[9]), gD = _mm512_load_si512(ptq1v_tab[10]);
    __m512 accf[PTQ1V_MAXNP], dxv[PTQ1V_MAXNP]; __m512i acci[PTQ1V_MAXNP][PTQ1V_NCH > 2 ? PTQ1V_NCH : 2];
    for (int p = 0; p < NP; ++p) {
        accf[p] = _mm512_setzero_ps(); dxv[p] = _mm512_setzero_ps();
        acci[p][0] = _mm512_set1_epi32(as[0]);
        for (int c = 1; c < NCH; ++c) acci[p][c] = _mm512_setzero_si512();
    }
    int64_t b0 = 0, s0 = 0;
#define PTQ1VN_STEP(s, CODES) do {                                                                            \
        const __m512i bq_ = _mm512_set1_epi32(*(const int32_t *) (q + 4 * (s0 + (s))));                       \
        for (int p = 0; p < NP; ++p) acci[p][(s) % NCH] = _mm512_dpbusd_epi32(acci[p][(s) % NCH], CODES(p), bq_); \
        if ((s) % 8 == 7) {                                                                                   \
            const int64_t b = b0 + (s) / 8;                                                                   \
            const __m512 dyv_ = _mm512_set1_ps(dy[b]); const __m512i nas_ = _mm512_set1_epi32(as[b + 1]);     \
            for (int p = 0; p < NP; ++p) {                                                                    \
                if ((b & 3) == 0) dxv[p] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (sc[p] + (b >> 2) * PTQ1P_ROWS))); \
                __m512i sum_ = acci[p][0];                                                                    \
                for (int c = 1; c < NCH; ++c) sum_ = _mm512_add_epi32(sum_, acci[p][c]);                      \
                accf[p] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum_), _mm512_mul_ps(dxv[p], dyv_), accf[p]);   \
                acci[p][0] = nas_;                                                                            \
                for (int c = 1; c < NCH; ++c) acci[p][c] = _mm512_setzero_si512();                            \
            }                                                                                                 \
        } } while (0)
#define PTQ1VN_A(p) _mm512_permutexvar_epi8(A[p], L)
#define PTQ1VN_B(p) _mm512_permutexvar_epi8(B[p], L)
#define PTQ1VN_E(p) _mm512_permutexvar_epi8(E[p], L)
#define PTQ1VN_T(p) _mm512_permutexvar_epi8(T[p], L)
#define PTQ1VN_PAIR(off, sl, pp, TAIL) do {                                                                   \
        __m512i A[PTQ1V_MAXNP], B[PTQ1V_MAXNP], E[PTQ1V_MAXNP];                                               \
        for (int p = 0; p < NP; ++p) {                                                                        \
            if (PF) { _mm_prefetch((const char *) (P[p] + (off) + PF), _MM_HINT_T0); _mm_prefetch((const char *) (P[p] + (off) + PF + 64), _MM_HINT_T0); } \
            const __m512i c0 = _mm512_loadu_si512((const void *) (P[p] + (off)));                             \
            const __m512i c1 = _mm512_loadu_si512((const void *) (P[p] + (off) + 64));                        \
            A[p] = _mm512_multishift_epi64_epi8(ctrlA, c0); B[p] = _mm512_multishift_epi64_epi8(ctrlA, c1);   \
            const __m512i C = _mm512_multishift_epi64_epi8(ctrlC, c0), D = _mm512_multishift_epi64_epi8(ctrlD, c1); \
            E[p] = _mm512_mask_blend_epi32((__mmask16) 0xAAAA, C, D);                                         \
            if (TAIL) {                                                                                       \
                if ((pp) == 0) T[p] = _mm512_permutex2var_epi8(C, g0, D);                                     \
                else {                                                                                        \
                    T[p] = _mm512_mask_permutex2var_epi8(T[p], (__mmask64) 0xFFu << (16 * (pp)), gC, C);      \
                    T[p] = _mm512_mask_permutex2var_epi8(T[p], (__mmask64) 0xFFu << (16 * (pp) + 8), gD, D);  \
                }                                                                                             \
            }                                                                                                 \
        }                                                                                                     \
        { const __m512i L = l30; PTQ1VN_STEP((sl) + 0, PTQ1VN_A); }                                           \
        { const __m512i L = l31; PTQ1VN_STEP((sl) + 1, PTQ1VN_A); }                                           \
        { const __m512i L = l32; PTQ1VN_STEP((sl) + 2, PTQ1VN_A); }                                           \
        { const __m512i L = l30; PTQ1VN_STEP((sl) + 3, PTQ1VN_B); }                                           \
        { const __m512i L = l31; PTQ1VN_STEP((sl) + 4, PTQ1VN_B); }                                           \
        { const __m512i L = l32; PTQ1VN_STEP((sl) + 5, PTQ1VN_B); }                                           \
        { const __m512i L = l30; PTQ1VN_STEP((sl) + 6, PTQ1VN_E); }                                           \
        { const __m512i L = l31; PTQ1VN_STEP((sl) + 7, PTQ1VN_E); }                                           \
        { const __m512i L = l32; PTQ1VN_STEP((sl) + 8, PTQ1VN_E); }                                           \
        } while (0)
    for (int64_t sb = 0; sb < e->nsb; ++sb, b0 += PTQ1V_STEPS / 8, s0 += PTQ1V_STEPS) {
#pragma GCC unroll 4
        for (int u = 0; u < 4; ++u) {
            __m512i T[PTQ1V_MAXNP];
            for (int p = 0; p < NP; ++p) T[p] = _mm512_setzero_si512();
#pragma GCC unroll 4
            for (int pp = 0; pp < 4; ++pp) PTQ1VN_PAIR(u * 512 + pp * 128, u * 38 + pp * 9, pp, 1);
            { const __m512i L = l20; PTQ1VN_STEP(u * 38 + 36, PTQ1VN_T); }
            { const __m512i L = l21; PTQ1VN_STEP(u * 38 + 37, PTQ1VN_T); }
        }
        for (int p = 0; p < NP; ++p) P[p] += PTQ1V_SBB;
    }
    for (int64_t sb = 0; sb < e->n36; ++sb, b0 += PTQ1W_STEPS / 8, s0 += PTQ1W_STEPS) {
        __m512i T[PTQ1V_MAXNP]; (void) T;
#pragma GCC unroll 8
        for (int pp = 0; pp < 8; ++pp) PTQ1VN_PAIR(pp * 128, pp * 9, pp, 0);
        for (int p = 0; p < NP; ++p) P[p] += PTQ1W_SBB;
    }
#undef PTQ1VN_PAIR
#undef PTQ1VN_STEP
#undef PTQ1VN_A
#undef PTQ1VN_B
#undef PTQ1VN_E
#undef PTQ1VN_T
    for (int p = 0; p < NP; ++p) {
        const int64_t r0 = row0 + p * PTQ1P_ROWS;
        const int64_t nr = (e->rows - r0) < PTQ1P_ROWS ? (e->rows - r0) : PTQ1P_ROWS;
        if (nr <= 0) continue;
        _mm512_mask_storeu_ps(dst + r0, (__mmask16) ((1u << nr) - 1), accf[p]);
    }
}
#ifndef PTQ1V7_NP
#define PTQ1V7_NP 2
#endif
#ifndef PTQ1V7_PF
#define PTQ1V7_PF 2048
#endif
// mode 7, M = 1: panels [p0, p1) in groups of PTQ1V7_NP, the odd one with ptq1v_panel1
static void ptq1v_panels7(const ptq1p_entry * e, const uint8_t * acts, float * dst, int64_t p0, int64_t p1) {
    int64_t p = p0;
    for (; p + PTQ1V7_NP <= p1; p += PTQ1V7_NP)
        ptq1v_panel_np(e, e->packed + p * e->panel_stride, acts, PTQ1V7_NP, PTQ1V_NCH, PTQ1V7_PF, dst, p * PTQ1P_ROWS);
    for (; p < p1; ++p) ptq1v_panel1(e, e->packed + p * e->panel_stride, acts, dst, p * PTQ1P_ROWS);
}

// GGML_PTQ1_FAST=6: the M = 2..8 (speculative verify) path. Same decode-once-apply-to-NT-tokens loop as ptq1v_gemm,
// with two changes measured in our microbenchmark (2026-10-26, bit-identical in every run):
//   * past 2 tokens each token keeps ONE int32 chain -- the tokens are the independent chains, and NT x 2 chains plus
//     NT float accumulators and the 11 table vectors spill out of the 32 zmm registers (1.1-1.2x at M = 3, 4, 8);
//   * the software prefetch runs 2 KB ahead instead of 512 B: with 4 tokens per decoded chunk the hardware prefetcher
//     no longer keeps the stream ahead of the dpbusd work, and the distance alone takes C(4)/C(1) from ~1.4 to ~1.1.
// Every int32 block sum is exact whatever the chain count, so the output is bit-identical to ptq1v_panel1 per token
// and to ptq1v_gemm (k3.c checks both). M = 1 is untouched (ptq1v_panel1, PTQ1V_PF).
#ifndef PTQ1V_TBM
#define PTQ1V_TBM 8
#endif
#ifndef PTQ1V_PFM
#define PTQ1V_PFM 2048
#endif
static void ptq1v_gemm_m(const ptq1p_entry * e, const uint8_t * acts, size_t cb, int64_t m,
                         int64_t p0, int64_t p1, float * dst, int64_t ld) {
    for (int64_t p = p0; p < p1; ++p) {
        const uint8_t * panel = e->packed + p * e->panel_stride;
        for (int64_t t0 = 0; t0 < m; t0 += PTQ1V_TBM) {
            const int nt = (int) (m - t0 < PTQ1V_TBM ? m - t0 : PTQ1V_TBM);
            switch (nt) {
                case 1: ptq1v_panel(e, panel, acts, cb, 1, PTQ1V_NCH, PTQ1V_PF, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 2: ptq1v_panel(e, panel, acts, cb, 2, 2, PTQ1V_PFM, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 3: ptq1v_panel(e, panel, acts, cb, 3, 1, PTQ1V_PFM, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 4: ptq1v_panel(e, panel, acts, cb, 4, 1, PTQ1V_PFM, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 5: ptq1v_panel(e, panel, acts, cb, 5, 1, PTQ1V_PFM, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 6: ptq1v_panel(e, panel, acts, cb, 6, 1, PTQ1V_PFM, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 7: ptq1v_panel(e, panel, acts, cb, 7, 1, PTQ1V_PFM, dst, ld, p * PTQ1P_ROWS, t0); break;
                default: ptq1v_panel(e, panel, acts, cb, 8, 1, PTQ1V_PFM, dst, ld, p * PTQ1P_ROWS, t0); break;
            }
        }
    }
}

// FL128 (GGML_PTQ1_FL128=1): the same M = 2..8 tiles with per-128 activation scales and one flush per 128 k
static void ptq1v_gemm_m_fl4(const ptq1p_entry * e, const uint8_t * acts, size_t cb, int64_t m,
                             int64_t p0, int64_t p1, float * dst, int64_t ld) {
    for (int64_t p = p0; p < p1; ++p) {
        const uint8_t * panel = e->packed + p * e->panel_stride;
        for (int64_t t0 = 0; t0 < m; t0 += PTQ1V_TBM) {
            const int nt = (int) (m - t0 < PTQ1V_TBM ? m - t0 : PTQ1V_TBM);
            switch (nt) {
                case 1: ptq1v_panel_fl(e, panel, acts, cb, 1, PTQ1V_NCH, PTQ1V_PF, 4, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 2: ptq1v_panel_fl(e, panel, acts, cb, 2, 2, PTQ1V_PFM, 4, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 3: ptq1v_panel_fl(e, panel, acts, cb, 3, 1, PTQ1V_PFM, 4, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 4: ptq1v_panel_fl(e, panel, acts, cb, 4, 1, PTQ1V_PFM, 4, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 5: ptq1v_panel_fl(e, panel, acts, cb, 5, 1, PTQ1V_PFM, 4, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 6: ptq1v_panel_fl(e, panel, acts, cb, 6, 1, PTQ1V_PFM, 4, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 7: ptq1v_panel_fl(e, panel, acts, cb, 7, 1, PTQ1V_PFM, 4, dst, ld, p * PTQ1P_ROWS, t0); break;
                default: ptq1v_panel_fl(e, panel, acts, cb, 8, 1, PTQ1V_PFM, 4, dst, ld, p * PTQ1P_ROWS, t0); break;
            }
        }
    }
}

// K4 (GGML_PTQ1_K4=1 on top of mode 6): the prefill GEMM for M > 8 (2026-10-27).
// ptq1v_gemm walks panel-outer, 8-token-tile-inner, so every panel streams the whole M x k activation matrix
// (8.9 MB at M = 512, k = 17408) out of L3: one activation byte per 16 MACs = ~11 GB/s per thread at the vpdpbusd
// peak, an L3 bound; its 2 chains x 8 tokens + 8 float accumulators + 11 tables also spill the 32 zmm registers.
// Here the k axis is cut into chunks of PTQ1K4_KC superblocks (608 k = 2 KB of panel per superblock):
//   for each token block of MB tokens, for each k-chunk, for each panel of this thread's group:
//     decode the panel chunk ONCE into an L1 buffer of codes (152 zmm per superblock, 9.7 KB), then
//     multiply token tiles of NT against it: per step 1 codes load + NT broadcasts + NT vpdpbusd, no table registers;
//     the float accumulator of (token, 16 rows) is carried through dst between chunks.
// The activation chunk all tokens need (M x 608 KC bytes, 311 KB at M = 512, KC = 1) now sits in L2, the codes in L1.
// Numerics: each 32-k block's int32 sum is exact and the per-block fma into the float accumulator happens in block
// order starting from 0, exactly as in ptq1v_panel (NCH = 1), so the result is bit-identical to modes 5/6 and to
// M x ptq1v_panel1 (k4.c checks). The scale group (4 blocks) is tracked by global block index, as there.
#ifndef PTQ1K4_NP
#define PTQ1K4_NP 2             // panels per register tile (each activation broadcast feeds NP vpdpbusd)
#endif
#ifndef PTQ1K4_NT
#define PTQ1K4_NT 6             // tokens per register tile: NP x NT int32 + NP x NT float accumulators + NP dxv + temps < 32 zmm
#endif
#ifndef PTQ1K4_KC
#define PTQ1K4_KC 16            // superblocks per k-chunk (9728 k, i.e. 1-2 chunks on the 27B): fewer chunk switches beat L2 residency of the activation chunk (k4.c: 16 ~ 32 > 8 > 4 > 2 > 1)
#endif
#ifndef PTQ1K4_MB
#define PTQ1K4_MB 512           // tokens per token block (bounds the L2 activation chunk: MB x 608 KC bytes)
#endif
#ifndef PTQ1K4_PF
#define PTQ1K4_PF 1             // prefetch the next panel's chunk while multiplying this one
#endif
#ifndef PTQ1K4_G
#define PTQ1K4_G 8              // panels per dynamic work item in the fork (the activation chunk is re-read from L2 once per panel pair)
#endif
#define PTQ1K4_MAXNP 3
#define PTQ1K4_MAXNT 12

// The Q8_0 quantizer of ptq1p_quantize_col_part_kp in AVX-512, same q, same -sum(q), same fp16-rounded d per block.
// roundf (half away from zero) is reproduced exactly: r = trunc(v), then r += sign(v) where |v - r| >= 0.5
// (v - trunc(v) is exact in float for |v| < 2^23). amax is a max of |x| as there, d = amax / 127, id = 1 / d.
// BS = 32 is that quantizer; BS = 128 (GGML_PTQ1_K4=2) scales per 128 k, one entry per 128-k group (NOT Q8_0 numerics):
// the layout is int8 q[kpad] | f32 dy[ceil(kpad/128)] | i32 -sum[ceil(kpad/128)] and still fits ptq1v_col_bytes.
static inline __attribute__((always_inline)) void ptq1k4_quantize_col_bs(const float * x, int64_t k, int64_t kpad, uint8_t * out, const int BS) {
    int8_t * q = (int8_t *) out;
    const int64_t ng = (kpad + BS - 1) / BS;
    float * dy = (float *) (out + kpad);
    int32_t * asum = (int32_t *) (dy + ng);
    memset(q + k, 0, (size_t) (kpad - k));
    for (int64_t b = k / BS; b < ng; ++b) { dy[b] = 0.0f; asum[b] = 0; }
    const __m512 half = _mm512_set1_ps(0.5f), one = _mm512_set1_ps(1.0f);
    for (int64_t b = 0; b < k / BS; ++b) {
        __m512 xv[8], am = _mm512_setzero_ps();
        for (int h = 0; h < BS / 16; ++h) { xv[h] = _mm512_loadu_ps(x + b * BS + 16 * h); am = _mm512_max_ps(am, _mm512_abs_ps(xv[h])); }
        const float amax = _mm512_reduce_max_ps(am);
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        const __m512 idv = _mm512_set1_ps(id);
        int s = 0;
        for (int h = 0; h < BS / 16; ++h) {
            const __m512 v = _mm512_mul_ps(xv[h], idv);
            const __m512 r = _mm512_roundscale_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            const __mmask16 away = _mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_sub_ps(v, r)), half, _CMP_GE_OQ);
            const __m512 sg = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_LT_OQ), one, _mm512_set1_ps(-1.0f));
            const __m512i qi = _mm512_cvttps_epi32(_mm512_mask_add_ps(r, away, r, sg));
            s += _mm512_reduce_add_epi32(qi);
            _mm_storeu_si128((__m128i *) (q + b * BS + 16 * h), _mm512_cvtepi32_epi8(qi));
        }
        dy[b] = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(d));
        asum[b] = -s;
    }
}
static void ptq1k4_quantize_col(const float * x, int64_t k, int64_t kpad, uint8_t * out) { ptq1k4_quantize_col_bs(x, k, kpad, out, 32); }
// FL128: groups [g0, g1) of 128 k of one token's column, layout int8 q[kpad] | f32 dy[ng] | i32 -sum[ng] (ptq1k4_quantize_col128's);
// the g0 == 0 caller zeroes the padding. Same d, rounding and fp16 scale as ptq1k4_quantize_col_bs(BS = 128).
static void ptq1k4_quantize_part128(const float * x, int64_t k, int64_t kpad, uint8_t * out, int64_t g0, int64_t g1) {
    int8_t * q = (int8_t *) out;
    const int64_t ng = (kpad + 127) / 128, nk = k / 128;
    float * dy = (float *) (out + kpad);
    int32_t * asum = (int32_t *) (dy + ng);
    if (g0 == 0) { memset(q + k, 0, (size_t) (kpad - k)); for (int64_t b = nk; b < ng; ++b) { dy[b] = 0.0f; asum[b] = 0; } }
    const __m512 half = _mm512_set1_ps(0.5f), one = _mm512_set1_ps(1.0f);
    for (int64_t b = g0; b < g1 && b < nk; ++b) {
        __m512 xv[8], am = _mm512_setzero_ps();
        for (int h = 0; h < 8; ++h) { xv[h] = _mm512_loadu_ps(x + b * 128 + 16 * h); am = _mm512_max_ps(am, _mm512_abs_ps(xv[h])); }
        const float amax = _mm512_reduce_max_ps(am);
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        const __m512 idv = _mm512_set1_ps(id);
        int s = 0;
        for (int h = 0; h < 8; ++h) {
            const __m512 v = _mm512_mul_ps(xv[h], idv);
            const __m512 r = _mm512_roundscale_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            const __mmask16 away = _mm512_cmp_ps_mask(_mm512_abs_ps(_mm512_sub_ps(v, r)), half, _CMP_GE_OQ);
            const __m512 sg = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_LT_OQ), one, _mm512_set1_ps(-1.0f));
            const __m512i qi = _mm512_cvttps_epi32(_mm512_mask_add_ps(r, away, r, sg));
            s += _mm512_reduce_add_epi32(qi);
            _mm_storeu_si128((__m128i *) (q + b * 128 + 16 * h), _mm512_cvtepi32_epi8(qi));
        }
        dy[b] = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(d));
        asum[b] = -s;
    }
}
static void ptq1k4_quantize_col128(const float * x, int64_t k, int64_t kpad, uint8_t * out) { ptq1k4_quantize_col_bs(x, k, kpad, out, 128); }

// one superblock (V: 152 steps, W: 72) -> out[]; returns the step count
static inline __attribute__((always_inline)) int ptq1v_decode_sb(const uint8_t * P, int w, __m512i * out) {
    const __m512i ctrlA = _mm512_load_si512(ptq1v_tab[0]), ctrlC = _mm512_load_si512(ptq1v_tab[1]), ctrlD = _mm512_load_si512(ptq1v_tab[2]);
    const __m512i l30 = _mm512_load_si512(ptq1v_tab[3]), l31 = _mm512_load_si512(ptq1v_tab[4]), l32 = _mm512_load_si512(ptq1v_tab[5]);
    const __m512i l20 = _mm512_load_si512(ptq1v_tab[6]), l21 = _mm512_load_si512(ptq1v_tab[7]);
    const __m512i g0 = _mm512_load_si512(ptq1v_tab[8]), gC = _mm512_load_si512(ptq1v_tab[9]), gD = _mm512_load_si512(ptq1v_tab[10]);
    int o = 0;
#define PTQ1K4_PAIR(off, p, TAIL) do {                                                                        \
        const __m512i c0 = _mm512_loadu_si512((const void *) (P + (off)));                                    \
        const __m512i c1 = _mm512_loadu_si512((const void *) (P + (off) + 64));                               \
        const __m512i A = _mm512_multishift_epi64_epi8(ctrlA, c0), B = _mm512_multishift_epi64_epi8(ctrlA, c1); \
        const __m512i C = _mm512_multishift_epi64_epi8(ctrlC, c0), D = _mm512_multishift_epi64_epi8(ctrlD, c1); \
        const __m512i E = _mm512_mask_blend_epi32((__mmask16) 0xAAAA, C, D);                                  \
        if (TAIL) {                                                                                           \
            if ((p) == 0) T = _mm512_permutex2var_epi8(C, g0, D);                                             \
            else {                                                                                            \
                T = _mm512_mask_permutex2var_epi8(T, (__mmask64) 0xFFu << (16 * (p)), gC, C);                 \
                T = _mm512_mask_permutex2var_epi8(T, (__mmask64) 0xFFu << (16 * (p) + 8), gD, D);             \
            }                                                                                                 \
        }                                                                                                     \
        out[o + 0] = _mm512_permutexvar_epi8(A, l30); out[o + 1] = _mm512_permutexvar_epi8(A, l31);           \
        out[o + 2] = _mm512_permutexvar_epi8(A, l32); out[o + 3] = _mm512_permutexvar_epi8(B, l30);           \
        out[o + 4] = _mm512_permutexvar_epi8(B, l31); out[o + 5] = _mm512_permutexvar_epi8(B, l32);           \
        out[o + 6] = _mm512_permutexvar_epi8(E, l30); out[o + 7] = _mm512_permutexvar_epi8(E, l31);           \
        out[o + 8] = _mm512_permutexvar_epi8(E, l32); o += 9;                                                 \
        } while (0)
    if (!w) {
        for (int u = 0; u < 4; ++u) {
            __m512i T = _mm512_setzero_si512();
            for (int p = 0; p < 4; ++p) PTQ1K4_PAIR(u * 512 + p * 128, p, 1);
            out[o++] = _mm512_permutexvar_epi8(T, l20);
            out[o++] = _mm512_permutexvar_epi8(T, l21);
        }
    } else {
        __m512i T = _mm512_setzero_si512(); (void) T;
        for (int p = 0; p < 8; ++p) PTQ1K4_PAIR(p * 128, p, 0);
    }
#undef PTQ1K4_PAIR
    return o;
}

// NP panels x NT tokens x the decoded chunk (nblk blocks from global block b0). codes[p] is panel p's decoded chunk
// (stride cstride vectors), sc[p] its scales, msk[p] its row mask; q/dy/as point at the tokens' columns, q pre-offset
// to the chunk's first step; dst[p] = output of (token 0, panel p's rows). first = this is block 0 (accumulators
// start at 0, else from dst). Each activation broadcast feeds NP vpdpbusd, which is what keeps the two load ports
// under the vector ports: with NP = 1 every vpdpbusd carries its own load (plus the flush's) and the loop runs at
// ~1.9 loads/cycle, load-bound at ~50 % of the dpbusd peak (k4.c, 2026-10-27).
static inline __attribute__((always_inline)) void ptq1k4_tile(
        const __m512i * codes, int cstride, int nblk, const ggml_half * const * sc, int64_t b0,
        const int8_t * const * q, const float * const * dy, const int32_t * const * as,
        float * const * dst, int64_t ld, const __mmask16 * msk, int first, const int NP, const int NT, const int FL) {
    __m512i acci[PTQ1K4_MAXNP][PTQ1K4_MAXNT]; __m512 accf[PTQ1K4_MAXNP][PTQ1K4_MAXNT]; __m512 dxv[PTQ1K4_MAXNP];
    for (int p = 0; p < NP; ++p) {
        for (int t = 0; t < NT; ++t) {
            accf[p][t] = first ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(msk[p], dst[p] + t * ld);
            acci[p][t] = _mm512_set1_epi32(as[t][FL == 1 ? b0 : b0 >> 2]);
        }
        dxv[p] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (sc[p] + (b0 >> 2) * PTQ1P_ROWS)));
    }
    for (int b = 0; b < nblk; ++b) {
#pragma GCC unroll 8
        for (int s = 0; s < 8; ++s) {
            for (int t = 0; t < NT; ++t) {
                const __m512i a = _mm512_set1_epi32(*(const int32_t *) (q[t] + 32 * b + 4 * s));
                for (int p = 0; p < NP; ++p) acci[p][t] = _mm512_dpbusd_epi32(acci[p][t], codes[p * cstride + 8 * b + s], a);
            }
        }
        const int64_t gb = b0 + b;
        if ((gb & 3) == 0) for (int p = 0; p < NP; ++p) dxv[p] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (sc[p] + (gb >> 2) * PTQ1P_ROWS)));
        if (FL == 4 && (gb & 3) != 3 && b != nblk - 1) continue;   // per-128 scales: flush at the group's end (chunks end on groups)
        const int64_t gi = FL == 1 ? gb : gb >> 2;
        for (int t = 0; t < NT; ++t) {
            const __m512 dyv = _mm512_set1_ps(dy[t][gi]);
            const __m512i nas = _mm512_set1_epi32(as[t][gi + 1]);
            for (int p = 0; p < NP; ++p) {
                accf[p][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acci[p][t]), _mm512_mul_ps(dxv[p], dyv), accf[p][t]);
                acci[p][t] = nas;
            }
        }
    }
    for (int p = 0; p < NP; ++p) for (int t = 0; t < NT; ++t) _mm512_mask_storeu_ps(dst[p] + t * ld, msk[p], accf[p][t]);
}

// panels [p0, p1) x all m tokens. NP, NT, KC, MB, PF are compile-time (always_inline from the wrappers below).
static inline __attribute__((always_inline)) void ptq1k4_gemm_i(
        const ptq1p_entry * e, const uint8_t * acts, size_t cb, int64_t m, int64_t p0, int64_t p1, float * dst, int64_t ld,
        const int NP, const int NT, const int KC, const int64_t MB, const int PF, const int FL) {
    __m512i codes[PTQ1K4_MAXNP * (KC + 4) * PTQ1V_STEPS] __attribute__((aligned(64)));
    const int cstride = (KC + 4) * PTQ1V_STEPS;
    const int64_t ng = FL == 1 ? e->kpad / 32 : (e->kpad + 127) / 128;   // scale entries per column
    const int64_t kpad = e->kpad, nsbt = e->nsb + e->n36;
    for (int64_t m0 = 0; m0 < m; m0 += MB) {
        const int64_t m1 = m0 + MB < m ? m0 + MB : m;
        int64_t s0 = 0, b0 = 0;
        for (int64_t sb = 0, sb1 = 0; sb < nsbt; sb = sb1) {   // the chunk may be extended past KC (below), so advance to sb1
            sb1 = sb + KC < nsbt ? sb + KC : nsbt;
            int steps = 0, bytes = 0;
            for (int64_t j = sb; j < sb1; ++j) { steps += j < e->nsb ? PTQ1V_STEPS : PTQ1W_STEPS; bytes += j < e->nsb ? PTQ1V_SBB : PTQ1W_SBB; }
            if (FL == 4) while (sb1 < nsbt && (b0 + steps / 8) % 4 != 0) {   // extend to a 128-k group boundary (at most 4 more superblocks)
                steps += sb1 < e->nsb ? PTQ1V_STEPS : PTQ1W_STEPS; bytes += sb1 < e->nsb ? PTQ1V_SBB : PTQ1W_SBB; ++sb1;
            }
            const int nblk = steps / 8;
            const int64_t choff = sb < e->nsb ? sb * PTQ1V_SBB : e->nsb * PTQ1V_SBB + (sb - e->nsb) * PTQ1W_SBB;   // chunk offset in a panel
            for (int64_t p = p0; p < p1; p += NP) {
                const int np = (int) (p1 - p < NP ? p1 - p : NP);
                const ggml_half * sc[PTQ1K4_MAXNP]; __mmask16 msk[PTQ1K4_MAXNP]; float * d[PTQ1K4_MAXNP];
                for (int i = 0; i < np; ++i) {
                    const uint8_t * panel = e->packed + (p + i) * e->panel_stride;
                    int o = 0;
                    for (int64_t j = sb; j < sb1; ++j) {
                        const int w = j >= e->nsb;
                        o += ptq1v_decode_sb(panel + (w ? e->nsb * PTQ1V_SBB + (j - e->nsb) * PTQ1W_SBB : j * PTQ1V_SBB), w, codes + i * cstride + o);
                    }
                    sc[i] = (const ggml_half *) (panel + ptq1v_data_bytes(e));
                    const int64_t row0 = (p + i) * PTQ1P_ROWS;
                    const int64_t nr = (e->rows - row0) < PTQ1P_ROWS ? (e->rows - row0) : PTQ1P_ROWS;
                    msk[i] = (__mmask16) ((1u << nr) - 1);
                    d[i] = dst + m0 * ld + row0;
                }
                if (PF) for (int i = 0; i < np && p + np + i < p1; ++i) {   // next panels' chunk: a 2 KB run that is otherwise a cold DRAM/L3 read
                    const uint8_t * nxt = e->packed + (p + np + i) * e->panel_stride + choff;
                    for (int o = 0; o < bytes; o += 64) _mm_prefetch((const char *) nxt + o, _MM_HINT_T1);
                }
                for (int64_t t0 = m0; t0 < m1; t0 += NT) {
                    const int nt = (int) (m1 - t0 < NT ? m1 - t0 : NT);
                    const int8_t * q[PTQ1K4_MAXNT]; const float * dy[PTQ1K4_MAXNT]; const int32_t * as[PTQ1K4_MAXNT];
                    for (int t = 0; t < nt; ++t) {
                        const uint8_t * base = acts + (size_t) (t0 + t) * cb;
                        q[t] = (const int8_t *) base + 4 * s0; dy[t] = (const float *) (base + kpad); as[t] = (const int32_t *) (dy[t] + ng);
                    }
                    float * dd[PTQ1K4_MAXNP];
                    for (int i = 0; i < np; ++i) dd[i] = d[i] + (t0 - m0) * ld;
#define PTQ1K4_T(P, n) ptq1k4_tile(codes, cstride, nblk, sc, b0, q, dy, as, dd, ld, msk, sb == 0, P, n, FL)
#define PTQ1K4_SW(P) switch (nt) { \
                        case 1: PTQ1K4_T(P, 1); break; case 2: PTQ1K4_T(P, 2); break; case 3: PTQ1K4_T(P, 3); break; \
                        case 4: PTQ1K4_T(P, 4); break; case 5: PTQ1K4_T(P, 5); break; case 6: PTQ1K4_T(P, 6); break; \
                        case 7: PTQ1K4_T(P, 7); break; case 8: PTQ1K4_T(P, 8); break; case 9: PTQ1K4_T(P, 9); break; \
                        case 10: PTQ1K4_T(P, 10); break; case 11: PTQ1K4_T(P, 11); break; default: PTQ1K4_T(P, 12); break; }
                    if (NP >= 3 && np == 3) { PTQ1K4_SW(3) }
                    else if (NP >= 2 && np == 2) { PTQ1K4_SW(2) }
                    else { PTQ1K4_SW(1) }
#undef PTQ1K4_SW
#undef PTQ1K4_T
                }
            }
            s0 += steps; b0 += nblk;
        }
    }
}

static void ptq1k4_gemm(const ptq1p_entry * e, const uint8_t * acts, size_t cb, int64_t m,
                        int64_t p0, int64_t p1, float * dst, int64_t ld) {
    ptq1k4_gemm_i(e, acts, cb, m, p0, p1, dst, ld, PTQ1K4_NP, PTQ1K4_NT, PTQ1K4_KC, PTQ1K4_MB, PTQ1K4_PF, 1);
}
// GGML_PTQ1_K4=2: activations quantized by ptq1k4_quantize_col128; one flush per 128 k (NOT bit-identical to modes 3-6)
static void ptq1k4_gemm2(const ptq1p_entry * e, const uint8_t * acts, size_t cb, int64_t m,
                         int64_t p0, int64_t p1, float * dst, int64_t ld) {
    ptq1k4_gemm_i(e, acts, cb, m, p0, p1, dst, ld, PTQ1K4_NP, PTQ1K4_NT, PTQ1K4_KC, PTQ1K4_MB, PTQ1K4_PF, 4);
}

static void ptq1v_gemm(const ptq1p_entry * e, const uint8_t * acts, size_t cb, int64_t m,
                       int64_t p0, int64_t p1, float * dst, int64_t ld) {
    for (int64_t p = p0; p < p1; ++p) {
        const uint8_t * panel = e->packed + p * e->panel_stride;
        for (int64_t t0 = 0; t0 < m; t0 += PTQ1P_TB) {
            const int nt = (int) (m - t0 < PTQ1P_TB ? m - t0 : PTQ1P_TB);
            switch (nt) {
                case 1: ptq1v_panel(e, panel, acts, cb, 1, PTQ1V_NCH, PTQ1V_PF, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 2: ptq1v_panel(e, panel, acts, cb, 2, PTQ1V_NCH, PTQ1V_PF, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 3: ptq1v_panel(e, panel, acts, cb, 3, PTQ1V_NCH, PTQ1V_PF, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 4: ptq1v_panel(e, panel, acts, cb, 4, PTQ1V_NCH, PTQ1V_PF, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 5: ptq1v_panel(e, panel, acts, cb, 5, PTQ1V_NCH, PTQ1V_PF, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 6: ptq1v_panel(e, panel, acts, cb, 6, PTQ1V_NCH, PTQ1V_PF, dst, ld, p * PTQ1P_ROWS, t0); break;
                case 7: ptq1v_panel(e, panel, acts, cb, 7, PTQ1V_NCH, PTQ1V_PF, dst, ld, p * PTQ1P_ROWS, t0); break;
                default: ptq1v_panel(e, panel, acts, cb, 8, PTQ1V_NCH, PTQ1V_PF, dst, ld, p * PTQ1P_ROWS, t0); break;
            }
        }
    }
}
#else
#define PTQ1V_OK 0
#endif
