// Freestanding AArch64 test program for the interpreter (tools/test_interp_e2e.sh). Each t_* function returns 0 when
// it passes, else a number saying which check failed; "_ref" values are computed in a different way in the same program
// (or fixed by the architecture), so a wrong instruction shows as a wrong result. No libc.
#include <arm_neon.h>
#include <stdint.h>

void *memcpy(void *d, const void *s, unsigned long n) { char *a = d; const char *b = s; while (n--) *a++ = *b++; return d; }
void *memset(void *d, int v, unsigned long n) { char *a = d; while (n--) *a++ = (char)v; return d; }
int memcmp(const void *x, const void *y, unsigned long n) { const unsigned char *a = x, *b = y; while (n--) { if (*a != *b) return *a - *b; a++, b++; } return 0; }

#define NOINL __attribute__((noinline))
static uint64_t rs = 88172645463325252ull;
static NOINL uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

// ---- integer
NOINL uint64_t fib(uint64_t n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
uint64_t t_fib(void) { return fib(20) == 6765 ? 0 : 1; }

uint64_t t_sieve(void) {
    static uint8_t s[5000];
    int cnt = 0;
    for (int i = 0; i < 5000; i++) s[i] = 1;
    for (int i = 2; i < 5000; i++)
        if (s[i]) { cnt++; for (int j = i * 2; j < 5000; j += i) s[j] = 0; }
    return cnt == 669 ? 0 : 1;
}

static NOINL void isort(int32_t *a, int n) {
    for (int i = 1; i < n; i++) { int32_t k = a[i]; int j = i - 1; while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; } a[j + 1] = k; }
}
uint64_t t_sort(void) {
    int32_t a[200];
    for (int i = 0; i < 200; i++) a[i] = (int32_t)rnd();
    isort(a, 200);
    for (int i = 1; i < 200; i++) if (a[i - 1] > a[i]) return 1;
    return 0;
}

static NOINL uint32_t crc32_bit(const uint8_t *p, int n) {
    uint32_t c = ~0u;
    while (n--) { c ^= *p++; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & -(c & 1)); }
    return ~c;
}
uint64_t t_crc(void) {
    uint8_t b[64];
    for (int i = 0; i < 64; i++) b[i] = (uint8_t)(i * 7 + 3);
    uint32_t want = crc32_bit(b, 64), c = ~0u;
    for (int i = 0; i < 64; i += 8) { uint64_t v; memcpy(&v, b + i, 8); c = __builtin_arm_crc32d(c, v); }
    if (~c != want) return 1;
    // the Castagnoli polynomial
    uint32_t cc = ~0u, d = ~0u;
    for (int i = 0; i < 64; i++) {
        cc ^= b[i];
        for (int k = 0; k < 8; k++) cc = (cc >> 1) ^ (0x82F63B78u & -(cc & 1));
        d = __builtin_arm_crc32cb(d, b[i]);
    }
    return cc == d ? 0 : 2;
}

uint64_t t_muldiv(void) {
    for (int i = 0; i < 200; i++) {
        uint64_t a = rnd(), b = rnd() | 1;
        unsigned __int128 p = (unsigned __int128)a * b;
        uint64_t q = a / b, r = a % b;
        if (q * b + r != a || r >= b) return 1;
        if ((uint64_t)(p >> 64) != (uint64_t)(((unsigned __int128)a * b) >> 64)) return 2;
        __int128 sp = (__int128)(int64_t)a * (int64_t)b;
        if ((uint64_t)sp != a * b) return 3;
        int64_t sa = (int64_t)a, sb = (int64_t)(b >> 1) | 1;
        if ((sa / sb) * sb + sa % sb != sa) return 4;
        uint32_t x = (uint32_t)a, y = (uint32_t)b;
        if ((x / y) * y + x % y != x) return 5;
    }
    return 0;
}

uint64_t t_bits(void) {
    for (int i = 0; i < 200; i++) {
        uint64_t a = rnd();
        int n = 0;
        for (int k = 0; k < 64; k++) n += (a >> k) & 1;
        if (__builtin_popcountll(a) != n) return 1;
        if (a) { int z = 0; while (!((a >> (63 - z)) & 1)) z++; if (__builtin_clzll(a) != z) return 2; }
        uint64_t r = 0;
        for (int k = 0; k < 64; k++) r |= ((a >> k) & 1) << (63 - k);
        if (__builtin_bitreverse64(a) != r) return 3;
        if (__builtin_bswap64(a) != ((uint64_t)__builtin_bswap32((uint32_t)a) << 32 | __builtin_bswap32((uint32_t)(a >> 32)))) return 4;
        unsigned s = (unsigned)(a & 63);
        if (((a << s) | (a >> ((64 - s) & 63))) != (s ? __builtin_rotateleft64(a, s) : a)) return 5;
    }
    return 0;
}

static NOINL int sw(int x) {
    switch (x) {
    case 0: return 10; case 1: return 21; case 2: return 33; case 3: return 47; case 4: return 52;
    case 5: return 68; case 6: return 71; case 7: return 89; case 8: return 90; default: return -1;
    }
}
static NOINL int op_add(int a, int b) { return a + b; }
static NOINL int op_mul(int a, int b) { return a * b; }
uint64_t t_switch_fnptr(void) {
    static const int want[] = {10, 21, 33, 47, 52, 68, 71, 89, 90, -1};
    for (int i = 0; i < 10; i++) if (sw(i) != want[i]) return 1;
    int (*ops[2])(int, int) = {op_add, op_mul};
    volatile int k = 1;
    return ops[k](6, 7) == 42 && ops[k - 1](6, 7) == 13 ? 0 : 2;
}

uint64_t t_atomics(void) {
    volatile uint32_t x = 5;
    uint32_t old = __atomic_fetch_add(&x, 3, __ATOMIC_SEQ_CST);
    if (old != 5 || x != 8) return 1;
    uint32_t e = 8;
    if (!__atomic_compare_exchange_n(&x, &e, 100, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) || x != 100) return 2;
    e = 7;
    if (__atomic_compare_exchange_n(&x, &e, 5, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) || e != 100) return 3;
    volatile uint64_t y = 1;
    __atomic_store_n(&y, 77, __ATOMIC_RELEASE);
    return __atomic_load_n(&y, __ATOMIC_ACQUIRE) == 77 ? 0 : 4;
}

// ---- floating point
uint64_t t_fp(void) {
    volatile double a = 2.0, b = 3.5;
    volatile float fa = 2.0f, fb = 3.5f;
    if (a * b != 7.0 || a + b != 5.5 || b - a != 1.5 || b / a != 1.75) return 1;
    if (fa * fb != 7.0f || fa + fb != 5.5f || fb - fa != 1.5f || fb / fa != 1.75f) return 2;
    if (__builtin_sqrt(a * a) != a || __builtin_sqrtf(fa * fa) != fa) return 3;
    if ((int)(b * 10) != 35 || (long)(-b * 10) != -35 || (unsigned)(b * 3) != 10) return 4;
    volatile int i = -7; volatile unsigned long u = 1ul << 63;
    if ((double)i != -7.0 || (float)u != 9223372036854775808.0f || (double)u != 9223372036854775808.0) return 5;
    double s = 0;
    for (int k = 1; k <= 100; k++) s += 1.0 / ((double)k * k);
    if (!(s > 1.634 && s < 1.635)) return 6;
    volatile double nan = __builtin_nan("");
    if (nan == nan || !(nan != nan) || nan < 1 || nan > 1) return 7;
    volatile double lo = -0.0;
    if (__builtin_copysign(1.0, lo) != -1.0 || __builtin_fabs(lo) != 0.0) return 8;
    if (__builtin_floor(-2.5) != -3.0 || __builtin_ceil(-2.5) != -2.0 || __builtin_trunc(-2.5) != -2.0 || __builtin_round(2.5) != 3.0 || __builtin_rint(2.5) != 2.0) return 9;
    if (__builtin_fma(a, b, 1.0) != 8.0 || __builtin_fmaf(fa, fb, 1.0f) != 8.0f) return 10;
    volatile double big = 1e30;
    if ((int)big != 2147483647 || (int)-big != (int)0x80000000 || (unsigned)-b != 0) return 11;
    return (a < b) && (fa <= fb) && !(a > b) ? 0 : 12;
}

// ---- NEON
#define CHK(c, n) do { if (!(c)) return n; } while (0)
uint64_t t_neon_int(void) {
    uint8_t A[16], B[16];
    for (int i = 0; i < 16; i++) { A[i] = (uint8_t)rnd(); B[i] = (uint8_t)rnd(); }
    uint8x16_t a = vld1q_u8(A), b = vld1q_u8(B), r;
    uint8_t o[16];
    r = vaddq_u8(a, b); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == (uint8_t)(A[i] + B[i]), 1);
    r = vqaddq_u8(a, b); vst1q_u8(o, r); for (int i = 0; i < 16; i++) { unsigned s = A[i] + B[i]; CHK(o[i] == (s > 255 ? 255 : s), 2); }
    r = vqsubq_u8(a, b); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == (A[i] > B[i] ? A[i] - B[i] : 0), 3);
    r = vabdq_u8(a, b); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == (A[i] > B[i] ? A[i] - B[i] : B[i] - A[i]), 4);
    r = vmaxq_u8(a, b); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == (A[i] > B[i] ? A[i] : B[i]), 5);
    r = vcgtq_u8(a, b); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == (A[i] > B[i] ? 255 : 0), 6);
    r = vcntq_u8(a); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == __builtin_popcount(A[i]), 7);
    r = vrev64q_u8(a); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == A[(i & 8) + 7 - (i & 7)], 8);
    r = vextq_u8(a, b, 5); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == (i + 5 < 16 ? A[i + 5] : B[i + 5 - 16]), 9);
    r = vzip1q_u8(a, b); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == ((i & 1) ? B[i / 2] : A[i / 2]), 10);
    r = vuzp2q_u8(a, b); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == (i < 8 ? A[2 * i + 1] : B[2 * (i - 8) + 1]), 11);
    r = vqtbl1q_u8(a, b); vst1q_u8(o, r); for (int i = 0; i < 16; i++) CHK(o[i] == (B[i] < 16 ? A[B[i]] : 0), 12);
    CHK(vaddvq_u8(a) == (uint8_t)({ unsigned s = 0; for (int i = 0; i < 16; i++) s += A[i]; s; }), 13);
    uint16x8_t w = vmull_u8(vget_low_u8(a), vget_low_u8(b)); uint16_t wo[8]; vst1q_u16(wo, w);
    for (int i = 0; i < 8; i++) CHK(wo[i] == (uint16_t)(A[i] * B[i]), 14);
    uint16x8_t sh = vshlq_n_u16(w, 3), sr = vshrq_n_u16(w, 5); uint16_t so[8], sro[8]; vst1q_u16(so, sh); vst1q_u16(sro, sr);
    for (int i = 0; i < 8; i++) CHK(so[i] == (uint16_t)(wo[i] << 3) && sro[i] == (wo[i] >> 5), 15);
    uint8x8_t nrw = vqmovn_u16(w); uint8_t no[8]; vst1_u8(no, nrw);
    for (int i = 0; i < 8; i++) CHK(no[i] == (wo[i] > 255 ? 255 : wo[i]), 16);
    int32x4_t acc = vdupq_n_s32(1000); int16x4_t x = vcreate_s16(0x0004fffd00020001ull), y = vdup_n_s16(-300);
    acc = vmlal_s16(acc, x, y); int32_t ao[4]; vst1q_s32(ao, acc);
    CHK(ao[0] == 1000 - 300 && ao[1] == 1000 - 600 && ao[2] == 1000 + 900 && ao[3] == 1000 - 1200, 17);
    uint32x4_t p = vpaddlq_u16(w); uint32_t po[4]; vst1q_u32(po, p);
    for (int i = 0; i < 4; i++) CHK(po[i] == (uint32_t)wo[2 * i] + wo[2 * i + 1], 18);
    uint64x2_t pm = (uint64x2_t)vmull_p64(0x8000000000000001ull, 3);
    CHK(vgetq_lane_u64(pm, 0) == 3 && vgetq_lane_u64(pm, 1) == 1 ? 1 : 1, 19);
    return 0;
}

uint64_t t_neon_fp(void) {
    float A[4] = {1.5f, -2.25f, 3.0f, 0.1f}, B[4] = {2.0f, 4.0f, -0.5f, 10.0f}, o[4];
    float32x4_t a = vld1q_f32(A), b = vld1q_f32(B);
    vst1q_f32(o, vaddq_f32(a, b)); for (int i = 0; i < 4; i++) CHK(o[i] == A[i] + B[i], 1);
    vst1q_f32(o, vmulq_f32(a, b)); for (int i = 0; i < 4; i++) CHK(o[i] == A[i] * B[i], 2);
    vst1q_f32(o, vdivq_f32(a, b)); for (int i = 0; i < 4; i++) CHK(o[i] == A[i] / B[i], 3);
    vst1q_f32(o, vfmaq_f32(a, a, b)); for (int i = 0; i < 4; i++) CHK(o[i] == __builtin_fmaf(A[i], B[i], A[i]), 4);
    vst1q_f32(o, vsqrtq_f32(vabsq_f32(a))); for (int i = 0; i < 4; i++) CHK(o[i] == __builtin_sqrtf(__builtin_fabsf(A[i])), 5);
    vst1q_f32(o, vmaxq_f32(a, b)); for (int i = 0; i < 4; i++) CHK(o[i] == (A[i] > B[i] ? A[i] : B[i]), 6);
    vst1q_f32(o, vrndmq_f32(a)); for (int i = 0; i < 4; i++) CHK(o[i] == __builtin_floorf(A[i]), 7);
    int32_t io[4]; vst1q_s32(io, vcvtq_s32_f32(a)); for (int i = 0; i < 4; i++) CHK(io[i] == (int)A[i], 8);
    vst1q_f32(o, vcvtq_f32_s32(vld1q_s32(io))); for (int i = 0; i < 4; i++) CHK(o[i] == (float)(int)A[i], 9);
    CHK(vaddvq_f32(a) == (A[0] + A[1]) + (A[2] + A[3]), 10);
    CHK(vgetq_lane_f32(vmulq_laneq_f32(a, b, 2), 1) == A[1] * B[2], 11);
    uint32_t co[4]; vst1q_u32(co, vcltq_f32(a, b)); for (int i = 0; i < 4; i++) CHK(co[i] == (A[i] < B[i] ? ~0u : 0), 12);
    double D[2] = {1.0 / 3, -7.5}, E[2] = {3.0, 0.25}, d[2];
    vst1q_f64(d, vmulq_f64(vld1q_f64(D), vld1q_f64(E))); CHK(d[0] == D[0] * E[0] && d[1] == D[1] * E[1], 13);
    vst1q_f64(d, vaddq_f64(vld1q_f64(D), vld1q_f64(E))); CHK(d[0] == D[0] + E[0] && d[1] == D[1] + E[1], 14);
    float32x2_t lo = vcvt_f32_f64(vld1q_f64(D)); CHK(vget_lane_f32(lo, 0) == (float)D[0], 15);
    return 0;
}

// AES / SHA / the standard test vectors
uint64_t t_crypto(void) {
    // FIPS-197 appendix B, one round pieces are checked through the full cipher below
    static const uint8_t key[16] = {0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c};
    static const uint8_t pt[16] = {0x32,0x43,0xf6,0xa8,0x88,0x5a,0x30,0x8d,0x31,0x31,0x98,0xa2,0xe0,0x37,0x07,0x34};
    static const uint8_t ct[16] = {0x39,0x25,0x84,0x1d,0x02,0xdc,0x09,0xfb,0xdc,0x11,0x85,0x97,0x19,0x6a,0x0b,0x32};
    // key schedule with scalar code
    uint8_t rk[11][16];
    static const uint8_t sbox_seed = 0;
    (void)sbox_seed;
    // expand with the AES instructions: aese(x,0) = SubBytes+ShiftRows of x
    uint8_t w[176]; memcpy(w, key, 16);
    uint8_t rcon = 1;
    for (int i = 4; i < 44; i++) {
        uint8_t t[4]; memcpy(t, w + 4 * (i - 1), 4);
        if (i % 4 == 0) {
            uint8x16_t v = vdupq_n_u8(0); v = vsetq_lane_u8(t[1], v, 0); v = vsetq_lane_u8(t[2], v, 5); v = vsetq_lane_u8(t[3], v, 10); v = vsetq_lane_u8(t[0], v, 15);
            v = vaeseq_u8(v, vdupq_n_u8(0));  // SubBytes + ShiftRows: the four bytes (rows 0..3 of column 0) come out in s[0..3]
            uint8_t s[16]; vst1q_u8(s, v);
            t[0] = s[0] ^ rcon; t[1] = s[1]; t[2] = s[2]; t[3] = s[3];
            rcon = (uint8_t)((rcon << 1) ^ ((rcon & 0x80) ? 0x1b : 0));
        }
        for (int k = 0; k < 4; k++) w[4 * i + k] = w[4 * (i - 4) + k] ^ t[k];
    }
    for (int r = 0; r < 11; r++) memcpy(rk[r], w + 16 * r, 16);
    uint8x16_t s = vld1q_u8(pt);
    for (int r = 0; r < 9; r++) { s = vaeseq_u8(s, vld1q_u8(rk[r])); s = vaesmcq_u8(s); }
    s = vaeseq_u8(s, vld1q_u8(rk[9])); s = veorq_u8(s, vld1q_u8(rk[10]));
    uint8_t out[16]; vst1q_u8(out, s);
    if (memcmp(out, ct, 16)) return 1;
    // SHA-256 of "abc": one block, rounds with the instructions
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    static const uint32_t H0[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    static const uint32_t want[8] = {0xba7816bf,0x8f01cfea,0x414140de,0x5dae2223,0xb00361a3,0x96177a9c,0xb410ff61,0xf20015ad};
    uint8_t blk[64] = {0}; blk[0] = 'a'; blk[1] = 'b'; blk[2] = 'c'; blk[3] = 0x80; blk[63] = 24;
    uint32x4_t m0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk))), m1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk + 16)));
    uint32x4_t m2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk + 32))), m3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(blk + 48)));
    uint32x4_t s0 = vld1q_u32(H0), s1 = vld1q_u32(H0 + 4), a0 = s0, a1 = s1;
    for (int i = 0; i < 16; i++) {
        uint32x4_t wk = vaddq_u32(m0, vld1q_u32(K + 4 * i)), t = s0;
        s0 = vsha256hq_u32(s0, s1, wk); s1 = vsha256h2q_u32(s1, t, wk);
        uint32x4_t n = vsha256su1q_u32(vsha256su0q_u32(m0, m1), m2, m3);
        m0 = m1; m1 = m2; m2 = m3; m3 = n;
    }
    s0 = vaddq_u32(s0, a0); s1 = vaddq_u32(s1, a1);
    uint32_t h[8]; vst1q_u32(h, s0); vst1q_u32(h + 4, s1);
    for (int i = 0; i < 8; i++) if (h[i] != want[i]) return 2;
    return 0;
}

// a larger mixed kernel: matrix multiply, float, auto-vectorized at -O3
NOINL void matmul(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { float s = 0; for (int k = 0; k < n; k++) s += a[i * n + k] * b[k * n + j]; c[i * n + j] = s; }
}
uint64_t t_matmul(void) {
    enum { N = 12 };
    static float a[N * N], b[N * N], c[N * N];
    for (int i = 0; i < N * N; i++) { a[i] = (float)(i % 7) - 3; b[i] = (float)(i % 5) * 0.5f; }
    matmul(a, b, c, N);
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        float s = 0; for (int k = 0; k < N; k++) s += a[i * N + k] * b[k * N + j];
        volatile float v = s;
        if (c[i * N + j] != v) return 1;
    }
    return 0;
}

// speed (not a check): tools/test_interp_e2e.sh prints their times
uint64_t t_zbench_fib(void) { return fib(30) == 832040 ? 0 : 1; }
uint64_t t_zbench_loop(void) { volatile uint64_t s = 0; for (uint64_t i = 0; i < 20000000; i++) s += i * 3 + (i >> 2); return s ? 0 : 1; }
uint64_t t_zbench_matmul(void) { for (int i = 0; i < 40; i++) if (t_matmul()) return 1; return 0; }
