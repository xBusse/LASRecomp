// FP / AdvSIMD / crypto / CRC32 and their loads and stores for the interpreter: a port of tools/simd_c.py (and the
// matching parts of tools/lifter.py). simd_c.py decodes a word at generation time and emits C calling the helpers in
// runtime/simd.h with constant fields; here the same decoding runs when the instruction executes and calls the same
// helpers with the fields as values, so the results are the generated code's, bit for bit. Where simd_c.py emits a
// vector expression (VW(vu8, Q, d, a + b)), a size switch (BYSZ) picks the vector type here. Encodings the lifter
// rejects return false.
//
// Everything is decoded on every execution: the cost of a handful of mask tests next to the arithmetic. (An
// instruction-specific predecode could replace the hottest ones later.)
#include <string.h>

#include "interp.h"
#include "simd.h"

static inline unsigned bits(uint32_t w, int lo, int n) { return (w >> lo) & ((1u << n) - 1u); }
static inline int64_t sext(uint64_t v, int n) { return (int64_t)(v << (64 - n)) >> (64 - n); }

static inline uint64_t rx(Cpu *c, unsigned r) { return r == 31 ? 0 : c->x[r]; }    // X(r): zr
static inline uint64_t rxs(Cpu *c, unsigned r) { return r == 31 ? c->sp : c->x[r]; } // XS(r): sp
static inline void setx(Cpu *c, unsigned r, uint64_t v) { if (r != 31) c->x[r] = v; }
static inline void setw(Cpu *c, unsigned r, uint32_t v) { if (r != 31) c->x[r] = v; }
static inline void setxs(Cpu *c, unsigned r, uint64_t v) { if (r == 31) c->sp = v; else c->x[r] = v; }

#define LANES(Q, sz) ((8 << (Q)) >> (sz))

// Runs BODY with T / S / E the unsigned vector / signed vector / unsigned element type of 8 << sz bit lanes.
#define BYSZ(sz, ...)                                                                                          \
    switch (sz) {                                                                                              \
    case 0: { typedef vu8 T; typedef vs8 S; typedef uint8_t E; (void)(T *)0; (void)(S *)0; (void)(E *)0; __VA_ARGS__ } break;     \
    case 1: { typedef vu16 T; typedef vs16 S; typedef uint16_t E; (void)(T *)0; (void)(S *)0; (void)(E *)0; __VA_ARGS__ } break; \
    case 2: { typedef vu32 T; typedef vs32 S; typedef uint32_t E; (void)(T *)0; (void)(S *)0; (void)(E *)0; __VA_ARGS__ } break; \
    default: { typedef vu64 T; typedef vs64 S; typedef uint64_t E; (void)(T *)0; (void)(S *)0; (void)(E *)0; __VA_ARGS__ } break; \
    }

#define LOPU(name, u) (LOP_##name | ((u) ? LOP_U : 0))

// the lane generator, with the arguments simd_c.py's `gen` / `same` helpers pass
static void gen(Cpu *c, int op, int sz, int ne, int d, int szd, int offd, int keep, int n, int szn, int offn, int m,
                int szm, int offm, int bidx, uint64_t bimm) {
    sl_gen(c, op, sz, ne, d, szd, offd, keep, n, szn, offn, m, szm, offm, bidx, bimm);
}
static void same(Cpu *c, int op, int sz, int ne, int d, int n, int m, uint64_t bimm, int bidx) {
    sl_gen(c, op, sz, ne, d, sz, 0, 0, n, sz, 0, m, sz, 0, bidx, bimm);
}

// ---------------------------------------------------------------- CRC32

uint32_t interp_crc(int castagnoli, uint32_t acc, uint64_t v, int bytes) {
    return castagnoli ? CRC32C(acc, v, bytes) : CRC32(acc, v, bytes);
}

// ---------------------------------------------------------------- immediates

static uint64_t vfp_expand_imm(unsigned imm8, int N) {
    int E = N == 32 ? 8 : 11, F = N - E - 1;
    uint64_t sign = imm8 >> 7, b6 = (imm8 >> 6) & 1;
    uint64_t exp = ((b6 ^ 1) << (E - 1)) | ((b6 ? ((1ull << (E - 3)) - 1) : 0) << 2) | ((imm8 >> 4) & 3);
    uint64_t frac = (uint64_t)(imm8 & 15) << (F - 4);
    return (sign << (N - 1)) | (exp << F) | frac;
}

static uint64_t simd_expand_imm(unsigned op, unsigned cmode, uint64_t imm8) {
    unsigned cc = cmode >> 1;
    uint64_t imm = 0;
    switch (cc) {
    case 0: imm = (imm8 << 32) | imm8; break;
    case 1: imm = ((imm8 << 8) << 32) | (imm8 << 8); break;
    case 2: imm = ((imm8 << 16) << 32) | (imm8 << 16); break;
    case 3: imm = ((imm8 << 24) << 32) | (imm8 << 24); break;
    case 4: for (int s = 0; s < 64; s += 16) imm |= imm8 << s; break;
    case 5: for (int s = 0; s < 64; s += 16) imm |= (imm8 << 8) << s; break;
    case 6: {
        uint64_t v = !(cmode & 1) ? ((imm8 << 8) | 0xFF) : ((imm8 << 16) | 0xFFFF);
        imm = (v << 32) | v;
        break;
    }
    default:
        if (!(cmode & 1) && !op) {
            for (int s = 0; s < 64; s += 8) imm |= imm8 << s;
        } else if (!(cmode & 1) && op) {
            for (int i = 0; i < 8; i++)
                if ((imm8 >> i) & 1) imm |= 0xFFull << (8 * i);
        } else if ((cmode & 1) && !op) {
            uint64_t v = vfp_expand_imm((unsigned)imm8, 32);
            imm = (v << 32) | v;
        } else {
            imm = vfp_expand_imm((unsigned)imm8, 64);
        }
    }
    return imm;
}

static inline int condv(Cpu *c, unsigned cc) {
    if (cc >= 14) return 1;
    unsigned n = (c->nzcv >> 31) & 1, z = (c->nzcv >> 30) & 1, cy = (c->nzcv >> 29) & 1, v = (c->nzcv >> 28) & 1;
    unsigned r;
    switch (cc >> 1) {
    case 0: r = z; break;
    case 1: r = cy; break;
    case 2: r = n; break;
    case 3: r = v; break;
    case 4: r = cy && !z; break;
    case 5: r = n == v; break;
    default: r = n == v && !z; break;
    }
    return (cc & 1) ? !r : r;
}

// ---------------------------------------------------------------- scalar FP

static inline int ftype(uint32_t w, int *t) {
    *t = (int)bits(w, 22, 2);
    return *t <= 1;
}

static bool fp_dp3(Cpu *c, uint32_t w) {
    int t;
    if (!ftype(w, &t)) return false;
    unsigned o1 = bits(w, 21, 1), o0 = bits(w, 15, 1), d = bits(w, 0, 5), n = bits(w, 5, 5), m = bits(w, 16, 5),
             a = bits(w, 10, 5);
    int na = (int)o1, nn = (int)(o1 ^ o0);
    if (!t) fp_fma_s(c, na, nn, (int)d, (int)n, (int)m, (int)a);
    else fp_fma_d(c, na, nn, (int)d, (int)n, (int)m, (int)a);
    return true;
}

static bool fp_dp2(Cpu *c, uint32_t w) {
    int t;
    if (!ftype(w, &t)) return false;
    unsigned opc = bits(w, 12, 4), d = bits(w, 0, 5), n = bits(w, 5, 5), m = bits(w, 16, 5);
    static const int fop[9] = {FOP_MUL, FOP_DIV, FOP_ADD, FOP_SUB, FOP_MAX, FOP_MIN, FOP_MAXNM, FOP_MINNM, FOP_NMUL};
    if (opc > 8) return false;
    if (!t) fp_op_s(c, fop[opc], (int)d, (int)n, (int)m);
    else fp_op_d(c, fop[opc], (int)d, (int)n, (int)m);
    return true;
}

static bool fp_dp1(Cpu *c, uint32_t w) {
    int t = (int)bits(w, 22, 2);
    unsigned opc = bits(w, 15, 6), d = bits(w, 0, 5), n = bits(w, 5, 5);
    if (t == 3) {  // half source: only FCVT
        if (opc == 4) { FP_SETS(d, simd_h2s(c, VS16(n))); return true; }
        if (opc == 5) { FP_SETD(d, simd_h2d(c, VS16(n))); return true; }
        return false;
    }
    if (t == 2) return false;
    uint64_t sign = t ? 0x8000000000000000ull : 0x80000000u;
    uint64_t su = t ? (uint64_t)VS64(n) : (uint64_t)VS32(n);
    if (opc == 0) { c->v[d] = su; return true; }
    if (opc == 1) { c->v[d] = su & ~sign; return true; }
    if (opc == 2) { c->v[d] = su ^ sign; return true; }
    if (opc == 3) {
        if (!t) fp_op_s(c, FOP_SQRT, (int)d, (int)n, (int)n);
        else fp_op_d(c, FOP_SQRT, (int)d, (int)n, (int)n);
        return true;
    }
    if (opc == 4 || opc == 5 || opc == 7) {
        int to = opc == 4 ? 0 : opc == 5 ? 1 : 3;
        if (to == t) return false;
        if (to == 3) {
            c->v[d] = t ? simd_dd2h(c, VSF64(n)) : simd_s2h(c, VSF32(n));
        } else if (to) {
            FP_SETD(d, simd_s2d(c, VSF32(n)));
        } else {
            FP_SETS(d, simd_d2s(c, VSF64(n)));
        }
        return true;
    }
    int rm;
    switch (opc) {
    case 8: rm = 0; break;
    case 9: rm = 1; break;
    case 10: rm = 2; break;
    case 11: rm = 3; break;
    case 12: rm = 4; break;
    case 14: rm = 5; break;
    case 15: rm = 6; break;
    default: return false;
    }
    if (!t) FP_SETS(d, frint_s(c, VSF32(n), rm));
    else FP_SETD(d, frint_d(c, VSF64(n), rm));
    return true;
}

static bool fp_cmp(Cpu *c, uint32_t w) {
    int t;
    if (!ftype(w, &t)) return false;
    if (bits(w, 14, 2) || bits(w, 0, 3)) return false;
    unsigned n = bits(w, 5, 5), m = bits(w, 16, 5), zero = bits(w, 3, 1);
    if (!t) c->nzcv = cmp_s(c, VSF32(n), zero ? 0.0f : VSF32(m));
    else c->nzcv = cmp_d(c, VSF64(n), zero ? 0.0 : VSF64(m));
    return true;
}

static bool fp_ccmp(Cpu *c, uint32_t w) {
    int t;
    if (!ftype(w, &t)) return false;
    unsigned n = bits(w, 5, 5), m = bits(w, 16, 5), cnd = bits(w, 12, 4), nzcv = bits(w, 0, 4);
    if (condv(c, cnd)) c->nzcv = t ? cmp_d(c, VSF64(n), VSF64(m)) : cmp_s(c, VSF32(n), VSF32(m));
    else c->nzcv = (uint64_t)nzcv << 28;
    return true;
}

static bool fp_csel(Cpu *c, uint32_t w) {
    int t;
    if (!ftype(w, &t)) return false;
    unsigned d = bits(w, 0, 5), n = bits(w, 5, 5), m = bits(w, 16, 5), cnd = bits(w, 12, 4);
    unsigned s = condv(c, cnd) ? n : m;
    c->v[d] = t ? (uint64_t)VS64(s) : (uint64_t)VS32(s);
    return true;
}

static bool fp_imm(Cpu *c, uint32_t w) {
    int t;
    if (!ftype(w, &t)) return false;
    if (bits(w, 5, 5)) return false;
    unsigned imm8 = bits(w, 13, 8), d = bits(w, 0, 5);
    c->v[d] = vfp_expand_imm(imm8, 32 << t);
    return true;
}

static bool fp_fixed(Cpu *c, uint32_t w) {
    unsigned sf = bits(w, 31, 1), rmode = bits(w, 19, 2), opc = bits(w, 16, 3), scale = bits(w, 10, 6), n = bits(w, 5, 5),
             d = bits(w, 0, 5);
    int t = (int)bits(w, 22, 2);
    if (t > 1) return false;
    if (!sf && scale < 32) return false;
    int fb = 64 - (int)scale, ib = sf ? 64 : 32;
    if (rmode == 0 && (opc == 2 || opc == 3)) {
        uint64_t src = sf ? rx(c, n) : (uint32_t)rx(c, n);
        if (!t) FP_SETS(d, fromint_s(c, src, ib, (int)(opc & 1), fb));
        else FP_SETD(d, fromint_d(c, src, ib, (int)(opc & 1), fb));
        return true;
    }
    if (rmode == 3 && (opc == 0 || opc == 1)) {
        uint64_t r = !t ? toint_s(c, VSF32(n), 3, fb, ib, (int)opc) : toint_d(c, VSF64(n), 3, fb, ib, (int)opc);
        if (sf) setx(c, d, r); else setw(c, d, (uint32_t)r);
        return true;
    }
    return false;
}

static bool fp_int(Cpu *c, uint32_t w) {
    unsigned sf = bits(w, 31, 1), rmode = bits(w, 19, 2), opc = bits(w, 16, 3), n = bits(w, 5, 5), d = bits(w, 0, 5);
    int t = (int)bits(w, 22, 2);
    if (opc == 6 || opc == 7) {  // FMOV
        if (t == 2 && sf && rmode == 1) {
            if (opc == 6) setx(c, d, VHI64(n));
            else VHI64(d) = rx(c, n);
            return true;
        }
        if (rmode || (unsigned)t != sf || t > 1) return false;
        if (opc == 6) {
            if (sf) setx(c, d, VS64(n)); else setw(c, d, VS32(n));
        } else {
            c->v[d] = sf ? rx(c, n) : (uint32_t)rx(c, n);
        }
        return true;
    }
    if (t > 1) return false;
    int ib = sf ? 64 : 32;
    if (opc == 2 || opc == 3) {
        if (rmode) return false;
        uint64_t src = sf ? rx(c, n) : (uint32_t)rx(c, n);
        if (!t) FP_SETS(d, fromint_s(c, src, ib, (int)(opc & 1), 0));
        else FP_SETD(d, fromint_d(c, src, ib, (int)(opc & 1), 0));
        return true;
    }
    int rm;
    if (opc == 4 || opc == 5) {
        if (rmode) return false;
        rm = 4;
    } else {
        rm = (int)rmode;
    }
    uint64_t r = !t ? toint_s(c, VSF32(n), rm, 0, ib, (int)(opc & 1)) : toint_d(c, VSF64(n), rm, 0, ib, (int)(opc & 1));
    if (sf) setx(c, d, r); else setw(c, d, (uint32_t)r);
    return true;
}

// ---------------------------------------------------------------- loads / stores

static inline u128 ld_scale(uint64_t ea, int scale) {
    switch (scale) {
    case 0: return LD8(ea);
    case 1: return LD16(ea);
    case 2: return LD32(ea);
    case 3: return LD64(ea);
    default: return LD128(ea);
    }
}
static inline void st_scale(uint64_t ea, int scale, const u128 *v) {
    switch (scale) {
    case 0: ST8(ea, *(const uint8_t *)v); break;
    case 1: ST16(ea, *(const sa16 *)v); break;
    case 2: ST32(ea, *(const sa32 *)v); break;
    case 3: ST64(ea, *(const sa64 *)v); break;
    default: ST128(ea, *v); break;
    }
}

static bool sldst_pair(Cpu *c, uint32_t w) {
    unsigned opc = bits(w, 30, 2), mode = bits(w, 23, 2), L = bits(w, 22, 1);
    int64_t imm7 = sext(bits(w, 15, 7), 7);
    unsigned rt2 = bits(w, 10, 5), rn = bits(w, 5, 5), rt = bits(w, 0, 5);
    if (opc == 3) return false;
    int scale = 2 + (int)opc;
    int64_t off = imm7 * ((int64_t)1 << scale);
    uint64_t base = rxs(c, rn);
    int post = mode == 1, wb = mode == 1 || mode == 3;
    uint64_t ea = post ? base : base + (uint64_t)off;
    uint64_t sz = 1ull << scale;
    if (L) {
        u128 a = ld_scale(ea, scale), b = ld_scale(ea + sz, scale);
        c->v[rt] = a;
        c->v[rt2] = b;
    } else {
        u128 a = c->v[rt], b = c->v[rt2];
        st_scale(ea, scale, &a);
        st_scale(ea + sz, scale, &b);
    }
    if (wb) setxs(c, rn, post ? ea + (uint64_t)off : ea);
    return true;
}

static bool sldst_reg(Cpu *c, uint32_t w) {
    unsigned size = bits(w, 30, 2), opc = bits(w, 22, 2), rn = bits(w, 5, 5), rt = bits(w, 0, 5);
    int scale;
    if (opc & 2) {
        if (size) return false;
        scale = 4;
    } else {
        scale = (int)size;
    }
    unsigned L = opc & 1;
    uint64_t base = rxs(c, rn), ea;
    int wb = 0;
    uint64_t wbv = 0;
    if (bits(w, 24, 2) == 1) {
        ea = base + ((uint64_t)bits(w, 10, 12) << scale);
    } else if (bits(w, 24, 2) == 0) {
        unsigned kind = bits(w, 10, 2);
        if (bits(w, 21, 1)) {
            if (kind != 2) return false;
            unsigned rm = bits(w, 16, 5), option = bits(w, 13, 3), S = bits(w, 12, 1);
            if (option != 2 && option != 3 && option != 6 && option != 7) return false;
            uint64_t xm = rx(c, rm), e;
            switch (option) {
            case 2: e = (uint32_t)xm; break;
            case 6: e = (uint64_t)(int64_t)(int32_t)xm; break;
            default: e = xm; break;
            }
            ea = base + (S ? e << scale : e);
        } else {
            int64_t imm9 = sext(bits(w, 12, 9), 9);
            if (kind == 0) {
                ea = base + (uint64_t)imm9;
            } else if (kind == 1) {
                ea = base;
                wb = 1;
                wbv = base + (uint64_t)imm9;
            } else if (kind == 3) {
                ea = base + (uint64_t)imm9;
                wb = 1;
                wbv = ea;
            } else {
                return false;
            }
        }
    } else {
        return false;
    }
    if (L) {
        c->v[rt] = ld_scale(ea, scale);
    } else {
        u128 v = c->v[rt];
        st_scale(ea, scale, &v);
    }
    if (wb) setxs(c, rn, wbv);
    return true;
}

static bool sldst_struct(Cpu *c, uint32_t w) {
    unsigned Q = bits(w, 30, 1), L = bits(w, 22, 1), post = bits(w, 23, 1), rm = bits(w, 16, 5), rn = bits(w, 5, 5),
             rt = bits(w, 0, 5), sz = bits(w, 10, 2);
    if (!post && rm) return false;
    uint64_t ea = rxs(c, rn);
    uint64_t total;
    if (bits(w, 24, 1) == 0) {  // multiple structures
        if (bits(w, 21, 1)) return false;
        unsigned opc = bits(w, 12, 4);
        int selem, rpt;
        switch (opc) {
        case 0: selem = 4, rpt = 1; break;
        case 2: selem = 1, rpt = 4; break;
        case 4: selem = 3, rpt = 1; break;
        case 6: selem = 1, rpt = 3; break;
        case 7: selem = 1, rpt = 1; break;
        case 8: selem = 2, rpt = 1; break;
        case 10: selem = 1, rpt = 2; break;
        default: return false;
        }
        if (sz == 3 && !Q && selem > 1) return false;
        total = (uint64_t)(8 << Q) * (uint64_t)selem * (uint64_t)rpt;
        if (selem == 1) {
            uint64_t step = Q ? 16 : 8;
            for (int i = 0; i < rpt; i++) {
                unsigned r = (rt + (unsigned)i) % 32;
                uint64_t a = ea + (uint64_t)i * step;
                if (L) c->v[r] = Q ? LD128(a) : (u128)LD64(a);
                else if (Q) ST128(a, c->v[r]);
                else ST64(a, VS64(r));
            }
        } else if (L) {
            sl_ldm(c, (int)Q, (int)sz, selem, rpt, (int)rt, ea);
        } else {
            sl_stm(c, (int)Q, (int)sz, selem, rpt, (int)rt, ea);
        }
    } else {  // single structure
        unsigned opc = bits(w, 13, 3), S = bits(w, 12, 1), R = bits(w, 21, 1);
        int scale = (int)(opc >> 1);
        int selem = (int)((((opc & 1) << 1) | R) + 1);
        if (scale == 3) {
            if (!L || S) return false;
            total = (uint64_t)selem << sz;
            sl_ldr(c, (int)Q, (int)sz, selem, (int)rt, ea);
        } else {
            int idx;
            if (scale == 0) {
                idx = (int)((Q << 3) | (S << 2) | sz);
            } else if (scale == 1) {
                if (sz & 1) return false;
                idx = (int)((Q << 2) | (S << 1) | (sz >> 1));
            } else {
                if (sz & 2) return false;
                if (sz & 1) {
                    if (S) return false;
                    idx = (int)Q;
                    scale = 3;
                } else {
                    idx = (int)((Q << 1) | S);
                }
            }
            total = (uint64_t)selem << scale;
            if (L) sl_lds(c, scale, selem, idx, (int)rt, ea);
            else sl_sts(c, scale, selem, idx, (int)rt, ea);
        }
    }
    if (post) setxs(c, rn, rm == 31 ? ea + total : ea + rx(c, rm));
    return true;
}

static bool sldst(Cpu *c, uint32_t w, uint64_t pc) {
    unsigned rt = bits(w, 0, 5);
    if (bits(w, 27, 3) == 3 && bits(w, 24, 2) == 0) {  // literal
        unsigned opc = bits(w, 30, 2);
        if (opc == 3) return false;
        uint64_t addr = pc + (uint64_t)(sext(bits(w, 5, 19), 19) * 4);
        c->v[rt] = opc == 0 ? (u128)LD32(addr) : opc == 1 ? (u128)LD64(addr) : (u128)LD128(addr);
        return true;
    }
    if (bits(w, 31, 1) == 0 && (bits(w, 24, 6) == 0x0C || bits(w, 24, 6) == 0x0D)) return sldst_struct(c, w);
    if (bits(w, 27, 3) == 5) return sldst_pair(c, w);
    if (bits(w, 27, 3) == 7) return sldst_reg(c, w);
    return false;
}

#include "interp_simd_vec.inc"

bool interp_simd(Cpu *c, uint32_t w, uint64_t pc) {
    if ((bits(w, 25, 4) & 5) == 4) return sldst(c, w, pc);
    return simd_dp(c, w);
}
