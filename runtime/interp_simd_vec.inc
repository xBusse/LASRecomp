// AdvSIMD / crypto data processing for the interpreter: the port of tools/simd_c.py's dp(), included by
// interp_simd.c (which has the helpers and the scalar FP / load-store part).

// ---- vector value helpers: V registers as u128 bit patterns, with the FP view picked by lane size
static inline u128 vq(Cpu *c, int Q, int n) { return Q ? c->v[n] : (u128)(uint64_t)c->v[n]; }
static inline void vw(Cpu *c, int Q, int d, u128 r) { c->v[d] = Q ? r : (u128)(uint64_t)r; }
static inline vf32 as_f32(u128 x) { vf32 v; memcpy(&v, &x, 16); return v; }
static inline vf64 as_f64(u128 x) { vf64 v; memcpy(&v, &x, 16); return v; }
static inline u128 of_f32(vf32 v) { u128 x; memcpy(&x, &v, 16); return x; }
static inline u128 of_f64(vf64 v) { u128 x; memcpy(&x, &v, 16); return x; }
static inline u128 of_u32(vu32 v) { u128 x; memcpy(&x, &v, 16); return x; }
static inline u128 of_u64(vu64 v) { u128 x; memcpy(&x, &v, 16); return x; }

// FP vector operations on the 4S (sz 0) or 2D (sz 1) view
static inline u128 fv_op(Cpu *c, int sz, int fop, u128 a, u128 b, u128 acc) {
    if (!sz) return of_f32(vop_s(c, fop, as_f32(a), as_f32(b), as_f32(acc)));
    return of_f64(vop_d(c, fop, as_f64(a), as_f64(b), as_f64(acc)));
}
static inline u128 fv_neg(int sz, u128 a) {
    if (!sz) return of_u32((vu32)as_f32(a) ^ 0x80000000u);
    return of_u64((vu64)as_f64(a) ^ 0x8000000000000000ull);
}
static inline u128 fv_cmp(Cpu *c, int sz, int cop, u128 a, u128 b) {
    if (!sz) return of_u32(vcmp_s(c, cop, as_f32(a), as_f32(b)));
    return of_u64(vcmp_d(c, cop, as_f64(a), as_f64(b)));
}
static inline u128 fv_frint(Cpu *c, int sz, u128 a, int rm) {
    if (!sz) return of_f32(vfrint_s(c, as_f32(a), rm));
    return of_f64(vfrint_d(c, as_f64(a), rm));
}
static inline u128 fv_toint(Cpu *c, int sz, u128 a, int rm, int fb, int U) {
    if (!sz) return of_u32(vtoint_s(c, as_f32(a), rm, fb, U));
    return of_u64(vtoint_d(c, as_f64(a), rm, fb, U));
}
static inline u128 fv_fromint(Cpu *c, int sz, u128 a, int fb, int U) {
    if (!sz) return of_f32(vfromint_s(c, (vu32)as_f32(a), fb, U));
    return of_f64(vfromint_d(c, (vu64)as_f64(a), fb, U));
}
static inline u128 fv_recpe(Cpu *c, int sz, u128 a, int rsq) {
    if (!sz) return of_f32(rsq ? vrsqrte_s(c, as_f32(a)) : vrecpe_s(c, as_f32(a)));
    return of_f64(rsq ? vrsqrte_d(c, as_f64(a)) : vrecpe_d(c, as_f64(a)));
}
static inline u128 fv_splat(int sz, u128 a, int idx, int Q) {
    if (!sz) return of_f32(vsplat_s(as_f32(a), idx, Q));
    return of_f64(vsplat_d(as_f64(a), idx, Q));
}
// scalar FP on the low lane: values are the register's lane bits
static inline uint64_t fs_get(Cpu *c, int sz, int r) { return sz ? (uint64_t)VS64(r) : (uint64_t)VS32(r); }
static inline uint64_t fs_op(Cpu *c, int sz, int fop, uint64_t x, uint64_t y, uint64_t z) {
    if (!sz) return fb_s(op_s(c, fop, fv_s((uint32_t)x), fv_s((uint32_t)y), fv_s((uint32_t)z)));
    return fb_d(op_d(c, fop, fv_d(x), fv_d(y), fv_d(z)));
}
static inline uint64_t fs_neg(int sz, uint64_t x) { return sz ? x ^ 0x8000000000000000ull : (uint32_t)(x ^ 0x80000000u); }

// element-wise permutation: result lane i = lane idx[i] of the concatenation a:b (16 / lane-size lanes each)
static void shuf_write(Cpu *c, int d, int Q, int sz, u128 a, u128 b, const uint8_t *idx) {
    int NL = 16 >> sz;
    u128 r = 0;
    for (int i = 0; i < NL; i++) {
        int k = idx[i];
        lset(&r, sz, i, k < NL ? lget(&a, sz, k) : lget(&b, sz, k - NL));
    }
    vw(c, Q, d, r);
}
static void pair_idx(int Q, int sz, uint8_t *ev, uint8_t *od) {
    int ne = LANES(Q, sz), NL = 16 >> sz;
    uint8_t src[32];
    for (int i = 0; i < ne; i++) src[i] = (uint8_t)i, src[ne + i] = (uint8_t)(NL + i);
    for (int k = 0; k < NL; k++) {
        ev[k] = k < ne ? src[2 * k] : (uint8_t)(ne % NL);
        od[k] = k < ne ? src[2 * k + 1] : (uint8_t)(ne % NL);
    }
}

// ---------------------------------------------------------------- crypto

static bool crypto_aes(Cpu *c, uint32_t w) {
    if (bits(w, 22, 2)) return false;
    unsigned opc = bits(w, 12, 5);
    if (opc < 4 || opc > 7) return false;
    sv_aes(c, (int)opc - 4, (int)bits(w, 0, 5), (int)bits(w, 5, 5));
    return true;
}
static bool crypto_sha3(Cpu *c, uint32_t w) {
    if (bits(w, 22, 2)) return false;
    unsigned opc = bits(w, 12, 3);
    int d = (int)bits(w, 0, 5), n = (int)bits(w, 5, 5), m = (int)bits(w, 16, 5);
    if (opc < 3) sv_sha1hash(c, (int)opc, d, n, m);
    else if (opc == 3) sv_sha_misc(c, 0, d, n, m);
    else if (opc == 4 || opc == 5) sv_sha256hash(c, opc == 4, d, n, m);
    else if (opc == 6) sv_sha_misc(c, 3, d, n, m);
    else return false;
    return true;
}
static bool crypto_sha2(Cpu *c, uint32_t w) {
    if (bits(w, 22, 2)) return false;
    unsigned opc = bits(w, 12, 5);
    int op;
    if (opc == 0) op = 4;
    else if (opc == 1) op = 1;
    else if (opc == 2) op = 2;
    else return false;
    sv_sha_misc(c, op, (int)bits(w, 0, 5), (int)bits(w, 5, 5), 0);
    return true;
}

// ---------------------------------------------------------------- copy / permute / extract / table

static bool imm5_split(unsigned imm5, int *sz, int *idx) {
    if (!(imm5 & 15)) return false;
    int s = __builtin_ctz(imm5);
    *sz = s;
    *idx = (int)(imm5 >> (s + 1));
    return true;
}

static bool scalar_copy(Cpu *c, uint32_t w) {
    if (bits(w, 29, 1) || bits(w, 11, 4)) return false;
    int sz, idx;
    if (!imm5_split(bits(w, 16, 5), &sz, &idx) || sz > 3) return false;
    unsigned d = bits(w, 0, 5), n = bits(w, 5, 5);
    c->v[d] = lget(&c->v[n], sz, idx);
    return true;
}

static bool vcopy(Cpu *c, uint32_t w) {
    unsigned Q = bits(w, 30, 1), op = bits(w, 29, 1), imm5 = bits(w, 16, 5), imm4 = bits(w, 11, 4);
    unsigned d = bits(w, 0, 5), n = bits(w, 5, 5);
    int sz, idx;
    if (!imm5_split(imm5, &sz, &idx) || sz > 3) return false;
    if (op) {  // INS (element)
        if (!Q) return false;
        int idx2 = (int)(imm4 >> sz);
        uint64_t e = lget(&c->v[n], sz, idx2);
        lset(&c->v[d], sz, idx, e);
        return true;
    }
    if (imm4 == 0 || imm4 == 1) {  // DUP (element) / DUP (general)
        if (sz == 3 && !Q) return false;
        uint64_t e = imm4 == 0 ? lget(&c->v[n], sz, idx) : (rx(c, n) & lmask(sz));
        u128 r = 0;
        for (int i = 0; i < (16 >> sz); i++) lset(&r, sz, i, e);
        vw(c, (int)Q, (int)d, r);
        return true;
    }
    if (imm4 == 3) {  // INS (general)
        if (!Q) return false;
        lset(&c->v[d], sz, idx, rx(c, n));
        return true;
    }
    if (imm4 == 5) {  // SMOV
        if (sz >= 2 + (int)Q) return false;
        uint64_t v = (uint64_t)lsx(lget(&c->v[n], sz, idx), sz);
        if (Q) setx(c, d, v); else setw(c, d, (uint32_t)v);
        return true;
    }
    if (imm4 == 7) {  // UMOV
        if ((Q && sz != 3) || (!Q && sz == 3)) return false;
        uint64_t v = lget(&c->v[n], sz, idx);
        if (Q) setx(c, d, v); else setw(c, d, (uint32_t)v);
        return true;
    }
    return false;
}

static bool permute(Cpu *c, uint32_t w) {
    unsigned Q = bits(w, 30, 1), sz = bits(w, 22, 2), m = bits(w, 16, 5), opc = bits(w, 12, 3), n = bits(w, 5, 5),
             d = bits(w, 0, 5);
    if (opc == 0 || opc == 4 || (sz == 3 && !Q)) return false;
    int ne = LANES(Q, sz), NL = 16 >> sz;
    int part = (int)(opc >> 2), kind = (int)(opc & 3);
    uint8_t idx[16] = {0};
    for (int i = 0; i < ne; i++) {
        if (kind == 1) {  // UZP
            int j = 2 * i + part;
            idx[i] = (uint8_t)(j < ne ? j : NL + j - ne);
        } else if (kind == 2) {  // TRN
            int p = i & ~1;
            idx[i] = (uint8_t)(!(i & 1) ? p + part : NL + p + part);
        } else {  // ZIP
            int base = part * ne / 2;
            idx[i] = (uint8_t)(!(i & 1) ? base + i / 2 : NL + base + i / 2);
        }
    }
    shuf_write(c, (int)d, (int)Q, (int)sz, c->v[n], c->v[m], idx);
    return true;
}

static bool vext(Cpu *c, uint32_t w) {
    unsigned Q = bits(w, 30, 1), m = bits(w, 16, 5), imm4 = bits(w, 11, 4), n = bits(w, 5, 5), d = bits(w, 0, 5);
    if (bits(w, 22, 2) || (!Q && imm4 >= 8)) return false;
    uint8_t idx[16];
    if (Q) {
        for (int i = 0; i < 16; i++) idx[i] = (uint8_t)(imm4 + i);
    } else {
        for (int i = 0; i < 8; i++) idx[i] = (uint8_t)((imm4 + i) < 8 ? imm4 + i : 16 + imm4 + i - 8);
        for (int i = 8; i < 16; i++) idx[i] = 0;
    }
    shuf_write(c, (int)d, (int)Q, 0, c->v[n], c->v[m], idx);
    return true;
}

static bool vtbl(Cpu *c, uint32_t w) {
    if (bits(w, 22, 2)) return false;
    sv_tbl(c, (int)bits(w, 30, 1), (int)bits(w, 13, 2), (int)bits(w, 12, 1), (int)bits(w, 0, 5), (int)bits(w, 5, 5),
           (int)bits(w, 16, 5));
    return true;
}

// ---------------------------------------------------------------- three same

static bool three_same_fp(Cpu *c, uint32_t w, bool scalar) {
    unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), a_ = bits(w, 23, 1), sz = bits(w, 22, 1), m = bits(w, 16, 5),
             opc = bits(w, 11, 5), n = bits(w, 5, 5), d = bits(w, 0, 5);
    if (!scalar && sz && !Q) return false;
    // comparisons
    int cop = -1;
    if (!U && !a_ && opc == 0x1C) cop = 0;
    else if (U && !a_ && opc == 0x1C) cop = 1;
    else if (U && a_ && opc == 0x1C) cop = 2;
    else if (U && !a_ && opc == 0x1D) cop = 3;
    else if (U && a_ && opc == 0x1D) cop = 4;
    if (cop >= 0) {
        if (scalar) {
            u128 a = sz ? (u128)VS64(n) : (u128)VS32(n), b = sz ? (u128)VS64(m) : (u128)VS32(m);
            u128 r = fv_cmp(c, (int)sz, cop, a, b);  // (the other lanes are zero: only lane 0 is used)
            c->v[d] = sz ? (uint64_t)r : (uint32_t)r;
        } else {
            vw(c, (int)Q, (int)d, fv_cmp(c, (int)sz, cop, vq(c, (int)Q, (int)n), vq(c, (int)Q, (int)m)));
        }
        return true;
    }
    // table: (U, a, opc) -> FOP
    int fop = -1;
    unsigned key = (U << 6) | (a_ << 5) | opc;
    switch (key) {
    case (0 << 6) | (0 << 5) | 0x18: fop = FOP_MAXNM; break;
    case (0 << 6) | (0 << 5) | 0x1A: fop = FOP_ADD; break;
    case (0 << 6) | (0 << 5) | 0x1B: fop = FOP_MULX; break;
    case (0 << 6) | (0 << 5) | 0x1E: fop = FOP_MAX; break;
    case (0 << 6) | (0 << 5) | 0x1F: fop = FOP_RECPS; break;
    case (0 << 6) | (1 << 5) | 0x18: fop = FOP_MINNM; break;
    case (0 << 6) | (1 << 5) | 0x1A: fop = FOP_SUB; break;
    case (0 << 6) | (1 << 5) | 0x1E: fop = FOP_MIN; break;
    case (0 << 6) | (1 << 5) | 0x1F: fop = FOP_RSQRTS; break;
    case (1 << 6) | (0 << 5) | 0x1B: fop = FOP_MUL; break;
    case (1 << 6) | (0 << 5) | 0x1F: fop = FOP_DIV; break;
    case (1 << 6) | (1 << 5) | 0x1A: fop = FOP_ABD; break;
    }
    if (scalar) {
        // (0,0,11011) FMULX, (0,0,11111) FRECPS, (0,1,11111) FRSQRTS, (1,1,11010) FABD
        if (!(key == 0x1B || key == 0x1F || key == 0x3F || key == 0x7A)) return false;
        uint64_t x = fs_get(c, (int)sz, (int)n), y = fs_get(c, (int)sz, (int)m);
        if (fop == FOP_RECPS || fop == FOP_RSQRTS) x = fs_neg((int)sz, x);
        c->v[d] = fs_op(c, (int)sz, fop, x, y, 0);
        return true;
    }
    u128 A = vq(c, (int)Q, (int)n), B = vq(c, (int)Q, (int)m);
    int pfop = -1;  // pairwise: (1,0,11000) FMAXNMP, (1,1,11000) FMINNMP, (1,0,11010) FADDP, (1,0,11110) FMAXP, (1,1,11110) FMINP
    switch (key) {
    case (1 << 6) | (0 << 5) | 0x18: pfop = FOP_MAXNM; break;
    case (1 << 6) | (1 << 5) | 0x18: pfop = FOP_MINNM; break;
    case (1 << 6) | (0 << 5) | 0x1A: pfop = FOP_ADD; break;
    case (1 << 6) | (0 << 5) | 0x1E: pfop = FOP_MAX; break;
    case (1 << 6) | (1 << 5) | 0x1E: pfop = FOP_MIN; break;
    }
    if (pfop >= 0) {
        uint8_t ev[16], od[16];
        int lsz = 2 + (int)sz, NL = 16 >> lsz;
        pair_idx((int)Q, lsz, ev, od);
        u128 L = 0, Rr = 0;
        for (int i = 0; i < NL; i++) {
            lset(&L, lsz, i, ev[i] < NL ? lget(&A, lsz, ev[i]) : lget(&B, lsz, ev[i] - NL));
            lset(&Rr, lsz, i, od[i] < NL ? lget(&A, lsz, od[i]) : lget(&B, lsz, od[i] - NL));
        }
        vw(c, (int)Q, (int)d, fv_op(c, (int)sz, pfop, L, Rr, 0));
        return true;
    }
    if (!U && opc == 0x19) {  // FMLA / FMLS: the multiplicand is negated for FMLS
        vw(c, (int)Q, (int)d, fv_op(c, (int)sz, FOP_MADD, vq(c, (int)Q, (int)d), a_ ? fv_neg((int)sz, A) : A, B));
        return true;
    }
    if (fop < 0) return false;
    if (fop == FOP_RECPS || fop == FOP_RSQRTS) A = fv_neg((int)sz, A);
    vw(c, (int)Q, (int)d, fv_op(c, (int)sz, fop, A, B, 0));
    return true;
}

static bool three_same_scalar(Cpu *c, unsigned U, int sz, unsigned opc, int d, int n, int m) {
    if (opc == 1 || opc == 5) {
        int op = opc == 1 ? (U ? LOP_UQADD : LOP_SQADD) : (U ? LOP_UQSUB : LOP_SQSUB);
        same(c, op, sz, 1, d, n, m, 0, -1);
        return true;
    }
    if (opc == 9 || opc == 11) {
        same(c, ((opc == 9 ? LOP_QSHL : LOP_QRSHL) | (U ? LOP_U : 0)), sz, 1, d, n, m, 0, -1);
        return true;
    }
    if (opc == 22) {
        if (sz != 1 && sz != 2) return false;
        same(c, U ? LOP_SQRDMULH : LOP_SQDMULH, sz, 1, d, n, m, 0, -1);
        return true;
    }
    if (sz != 3) return false;
    uint64_t a = VS64(n), b = VS64(m);
    int64_t sa = (int64_t)a, sb = (int64_t)b;
    switch (opc) {
    case 6: c->v[d] = (U ? a > b : sa > sb) ? ~0ull : 0; return true;
    case 7: c->v[d] = (U ? a >= b : sa >= sb) ? ~0ull : 0; return true;
    case 8: case 10: same(c, ((opc == 8 ? LOP_SHL : LOP_RSHL) | (U ? LOP_U : 0)), 3, 1, d, n, m, 0, -1); return true;
    case 16: c->v[d] = U ? a - b : a + b; return true;
    case 17: c->v[d] = (U ? a == b : (a & b) != 0) ? ~0ull : 0; return true;
    }
    return false;
}

static bool three_same(Cpu *c, uint32_t w, bool scalar) {
    unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), m = bits(w, 16, 5), opc = bits(w, 11, 5),
             n = bits(w, 5, 5), d = bits(w, 0, 5);
    if (opc >= 0x18) return three_same_fp(c, w, scalar);
    if (scalar) return three_same_scalar(c, U, (int)sz, opc, (int)d, (int)n, (int)m);
    if (sz == 3 && !Q && opc != 3) return false;
    int ne = LANES(Q, sz);
    if ((opc == 0 || opc == 2 || opc == 4 || opc == 14 || opc == 15 || opc == 12 || opc == 13) && sz == 3) return false;
    switch (opc) {
    case 0: same(c, LOPU(HADD, U), (int)sz, ne, (int)d, (int)n, (int)m, 0, -1); return true;
    case 2: same(c, LOPU(RHADD, U), (int)sz, ne, (int)d, (int)n, (int)m, 0, -1); return true;
    case 4: same(c, LOPU(HSUB, U), (int)sz, ne, (int)d, (int)n, (int)m, 0, -1); return true;
    case 3: {
        vu64 a = VREG(vu64, n), b = VREG(vu64, m), o = VREG(vu64, d), r;
        if (!U) {
            switch (sz) {
            case 0: r = a & b; break;
            case 1: r = a & ~b; break;
            case 2: r = a | b; break;
            default: r = a | ~b; break;
            }
        } else {
            switch (sz) {
            case 0: r = a ^ b; break;
            case 1: r = (o & a) | (~o & b); break;
            case 2: r = (o & ~b) | (a & b); break;
            default: r = (o & b) | (a & ~b); break;
            }
        }
        VW(vu64, Q, d, r);
        return true;
    }
    case 8: case 9: case 10: case 11: {
        static const int ops[4] = {LOP_SHL, LOP_QSHL, LOP_RSHL, LOP_QRSHL};
        same(c, ops[opc - 8] | (U ? LOP_U : 0), (int)sz, ne, (int)d, (int)n, (int)m, 0, -1);
        return true;
    }
    }
    if (sz == 3 && opc != 23 && opc >= 18) return false;  // (the python checks this after opc 17; see below)
    switch (opc) {
    case 1: case 5: case 6: case 7: case 12: case 13: case 14: case 15: case 16: case 17: {
        BYSZ(sz, {
            T a = VREG(T, n), b = VREG(T, m), o = VREG(T, d);
            S sa = VREG(S, n), sb = VREG(S, m);
            switch (opc) {
            case 1: VW(T, Q, d, U ? (T)__builtin_elementwise_add_sat(a, b) : (T)__builtin_elementwise_add_sat(sa, sb)); break;
            case 5: VW(T, Q, d, U ? (T)__builtin_elementwise_sub_sat(a, b) : (T)__builtin_elementwise_sub_sat(sa, sb)); break;
            case 6: VW(T, Q, d, U ? (T)(a > b) : (T)(sa > sb)); break;
            case 7: VW(T, Q, d, U ? (T)(a >= b) : (T)(sa >= sb)); break;
            case 12: VW(T, Q, d, U ? (T)__builtin_elementwise_max(a, b) : (T)__builtin_elementwise_max(sa, sb)); break;
            case 13: VW(T, Q, d, U ? (T)__builtin_elementwise_min(a, b) : (T)__builtin_elementwise_min(sa, sb)); break;
            case 14: case 15: {
                T e = U ? (T)__builtin_elementwise_max(a, b) - (T)__builtin_elementwise_min(a, b)
                        : (T)__builtin_elementwise_max(sa, sb) - (T)__builtin_elementwise_min(sa, sb);
                VW(T, Q, d, opc == 15 ? o + e : e);
                break;
            }
            case 16: VW(T, Q, d, U ? a - b : a + b); break;
            default: VW(T, Q, d, U ? (T)(a == b) : (T)((a & b) != 0)); break;
            }
        })
        return true;
    }
    case 18: case 19: case 20: case 21: case 22: case 23:
        break;
    default:
        return false;
    }
    if (sz == 3 && opc != 23) return false;
    switch (opc) {
    case 18:
        BYSZ(sz, { T a = VREG(T, n), b = VREG(T, m), o = VREG(T, d); VW(T, Q, d, U ? o - a * b : o + a * b); })
        return true;
    case 19:
        if (U) {
            if (sz) return false;
            same(c, LOP_PMUL, 0, ne, (int)d, (int)n, (int)m, 0, -1);
            return true;
        }
        BYSZ(sz, { T a = VREG(T, n), b = VREG(T, m); VW(T, Q, d, a * b); })
        return true;
    case 20: case 21: case 23: {
        if (opc == 23 && U) return false;
        uint8_t ev[16], od[16];
        pair_idx((int)Q, (int)sz, ev, od);
        int NL = 16 >> sz;
        u128 A = c->v[n], B = c->v[m];
        BYSZ(sz, {
            T a = *(T *)&A, b = *(T *)&B;
            S sa = *(S *)&A, sb = *(S *)&B;
            T L, Rr;
            S sL, sR;
            for (int i = 0; i < NL; i++) {
                int e = ev[i], o = od[i];
                L[i] = e < NL ? a[e] : b[e - NL];
                Rr[i] = o < NL ? a[o] : b[o - NL];
                sL[i] = e < NL ? sa[e] : sb[e - NL];
                sR[i] = o < NL ? sa[o] : sb[o - NL];
            }
            if (opc == 23) {
                VW(T, Q, d, (T)(L + Rr));
            } else if (U) {
                VW(T, Q, d, opc == 20 ? (T)__builtin_elementwise_max(L, Rr) : (T)__builtin_elementwise_min(L, Rr));
            } else {
                VW(T, Q, d, opc == 20 ? (T)__builtin_elementwise_max(sL, sR) : (T)__builtin_elementwise_min(sL, sR));
            }
        })
        return true;
    }
    case 22:
        if (sz != 1 && sz != 2) return false;
        same(c, U ? LOP_SQRDMULH : LOP_SQDMULH, (int)sz, ne, (int)d, (int)n, (int)m, 0, -1);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------- three different

static bool three_diff(Cpu *c, uint32_t w, bool scalar) {
    unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), m = bits(w, 16, 5), opc = bits(w, 12, 4),
             n = bits(w, 5, 5), d = bits(w, 0, 5);
    if (scalar) {
        if (U || (opc != 9 && opc != 11 && opc != 13) || (sz != 1 && sz != 2)) return false;
        int op = opc == 9 ? LOP_SQDMLAL : opc == 11 ? LOP_SQDMLSL : LOP_SQDMULL;
        gen(c, op, (int)sz, 1, (int)d, (int)sz + 1, 0, 0, (int)n, (int)sz, 0, (int)m, (int)sz, 0, -1, 0);
        return true;
    }
    if (opc == 14) {  // PMULL
        if (U) return false;
        if (sz == 3) { sl_pmull64(c, (int)Q, (int)d, (int)n, (int)m); return true; }
        if (sz) return false;
    }
    if (sz == 3 || opc == 15) return false;
    int ne = 8 >> sz, off = (int)Q * ne;
    if (opc == 4 || opc == 6) {  // ADDHN / RADDHN / SUBHN / RSUBHN: narrow into the low or high half
        static const int ops[4] = {LOP_ADDHN, LOP_RADDHN, LOP_SUBHN, LOP_RSUBHN};
        gen(c, ops[(opc == 6) * 2 + U], (int)sz, ne, (int)d, (int)sz, off, (int)Q, (int)n, (int)sz + 1, 0, (int)m,
            (int)sz + 1, 0, -1, 0);
        return true;
    }
    if (opc == 1 || opc == 3) {  // wide
        gen(c, (opc == 1 ? LOP_ADDW : LOP_SUBW) | (U ? LOP_U : 0), (int)sz, ne, (int)d, (int)sz + 1, 0, 0, (int)n,
            (int)sz + 1, 0, (int)m, (int)sz, off, -1, 0);
        return true;
    }
    int op, sq = 0, pm = 0;
    switch (opc) {
    case 0: op = LOP_ADDL; break;
    case 2: op = LOP_SUBL; break;
    case 5: op = LOP_ABAL; break;
    case 7: op = LOP_ABDL; break;
    case 8: op = LOP_MLAL; break;
    case 10: op = LOP_MLSL; break;
    case 12: op = LOP_MULL; break;
    case 9: op = LOP_SQDMLAL; sq = 1; break;
    case 11: op = LOP_SQDMLSL; sq = 1; break;
    case 13: op = LOP_SQDMULL; sq = 1; break;
    case 14: op = LOP_PMULL; pm = 1; break;
    default: return false;
    }
    if (sq && (U || (sz != 1 && sz != 2))) return false;
    gen(c, op | ((!sq && !pm && U) ? LOP_U : 0), (int)sz, ne, (int)d, (int)sz + 1, 0, 0, (int)n, (int)sz, off, (int)m,
        (int)sz, off, -1, 0);
    return true;
}

// ---------------------------------------------------------------- two-register misc

static bool misc2_scalar(Cpu *c, unsigned U, int sz, unsigned opc, int d, int n) {
    if (opc == 3) { same(c, U ? LOP_USQADD : LOP_SUQADD, sz, 1, d, n, -1, 0, -1); return true; }
    if (opc == 7) { same(c, U ? LOP_SQNEG : LOP_SQABS, sz, 1, d, n, -1, 0, -1); return true; }
    if (opc == 0x12 || opc == 0x14) {
        if (sz == 3 || (opc == 0x12 && !U)) return false;
        int op;
        if (opc == 0x12) op = LOP_SQXTUN;
        else op = U ? LOP_UQXTN : LOP_SQXTN;
        gen(c, op, sz, 1, d, sz, 0, 0, n, sz + 1, 0, -1, 0, 0, -1, 0);
        return true;
    }
    if (sz != 3) return false;
    uint64_t a = VS64(n);
    int64_t sa = (int64_t)a;
    switch (opc) {
    case 8: c->v[d] = (U ? sa >= 0 : sa > 0) ? ~0ull : 0; return true;
    case 9: c->v[d] = (U ? sa <= 0 : sa == 0) ? ~0ull : 0; return true;
    case 10: if (U) return false; c->v[d] = sa < 0 ? ~0ull : 0; return true;
    case 11: c->v[d] = U ? (uint64_t)-a : (sa < 0 ? (uint64_t)-a : a); return true;
    }
    return false;
}

static bool misc2_fp(Cpu *c, uint32_t w, bool scalar) {
    unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), a_ = bits(w, 23, 1), sz = bits(w, 22, 1), opc = bits(w, 12, 5),
             n = bits(w, 5, 5), d = bits(w, 0, 5);
    if (opc == 0x16 || opc == 0x17) {  // FCVTN / FCVTL / FCVTXN
        if (a_) return false;
        if (opc == 0x16 && U) {
            if (!sz) return false;
            if (scalar) { FP_SETS(d, simd_d2s_odd(c, VSF64(n))); return true; }
            sv_fcvtn(c, 1, (int)Q, 1, (int)d, (int)n);
            return true;
        }
        if (scalar || U) return false;
        if (opc == 0x16) sv_fcvtn(c, (int)sz, (int)Q, 0, (int)d, (int)n);
        else sv_fcvtl(c, (int)sz, (int)Q, (int)d, (int)n);
        return true;
    }
    if (!scalar && sz && !Q) return false;
    u128 x = vq(c, (int)Q, (int)n);
    if (opc >= 0x0C && opc <= 0x0F) {
        if (!a_) return false;
        if (opc == 0x0F) {  // FABS / FNEG
            if (scalar) return false;
            u128 mask = sz ? ((u128)0x8000000000000000ull << 64 | 0x8000000000000000ull) : 0;
            if (!sz) {
                vu32 v = (vu32)as_f32(c->v[n]);
                v = U ? v ^ 0x80000000u : v & ~0x80000000u;
                vw(c, (int)Q, (int)d, of_u32(v));
            } else {
                vu64 v = (vu64)as_f64(c->v[n]);
                v = U ? v ^ 0x8000000000000000ull : v & ~0x8000000000000000ull;
                vw(c, (int)Q, (int)d, of_u64(v));
            }
            (void)mask;
            return true;
        }
        int op, swap;
        if (opc == 0x0C && !U) op = 2, swap = 0;
        else if (opc == 0x0C && U) op = 1, swap = 0;
        else if (opc == 0x0D && !U) op = 0, swap = 0;
        else if (opc == 0x0D && U) op = 1, swap = 1;
        else if (opc == 0x0E && !U) op = 2, swap = 1;
        else return false;
        if (scalar) {
            u128 a = sz ? (u128)VS64(n) : (u128)VS32(n), b = 0;
            if (swap) { u128 t = a; a = b; b = t; }
            u128 r = fv_cmp(c, (int)sz, op, a, b);
            c->v[d] = sz ? (uint64_t)r : (uint32_t)r;
            return true;
        }
        u128 a = x, b = 0;
        if (swap) { u128 t = a; a = b; b = t; }
        vw(c, (int)Q, (int)d, fv_cmp(c, (int)sz, op, a, b));
        return true;
    }
    if (opc == 0x1F) {
        if (!a_) return false;
        if (U) {  // FSQRT
            if (scalar) return false;
            vw(c, (int)Q, (int)d, fv_op(c, (int)sz, FOP_SQRT, x, x, 0));
            return true;
        }
        if (!scalar) return false;  // FRECPX
        if (!sz) FP_SETS(d, recpx_s(c, VSF32(n)));
        else FP_SETD(d, recpx_d(c, VSF64(n)));
        return true;
    }
    if (opc == 0x18 || opc == 0x19) {  // FRINT*
        if (scalar) return false;
        int rm;
        unsigned k = (U << 2) | (a_ << 1) | (opc & 1);
        switch (k) {
        case 0: rm = 0; break;  // (0,0,0)
        case 1: rm = 2; break;  // (0,0,1)
        case 2: rm = 1; break;  // (0,1,0)
        case 3: rm = 3; break;  // (0,1,1)
        case 4: rm = 4; break;  // (1,0,0)
        case 5: rm = 5; break;  // (1,0,1)
        case 7: rm = 6; break;  // (1,1,1)
        default: return false;
        }
        vw(c, (int)Q, (int)d, fv_frint(c, (int)sz, x, rm));
        return true;
    }
    if (opc == 0x1A || opc == 0x1B || (opc == 0x1C && !a_)) {  // FCVT[NPMZA][SU]
        int rm = opc == 0x1A ? (a_ ? 1 : 0) : opc == 0x1B ? (a_ ? 3 : 2) : 4;
        if (scalar) {
            uint64_t r = !sz ? toint_s(c, VSF32(n), rm, 0, 32, (int)U) : toint_d(c, VSF64(n), rm, 0, 64, (int)U);
            c->v[d] = sz ? r : (uint32_t)r;
            return true;
        }
        vw(c, (int)Q, (int)d, fv_toint(c, (int)sz, x, rm, 0, (int)U));
        return true;
    }
    if (opc == 0x1C) {  // URECPE / URSQRTE
        if (scalar || sz) return false;
        same(c, U ? LOP_URSQRTE : LOP_URECPE, 2, LANES(Q, 2), (int)d, (int)n, -1, 0, -1);
        return true;
    }
    if (opc == 0x1D) {
        if (!a_) {  // SCVTF / UCVTF
            if (scalar) {
                if (!sz) FP_SETS(d, fromint_s(c, VS32(n), 32, (int)U, 0));
                else FP_SETD(d, fromint_d(c, VS64(n), 64, (int)U, 0));
                return true;
            }
            vw(c, (int)Q, (int)d, fv_fromint(c, (int)sz, c->v[n], 0, (int)U));
            return true;
        }
        if (scalar) {  // FRECPE / FRSQRTE
            if (!sz) FP_SETS(d, U ? rsqrte_s(c, VSF32(n)) : recpe_s(c, VSF32(n)));
            else FP_SETD(d, U ? rsqrte_d(c, VSF64(n)) : recpe_d(c, VSF64(n)));
            return true;
        }
        vw(c, (int)Q, (int)d, fv_recpe(c, (int)sz, x, (int)U));
        return true;
    }
    return false;
}

static bool misc2(Cpu *c, uint32_t w, bool scalar) {
    unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), opc = bits(w, 12, 5), n = bits(w, 5, 5),
             d = bits(w, 0, 5);
    if (opc >= 0x18 || opc == 0x16 || opc == 0x17 || (opc >= 0x0C && opc <= 0x0F)) return misc2_fp(c, w, scalar);
    if (scalar) return misc2_scalar(c, U, (int)sz, opc, (int)d, (int)n);
    int ne = LANES(Q, sz);
    if (opc == 0 || (opc == 1 && !U)) {  // REV64 / REV32 / REV16
        int gsz;
        if (U == 0 && opc == 0) gsz = 3;
        else if (U == 1 && opc == 0) gsz = 2;
        else gsz = 1;  // (0, 1)
        if ((int)sz >= gsz) return false;
        int es = 1 << sz, gb = 1 << gsz;
        uint8_t idx[16];
        for (int i = 0; i < 16; i++) {
            int g = i / gb * gb, o = i % gb, e = o / es, b = o % es;
            idx[i] = (uint8_t)(g + (gb / es - 1 - e) * es + b);
        }
        shuf_write(c, (int)d, (int)Q, 0, c->v[n], c->v[n], idx);
        return true;
    }
    if (sz == 3 && !Q && opc != 0x12 && opc != 0x14 && opc != 0x13) return false;
    if (opc == 2 || opc == 6) {
        if (sz == 3) return false;
        sl_addlp(c, (int)U, opc == 6, (int)Q, (int)sz, (int)d, (int)n);
        return true;
    }
    if (opc == 3) { same(c, U ? LOP_USQADD : LOP_SUQADD, (int)sz, ne, (int)d, (int)n, -1, 0, -1); return true; }
    if (opc == 4) {
        if (sz == 3) return false;
        same(c, U ? LOP_CLZ : LOP_CLS, (int)sz, ne, (int)d, (int)n, -1, 0, -1);
        return true;
    }
    if (opc == 5) {
        if (!U) {
            if (sz) return false;
            same(c, LOP_CNT, 0, ne, (int)d, (int)n, -1, 0, -1);
            return true;
        }
        if (sz == 0) { VW(vu64, Q, d, ~VREG(vu64, n)); return true; }
        if (sz == 1) { same(c, LOP_RBIT, 0, LANES(Q, 0), (int)d, (int)n, -1, 0, -1); return true; }
        return false;
    }
    if (opc == 7) { same(c, U ? LOP_SQNEG : LOP_SQABS, (int)sz, ne, (int)d, (int)n, -1, 0, -1); return true; }
    if (opc == 8 || opc == 9 || (opc == 10 && !U) || opc == 11) {
        BYSZ(sz, {
            T a = VREG(T, n);
            S sa = VREG(S, n);
            int es = 8 << sz;
            if (opc == 8) VW(T, Q, d, U ? (T)(sa >= 0) : (T)(sa > 0));
            else if (opc == 9) VW(T, Q, d, U ? (T)(sa <= 0) : (T)(sa == 0));
            else if (opc == 10) VW(T, Q, d, (T)(sa < 0));
            else if (U) VW(T, Q, d, (T)-a);
            else VW(T, Q, d, ((T)sa ^ (T)(sa >> (es - 1))) - (T)(sa >> (es - 1)));
        })
        return true;
    }
    if (opc == 0x12 || opc == 0x14) {  // XTN / SQXTUN / SQXTN / UQXTN
        if (sz == 3) return false;
        int op;
        if (opc == 0x12) op = U ? LOP_SQXTUN : LOP_XTN;
        else op = U ? LOP_UQXTN : LOP_SQXTN;
        int nl = 8 >> sz;
        gen(c, op, (int)sz, nl, (int)d, (int)sz, (int)Q * nl, (int)Q, (int)n, (int)sz + 1, 0, -1, 0, 0, -1, 0);
        return true;
    }
    if (opc == 0x13 && U) {  // SHLL
        if (sz == 3) return false;
        int nl = 8 >> sz;
        gen(c, LOP_SHLL, (int)sz, nl, (int)d, (int)sz + 1, 0, 0, (int)n, (int)sz, (int)Q * nl, -1, 0, 0, -1, 0);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------- across lanes / pairwise

static bool across(Cpu *c, uint32_t w) {
    unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), opc = bits(w, 12, 5), n = bits(w, 5, 5),
             d = bits(w, 0, 5);
    if (opc == 0x0C || opc == 0x0F) {
        if (!U) return false;
        if (!Q || (sz & 1)) return false;
        int fop;
        if (opc == 0x0C) fop = (sz >> 1) ? FOP_MINNM : FOP_MAXNM;
        else fop = (sz >> 1) ? FOP_MIN : FOP_MAX;
        sv_fred_s(c, fop, (int)d, (int)n);
        return true;
    }
    int op;
    switch (opc) {
    case 0x03: op = 1; break;
    case 0x0A: op = 2; break;
    case 0x1A: op = 3; break;
    case 0x1B: op = 0; break;
    default: return false;
    }
    if ((op == 0 && U) || sz == 3 || (sz == 2 && !Q)) return false;
    sl_across(c, op, (int)U, (int)Q, (int)sz, (int)d, (int)n);
    return true;
}

static bool scalar_pairwise(Cpu *c, uint32_t w) {
    unsigned U = bits(w, 29, 1), sz = bits(w, 22, 2), opc = bits(w, 12, 5), n = bits(w, 5, 5), d = bits(w, 0, 5);
    if (!U) {
        if (opc == 0x1B && sz == 3) { c->v[d] = (uint64_t)(VS64(n) + VHI64(n)); return true; }
        return false;
    }
    int fop = -1;
    unsigned hi = sz >> 1;
    if (opc == 0x0C) fop = hi ? FOP_MINNM : FOP_MAXNM;
    else if (opc == 0x0D && !hi) fop = FOP_ADD;
    else if (opc == 0x0F) fop = hi ? FOP_MIN : FOP_MAX;
    if (fop < 0) return false;
    if (!(sz & 1)) {
        vf32 v = VREG(vf32, n);
        FP_SETS(d, pair_s(c, fop, v[0], v[1]));
    } else {
        vf64 v = VREG(vf64, n);
        FP_SETD(d, pair_d(c, fop, v[0], v[1]));
    }
    return true;
}

// ---------------------------------------------------------------- modified immediate

static bool modimm(Cpu *c, uint32_t w) {
    unsigned Q = bits(w, 30, 1), op = bits(w, 29, 1), cmode = bits(w, 12, 4), o2 = bits(w, 11, 1), d = bits(w, 0, 5);
    unsigned imm8 = (bits(w, 16, 3) << 5) | bits(w, 5, 5);
    if (o2) return false;
    if (cmode == 15 && op && !Q) return false;
    uint64_t imm = simd_expand_imm(op, cmode, imm8);
    if ((cmode < 8 && (cmode & 1)) || (cmode >= 8 && cmode < 12 && (cmode & 1))) {  // ORR / BIC
        vu64 v = VREG(vu64, d);
        if (op) v = v & (vu64){~imm, ~imm};
        else v = v | (vu64){imm, imm};
        VW(vu64, Q, d, v);
        return true;
    }
    if (op && !(cmode == 14 || cmode == 15)) imm = ~imm;  // MVNI
    c->v[d] = Q ? ((u128)imm << 64) | imm : (u128)imm;
    return true;
}

// ---------------------------------------------------------------- shift by immediate

static bool shift_imm(Cpu *c, uint32_t w, bool scalar) {
    unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), immh = bits(w, 19, 4), immb = bits(w, 16, 3), opc = bits(w, 11, 5),
             n = bits(w, 5, 5), d = bits(w, 0, 5);
    if (!immh) return false;
    int sz = 31 - __builtin_clz(immh);
    int es = 8 << sz, ih = (int)((immh << 3) | immb);
    int rsh = 2 * es - ih, lsh = ih - es;
    int ne;
    if (scalar) Q = 1;
    ne = scalar ? 1 : LANES(Q, sz);
    int narrow = opc >= 0x10 && opc <= 0x13;
    if (opc == 0x1C || opc == 0x1F) {  // fixed-point conversions
        if (sz < 2) return false;
        if (sz == 3 && !Q) return false;
        int fs = sz - 2, fb = rsh;
        if (opc == 0x1C) {
            if (scalar) {
                if (!fs) FP_SETS(d, fromint_s(c, VS32(n), es, (int)U, fb));
                else FP_SETD(d, fromint_d(c, VS64(n), es, (int)U, fb));
                return true;
            }
            vw(c, (int)Q, (int)d, fv_fromint(c, fs, c->v[n], fb, (int)U));
            return true;
        }
        if (scalar) {
            uint64_t r = !fs ? toint_s(c, VSF32(n), 3, fb, es, (int)U) : toint_d(c, VSF64(n), 3, fb, es, (int)U);
            c->v[d] = fs ? r : (uint32_t)r;
            return true;
        }
        vw(c, (int)Q, (int)d, fv_toint(c, fs, vq(c, (int)Q, (int)n), 3, fb, (int)U));
        return true;
    }
    if (narrow) {
        if (sz == 3) return false;
        ne = scalar ? 1 : 8 >> sz;
        int op, u = 0;
        switch ((opc << 1) | U) {
        case (0x10 << 1) | 0: op = LOP_SHRN; break;
        case (0x11 << 1) | 0: op = LOP_RSHRN; break;
        case (0x10 << 1) | 1: op = LOP_QSHRUN; break;
        case (0x11 << 1) | 1: op = LOP_QRSHRUN; break;
        case (0x12 << 1) | 0: op = LOP_QSHRN; break;
        case (0x12 << 1) | 1: op = LOP_QSHRN; u = 1; break;
        case (0x13 << 1) | 0: op = LOP_QRSHRN; break;
        default: op = LOP_QRSHRN; u = 1; break;
        }
        if (scalar && (op == LOP_SHRN || op == LOP_RSHRN)) return false;
        int part = scalar ? 0 : (int)Q;
        gen(c, op | (u ? LOP_U : 0), sz, ne, (int)d, sz, part * ne, part, (int)n, sz + 1, 0, -1, 0, 0, -1, (uint64_t)rsh);
        return true;
    }
    if (opc == 0x14) {  // SSHLL / USHLL
        if (scalar || sz == 3) return false;
        ne = 8 >> sz;
        gen(c, LOP_SHLLI | (U ? LOP_U : 0), sz, ne, (int)d, sz + 1, 0, 0, (int)n, sz, (int)Q * ne, -1, 0, 0, -1, (uint64_t)lsh);
        return true;
    }
    if (sz == 3 && !Q) return false;
    if (scalar && sz != 3 && opc != 0x0C && opc != 0x0E) return false;
    if (opc == 0 || opc == 2) {  // SSHR / USHR, SSRA / USRA
        BYSZ(sz, {
            T a = VREG(T, n), o = VREG(T, d), e;
            S sa = VREG(S, n);
            if (U) e = rsh >= es ? (T){} : (a >> rsh);
            else e = (T)(sa >> (rsh < es - 1 ? rsh : es - 1));
            if (opc == 2) e = o + e;
            if (scalar) c->v[d] = (uint64_t)e[0];
            else VW(T, Q, d, e);
        })
        return true;
    }
    switch (opc) {
    case 4: case 6: same(c, (opc == 4 ? LOP_RSHRI : LOP_RSRAI) | (U ? LOP_U : 0), sz, ne, (int)d, (int)n, -1, (uint64_t)rsh, -1); return true;
    case 8:
        if (!U) return false;
        same(c, LOP_SRI, sz, ne, (int)d, (int)n, -1, (uint64_t)rsh, -1);
        return true;
    case 10:
        if (U) { same(c, LOP_SLI, sz, ne, (int)d, (int)n, -1, (uint64_t)lsh, -1); return true; }
        BYSZ(sz, {
            T a = VREG(T, n);
            T e = a << lsh;
            if (scalar) c->v[d] = (uint64_t)e[0];
            else VW(T, Q, d, e);
        })
        return true;
    case 12:
        if (!U) return false;
        same(c, LOP_QSHLUI, sz, ne, (int)d, (int)n, -1, (uint64_t)lsh, -1);
        return true;
    case 14: same(c, LOP_QSHLI | (U ? LOP_U : 0), sz, ne, (int)d, (int)n, -1, (uint64_t)lsh, -1); return true;
    }
    return false;
}

// ---------------------------------------------------------------- indexed element

static bool indexed(Cpu *c, uint32_t w, bool scalar) {
    unsigned Q = bits(w, 30, 1), U = bits(w, 29, 1), sz = bits(w, 22, 2), L = bits(w, 21, 1), M = bits(w, 20, 1),
             rm = bits(w, 16, 4), opc = bits(w, 12, 4), H = bits(w, 11, 1), n = bits(w, 5, 5), d = bits(w, 0, 5);
    int fp = (opc == 1 || opc == 5 || opc == 9) && !(U && opc != 9);
    if (fp) {
        if (sz < 2) return false;
        int fsz = (int)(sz & 1), idx;
        if (fsz) {
            if (L) return false;
            idx = (int)H;
        } else {
            idx = (int)((H << 1) | L);
        }
        unsigned m = (M << 4) | rm;
        if (!scalar && fsz && !Q) return false;
        if (scalar) {
            uint64_t el = fsz ? (uint64_t)((const sa64 *)&c->v[m])[idx] : (uint64_t)((const sa32 *)&c->v[m])[idx];
            if (opc == 9) {
                c->v[d] = fs_op(c, fsz, U ? FOP_MULX : FOP_MUL, fs_get(c, fsz, (int)n), el, 0);
            } else {
                uint64_t x = fs_get(c, fsz, (int)n);
                if (opc != 1) x = fs_neg(fsz, x);
                c->v[d] = fs_op(c, fsz, FOP_MADD, fs_get(c, fsz, (int)d), x, el);
            }
            return true;
        }
        u128 el = fv_splat(fsz, c->v[m], idx, (int)Q);
        if (opc == 9) {
            vw(c, (int)Q, (int)d, fv_op(c, fsz, U ? FOP_MULX : FOP_MUL, vq(c, (int)Q, (int)n), el, 0));
        } else {
            u128 x = vq(c, (int)Q, (int)n);
            if (opc != 1) x = fv_neg(fsz, x);
            vw(c, (int)Q, (int)d, fv_op(c, fsz, FOP_MADD, vq(c, (int)Q, (int)d), x, el));
        }
        return true;
    }
    int idx;
    unsigned m;
    if (sz == 1) {
        idx = (int)((H << 2) | (L << 1) | M);
        m = rm;
    } else if (sz == 2) {
        idx = (int)((H << 1) | L);
        m = (M << 4) | rm;
    } else {
        return false;
    }
    if (scalar) {
        int op;
        if (U) return false;
        switch (opc) {
        case 3: op = LOP_SQDMLAL; break;
        case 7: op = LOP_SQDMLSL; break;
        case 11: op = LOP_SQDMULL; break;
        case 12: op = LOP_SQDMULH; break;
        case 13: op = LOP_SQRDMULH; break;
        default: return false;
        }
        if (op == LOP_SQDMULH || op == LOP_SQRDMULH)
            gen(c, op, (int)sz, 1, (int)d, (int)sz, 0, 0, (int)n, (int)sz, 0, (int)m, (int)sz, 0, idx, 0);
        else
            gen(c, op, (int)sz, 1, (int)d, (int)sz + 1, 0, 0, (int)n, (int)sz, 0, (int)m, (int)sz, 0, idx, 0);
        return true;
    }
    int ne = LANES(Q, sz);
    if (!U && opc == 8) {
        BYSZ(sz, {
            T a = VREG(T, n), mm = VREG(T, m), el;
            for (int i = 0; i < (16 >> sz); i++) el[i] = mm[idx];
            VW(T, Q, d, a * el);
        })
        return true;
    }
    if (U && (opc == 0 || opc == 4)) {
        BYSZ(sz, {
            T a = VREG(T, n), mm = VREG(T, m), o = VREG(T, d), el;
            for (int i = 0; i < (16 >> sz); i++) el[i] = mm[idx];
            VW(T, Q, d, opc == 0 ? o + a * el : o - a * el);
        })
        return true;
    }
    if (!U && (opc == 12 || opc == 13)) {
        same(c, opc == 12 ? LOP_SQDMULH : LOP_SQRDMULH, (int)sz, ne, (int)d, (int)n, (int)m, 0, idx);
        return true;
    }
    int lng;
    int sq = 0;
    switch (opc) {
    case 2: lng = LOP_MLAL; break;
    case 6: lng = LOP_MLSL; break;
    case 10: lng = LOP_MULL; break;
    case 3: lng = LOP_SQDMLAL; sq = 1; break;
    case 7: lng = LOP_SQDMLSL; sq = 1; break;
    case 11: lng = LOP_SQDMULL; sq = 1; break;
    default: return false;
    }
    if (U && sq) return false;
    ne = 8 >> sz;
    gen(c, lng | ((!sq && U) ? LOP_U : 0), (int)sz, ne, (int)d, (int)sz + 1, 0, 0, (int)n, (int)sz, (int)Q * ne, (int)m,
        (int)sz, 0, idx, 0);
    return true;
}

// ---------------------------------------------------------------- top level

static bool simd_dp(Cpu *c, uint32_t w) {
    unsigned op0 = bits(w, 28, 4), op1 = bits(w, 23, 2), op2 = bits(w, 19, 4), op3 = bits(w, 10, 9);
    if (op0 == 4 && !(op1 & 2) && (op2 & 7) == 5 && (op3 & 0x183) == 2) return crypto_aes(c, w);
    if (op0 == 5 && !(op1 & 2) && !(op2 & 4) && (op3 & 0x023) == 0) return crypto_sha3(c, w);
    if (op0 == 5 && !(op1 & 2) && (op2 & 7) == 5 && (op3 & 0x183) == 2) return crypto_sha2(c, w);
    if ((op0 & 0xD) == 5) {  // AdvSIMD scalar
        if (op1 == 0 && !(op2 & 0xC) && (op3 & 0x021) == 1) return scalar_copy(c, w);
        if (!(op1 & 2) && (op2 & 7) == 4 && (op3 & 0x183) == 2) return misc2(c, w, true);
        if (!(op1 & 2) && (op2 & 7) == 6 && (op3 & 0x183) == 2) return scalar_pairwise(c, w);
        if (!(op1 & 2) && (op2 & 4) && (op3 & 3) == 0) return three_diff(c, w, true);
        if (!(op1 & 2) && (op2 & 4) && (op3 & 1)) return three_same(c, w, true);
        if (op1 == 2 && (op3 & 1)) return shift_imm(c, w, true);
        if ((op1 & 2) && !(op3 & 1)) return indexed(c, w, true);
        return false;
    }
    if ((op0 & 0xB) == 0 && !(op1 & 2) && !(op2 & 4) && (op3 & 0x023) == 0) return vtbl(c, w);
    if ((op0 & 0xB) == 0 && !(op1 & 2) && !(op2 & 4) && (op3 & 0x023) == 2) return permute(c, w);
    if ((op0 & 0xB) == 2 && !(op1 & 2) && !(op2 & 4) && (op3 & 0x021) == 0) return vext(c, w);
    if ((op0 & 0x9) == 0) {  // AdvSIMD vector
        if (op1 == 0 && !(op2 & 0xC) && (op3 & 0x021) == 1) return vcopy(c, w);
        if (!(op1 & 2) && (op2 & 4) && (op3 & 1)) return three_same(c, w, false);
        if (!(op1 & 2) && (op2 & 4) && (op3 & 3) == 0) return three_diff(c, w, false);
        if (!(op1 & 2) && (op2 & 7) == 4 && (op3 & 0x183) == 2) return misc2(c, w, false);
        if (!(op1 & 2) && (op2 & 7) == 6 && (op3 & 0x183) == 2) return across(c, w);
        if (op1 == 2 && op2 == 0 && (op3 & 1)) return modimm(c, w);
        if (op1 == 2 && op2 && (op3 & 1)) return shift_imm(c, w, false);
        if ((op1 & 2) && !(op3 & 1)) return indexed(c, w, false);
        return false;
    }
    if ((op0 & 5) == 1) {  // scalar FP
        if (bits(w, 29, 1) || (bits(w, 31, 1) && ((op1 & 2) || ((op2 & 4) && (op3 & 0x3F))))) return false;
        if (op1 & 2) return fp_dp3(c, w);
        if (!(op2 & 4)) return fp_fixed(c, w);
        if ((op3 & 0x3F) == 0) return fp_int(c, w);
        if ((op3 & 0x1F) == 0x10) return fp_dp1(c, w);
        if ((op3 & 0xF) == 8) return fp_cmp(c, w);
        if ((op3 & 7) == 4) return fp_imm(c, w);
        if ((op3 & 3) == 1) return fp_ccmp(c, w);
        if ((op3 & 3) == 2) return fp_dp2(c, w);
        if ((op3 & 3) == 3) return fp_csel(c, w);
        return false;
    }
    return false;
}
