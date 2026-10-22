// Instruction word -> Op for the interpreter (runtime/interp.h). A line-by-line port of tools/lifter.py's integer,
// branch, system and load/store decoding; FP / AdvSIMD words are only classified here (K_SIMD) and decoded when they
// run (interp_simd.c, the port of tools/simd_c.py). Encodings the lifter rejects (Unsupported) become K_UDF, like the
// generated code's rt_udf.
#include <string.h>

#include "interp_int.h"

static inline unsigned bits(uint32_t w, int lo, int n) { return (w >> lo) & ((1u << n) - 1u); }
static inline int64_t sext(uint64_t v, int n) { return (int64_t)(v << (64 - n)) >> (64 - n); }
static inline uint64_t ones(int n) { return n >= 64 ? ~0ull : (1ull << n) - 1; }

static inline uint8_t rz(unsigned r) { return (uint8_t)r; }                         // source: 31 = zr
static inline uint8_t rsp(unsigned r) { return r == 31 ? REG_SP : (uint8_t)r; }     // source or dest: 31 = sp
static inline uint8_t wz(unsigned r) { return r == 31 ? REG_DUMP : (uint8_t)r; }    // dest: 31 = zr (thrown away)

static uint64_t ror_(uint64_t v, unsigned r, unsigned size) {
    r %= size;
    return r ? ((v >> r) | (v << (size - r))) & ones(size) : v;
}

static bool bit_masks(unsigned N, unsigned imms, unsigned immr, unsigned size, bool immediate, uint64_t *wmask,
                      uint64_t *tmask) {
    unsigned x = (N << 6) | (~imms & 0x3F);
    if (!x) return false;
    int length = 31 - __builtin_clz(x);
    if (length < 1) return false;
    unsigned levels = (1u << length) - 1;
    if (immediate && (imms & levels) == levels) return false;
    unsigned S = imms & levels, R = immr & levels;
    unsigned diff = (S - R) & 0x3F;
    unsigned esize = 1u << length;
    unsigned d = diff & levels;
    uint64_t w = ror_(ones(S + 1), R, esize), t = ones(d + 1), wm = 0, tm = 0;
    for (unsigned i = 0; i < size; i += esize) wm |= w << i, tm |= t << i;
    *wmask = wm;
    *tmask = tm;
    return true;
}

// ---------------------------------------------------------------- data processing (immediate)

static bool dp_imm(Op *o, uint32_t w, uint64_t pc) {
    unsigned op = bits(w, 23, 3), sf = bits(w, 31, 1), rd = bits(w, 0, 5), rn = bits(w, 5, 5);
    switch (op) {
    case 0: case 1: {  // ADR / ADRP
        int64_t imm = sext(((uint64_t)bits(w, 5, 19) << 2) | bits(w, 29, 2), 21);
        o->k = K_MOVI;
        o->d = wz(rd);
        o->imm = (int64_t)(bits(w, 31, 1) ? (pc & ~0xFFFull) + ((uint64_t)imm << 12) : pc + (uint64_t)imm);
        return true;
    }
    case 2: {  // add / sub immediate
        unsigned sub = bits(w, 30, 1), S = bits(w, 29, 1), sh = bits(w, 22, 1);
        uint64_t imm = (uint64_t)bits(w, 10, 12) << (sh ? 12 : 0);
        o->n = rsp(rn);
        if (S) {
            o->d = wz(rd);
            o->imm = (int64_t)imm;
            o->k = sub ? (sf ? K_SUBSI64 : K_SUBSI32) : (sf ? K_ADDSI64 : K_ADDSI32);
        } else {
            o->d = rsp(rd);
            o->imm = (int64_t)(sub ? -imm : imm);
            o->k = sf ? K_ADDI64 : K_ADDI32;
        }
        return true;
    }
    case 4: {  // logical immediate
        unsigned opc = bits(w, 29, 2), N = bits(w, 22, 1);
        if (!sf && N) return false;
        unsigned size = sf ? 64 : 32;
        uint64_t imm, dummy;
        if (!bit_masks(N, bits(w, 10, 6), bits(w, 16, 6), size, true, &imm, &dummy)) return false;
        o->n = rz(rn);
        o->imm = (int64_t)imm;
        o->d = opc == 3 ? wz(rd) : rsp(rd);
        static const uint16_t k64[4] = {K_ANDI64, K_ORRI64, K_EORI64, K_ANDSI64};
        static const uint16_t k32[4] = {K_ANDI32, K_ORRI32, K_EORI32, K_ANDSI32};
        o->k = sf ? k64[opc] : k32[opc];
        return true;
    }
    case 5: {  // move wide
        unsigned opc = bits(w, 29, 2), hw = bits(w, 21, 2);
        if (!sf && hw > 1) return false;
        uint64_t imm = (uint64_t)bits(w, 5, 16) << (hw * 16);
        o->d = wz(rd);
        if (opc == 0) {
            o->k = K_MOVI;
            o->imm = (int64_t)(~imm & ones(sf ? 64 : 32));
        } else if (opc == 2) {
            o->k = K_MOVI;
            o->imm = (int64_t)imm;
        } else if (opc == 3) {
            o->k = sf ? K_MOVK64 : K_MOVK32;
            o->d = rz(rd);  // (reads and writes it: xzr as the target is a plain no-effect write to the dump slot)
            if (rd == 31) o->d = REG_DUMP;
            o->x = (uint8_t)(hw * 16);
            o->imm = (int64_t)imm;
        } else {
            return false;
        }
        return true;
    }
    case 6: {  // bitfield
        unsigned opc = bits(w, 29, 2), N = bits(w, 22, 1), immr = bits(w, 16, 6), imms = bits(w, 10, 6);
        unsigned size = sf ? 64 : 32;
        if (N != sf) return false;
        uint64_t wmask, tmask;
        if (!bit_masks(N, imms, immr, size, false, &wmask, &tmask)) return false;
        o->d = wz(rd);
        o->n = rz(rn);
        o->x = (uint8_t)immr;
        o->y = (uint8_t)imms;
        if (opc == 2) {
            o->k = sf ? K_UBFM64 : K_UBFM32;
            o->imm = (int64_t)(wmask & tmask);
        } else if (opc == 0) {
            o->k = sf ? K_SBFM64 : K_SBFM32;
            o->imm = (int64_t)(wmask & tmask);
        } else if (opc == 1) {
            o->k = sf ? K_BFM64 : K_BFM32;
            o->imm = (int64_t)wmask;
            o->a = rz(rd);  // (the old value of rd; a read, unlike d, which may be the dump slot)
        } else {
            return false;
        }
        return true;
    }
    case 7: {  // EXTR
        o->d = wz(rd);
        o->n = rz(rn);
        o->m = rz(bits(w, 16, 5));
        o->x = (uint8_t)bits(w, 10, 6);
        o->k = sf ? K_EXTR64 : K_EXTR32;
        return true;
    }
    }
    return false;
}

// ---------------------------------------------------------------- branches / system

static bool near_target(const IMod *mod, uint64_t t) { return t >= mod->lo && t < mod->hi; }

static bool system_insn(Op *o, uint32_t w) {
    unsigned L = bits(w, 21, 1), op0 = bits(w, 19, 2), op1 = bits(w, 16, 3), crn = bits(w, 12, 4), crm = bits(w, 8, 4),
             op2 = bits(w, 5, 3), rt = bits(w, 0, 5);
    if (op0 == 0 && !L) {
        if (crn == 2 && op1 == 3 && rt == 31) { o->k = K_NOP; return true; }  // hints
        if (crn == 3 && op1 == 3 && rt == 31) { o->k = op2 == 2 ? K_CLREX : K_FENCE; return true; }
        return false;
    }
    if (op0 == 1) {  // SYS
        if (!L && op1 == 3 && crn == 7 && crm == 4 && op2 == 1) {
            o->k = K_DCZVA;
            o->n = rz(rt);
        } else {
            o->k = K_NOP;  // cache maintenance
        }
        return true;
    }
    unsigned key = (op0 << 16) | (op1 << 12) | (crn << 8) | (crm << 4) | op2;
#define KEY(a, b, c, d, e) (((a) << 16) | ((b) << 12) | ((c) << 8) | ((d) << 4) | (e))
    if (L) {
        o->d = wz(rt);
        switch (key) {
        case KEY(3, 3, 13, 0, 3): o->k = K_MRS_TPIDRRO; return true;
        case KEY(3, 3, 13, 0, 2): o->k = K_MRS_TPIDR; return true;
        case KEY(3, 3, 14, 0, 0): o->k = K_MRS_CONST; o->imm = 19200000; return true;
        case KEY(3, 3, 14, 0, 1): case KEY(3, 3, 14, 0, 2): o->k = K_MRS_TICKS; return true;
        case KEY(3, 3, 4, 2, 0): o->k = K_MRS_NZCV; return true;
        case KEY(3, 3, 4, 4, 0): o->k = K_MRS_FPCR; return true;
        case KEY(3, 3, 4, 4, 1): o->k = K_MRS_FPSR; return true;
        case KEY(3, 3, 0, 0, 7): o->k = K_MRS_CONST; o->imm = 4; return true;
        case KEY(3, 3, 0, 0, 1): o->k = K_MRS_CONST; o->imm = 0x8444c004; return true;
        }
        return false;
    }
    o->n = rz(rt);
    switch (key) {
    case KEY(3, 3, 13, 0, 2): o->k = K_MSR_TPIDR; return true;
    case KEY(3, 3, 4, 2, 0): o->k = K_MSR_NZCV; return true;
    case KEY(3, 3, 4, 4, 0): o->k = K_MSR_FPCR; return true;
    case KEY(3, 3, 4, 4, 1): o->k = K_MSR_FPSR; return true;
    }
#undef KEY
    return false;
}

static bool branch_sys(const IMod *mod, Op *o, uint32_t w, uint64_t pc) {
    if (bits(w, 26, 5) == 0x05) {  // B / BL
        uint64_t t = pc + (uint64_t)(sext(bits(w, 0, 26), 26) * 4);
        bool link = bits(w, 31, 1);
        if (near_target(mod, t)) {
            o->k = link ? K_BL_NEAR : K_B_NEAR;
            o->imm = ((int64_t)t - (int64_t)pc) / 4;
        } else {
            o->k = link ? K_BL_FAR : K_B_FAR;
            o->imm = (int64_t)t;
        }
        return true;
    }
    if (bits(w, 25, 7) == 0x2A && !bits(w, 4, 1)) {  // B.cond
        uint64_t t = pc + (uint64_t)(sext(bits(w, 5, 19), 19) * 4);
        unsigned cond = bits(w, 0, 4);
        if (!near_target(mod, t)) return false;
        o->k = cond >= 14 ? K_B_NEAR : K_BCC;
        o->x = (uint8_t)cond;
        o->imm = ((int64_t)t - (int64_t)pc) / 4;
        return true;
    }
    if (bits(w, 25, 6) == 0x1A) {  // CBZ / CBNZ
        unsigned sf = bits(w, 31, 1), nz = bits(w, 24, 1);
        uint64_t t = pc + (uint64_t)(sext(bits(w, 5, 19), 19) * 4);
        if (!near_target(mod, t)) return false;
        o->k = sf ? (nz ? K_CBNZ64 : K_CBZ64) : (nz ? K_CBNZ32 : K_CBZ32);
        o->n = rz(bits(w, 0, 5));
        o->imm = ((int64_t)t - (int64_t)pc) / 4;
        return true;
    }
    if (bits(w, 25, 6) == 0x1B) {  // TBZ / TBNZ
        unsigned b = (bits(w, 31, 1) << 5) | bits(w, 19, 5), nz = bits(w, 24, 1);
        uint64_t t = pc + (uint64_t)(sext(bits(w, 5, 14), 14) * 4);
        if (!near_target(mod, t)) return false;
        o->k = nz ? K_TBNZ : K_TBZ;
        o->n = rz(bits(w, 0, 5));
        o->x = (uint8_t)b;
        o->imm = ((int64_t)t - (int64_t)pc) / 4;
        return true;
    }
    if (bits(w, 25, 7) == 0x6B) {  // branch register
        unsigned opc = bits(w, 21, 4), rn = bits(w, 5, 5);
        if (bits(w, 16, 5) != 31 || bits(w, 10, 6) || bits(w, 0, 5)) return false;
        if (opc == 0) { o->k = K_BR; o->n = rz(rn); return true; }
        if (opc == 1) { o->k = K_BLR; o->n = rz(rn); return true; }
        if (opc == 2) { o->k = K_RET; return true; }
        return false;
    }
    if (bits(w, 24, 8) == 0xD4) {  // exception generation
        unsigned opc = bits(w, 21, 3), ll = bits(w, 0, 5), imm = bits(w, 5, 16);
        if (opc == 0 && ll == 1) { o->k = K_SVC; o->imm = imm; return true; }
        if (opc == 1 && ll == 0) { o->k = K_BRK; o->imm = imm; return true; }
        return false;
    }
    if (bits(w, 22, 10) == 0x354) return system_insn(o, w);
    return false;
}

// ---------------------------------------------------------------- loads / stores

static bool ldst(Op *o, uint32_t w, uint64_t pc) {
    if (bits(w, 26, 1)) {  // FP / SIMD
        o->k = K_SIMD;
        o->imm = w;
        return true;
    }
    if (bits(w, 24, 6) == 0x08) {  // exclusive / acquire-release
        unsigned size = bits(w, 30, 2), o2 = bits(w, 23, 1), L = bits(w, 22, 1), o1 = bits(w, 21, 1),
                 rs = bits(w, 16, 5), rn = bits(w, 5, 5), rt = bits(w, 0, 5);
        if (o1) return false;
        o->y = (uint8_t)size;
        o->n = rsp(rn);
        if (o2 == 0) {
            if (L) { o->k = K_LDXR; o->d = wz(rt); }
            else { o->k = K_STXR; o->d = rz(rt); o->m = wz(rs); }
        } else {
            if (L) { o->k = K_LDAR; o->d = wz(rt); }
            else { o->k = K_STLR; o->d = rz(rt); }
        }
        return true;
    }
    if (bits(w, 27, 3) == 3 && bits(w, 24, 2) == 0) {  // literal
        unsigned opc = bits(w, 30, 2), rt = bits(w, 0, 5);
        uint64_t addr = pc + (uint64_t)(sext(bits(w, 5, 19), 19) * 4);
        o->d = wz(rt);
        o->imm = (int64_t)addr;
        if (opc == 0) o->k = K_LDRW_ABS;
        else if (opc == 1) o->k = K_LDRX_ABS;
        else if (opc == 2) o->k = K_LDRSW_ABS;
        else o->k = K_NOP;  // prfm
        return true;
    }
    if (bits(w, 27, 3) == 5) {  // pair
        unsigned opc = bits(w, 30, 2), mode = bits(w, 23, 2), L = bits(w, 22, 1);
        int64_t imm7 = sext(bits(w, 15, 7), 7);
        unsigned rt2 = bits(w, 10, 5), rn = bits(w, 5, 5), rt = bits(w, 0, 5);
        if (opc == 3) return false;
        int scale = opc == 2 ? 3 : 2;
        o->imm = imm7 * (1 << scale);
        o->n = rsp(rn);
        if (L) { o->d = wz(rt); o->m = wz(rt2); }
        else { o->d = rz(rt); o->m = rz(rt2); }
        // modes: 1 post-index, 3 pre-index, 0 / 2 offset
        int mi = mode == 1 ? 2 : mode == 3 ? 1 : 0;
        static const uint16_t ldw[3] = {K_LDPW_O, K_LDPW_PRE, K_LDPW_POST};
        static const uint16_t ldx[3] = {K_LDPX_O, K_LDPX_PRE, K_LDPX_POST};
        static const uint16_t lds[3] = {K_LDPSW_O, K_LDPSW_PRE, K_LDPSW_POST};
        static const uint16_t stw[3] = {K_STPW_O, K_STPW_PRE, K_STPW_POST};
        static const uint16_t stx[3] = {K_STPX_O, K_STPX_PRE, K_STPX_POST};
        if (L) o->k = opc == 2 ? ldx[mi] : opc == 1 ? lds[mi] : ldw[mi];
        else o->k = opc == 2 ? stx[mi] : stw[mi];
        return true;
    }
    if (bits(w, 27, 3) == 7) {  // register
        unsigned size = bits(w, 30, 2), opc = bits(w, 22, 2), rn = bits(w, 5, 5), rt = bits(w, 0, 5);
        int mode;  // 0 offset, 1 pre, 2 post, 3 register offset
        int64_t imm = 0;
        o->n = rsp(rn);
        if (bits(w, 24, 2) == 1) {
            mode = 0;
            imm = (int64_t)((uint64_t)bits(w, 10, 12) << size);
        } else {
            unsigned kind = bits(w, 10, 2);
            if (bits(w, 21, 1)) {
                if (kind != 2) return false;
                unsigned rm = bits(w, 16, 5), option = bits(w, 13, 3), S = bits(w, 12, 1);
                if (option != 2 && option != 3 && option != 6 && option != 7) return false;
                mode = 3;
                o->m = rz(rm);
                o->x = (uint8_t)option;
                o->y = (uint8_t)(S ? size : 0);
            } else {
                imm = sext(bits(w, 12, 9), 9);
                mode = kind == 1 ? 2 : kind == 3 ? 1 : 0;
            }
        }
        o->imm = imm;
        uint16_t base;  // first op of the kind for mode 0; the kinds are laid out per mode in the same order
        enum { LB, LH, LW, LX, LSB64, LSH64, LSW64, LSB32, LSH32, SB, SH, SW, SX };
        int kind;
        if (opc == 0) {
            kind = SB + (int)size;
            o->d = rz(rt);
        } else if (opc == 1) {
            kind = LB + (int)size;
            o->d = wz(rt);
        } else if (size == 3) {
            o->k = K_NOP;  // prfm
            return true;
        } else if (size == 2 && opc == 3) {
            return false;
        } else {
            o->d = wz(rt);
            kind = opc == 2 ? LSB64 + (int)size : LSB32 + (int)size;
        }
        static const uint16_t first[4] = {K_LDRB_O, K_LDRB_PRE, K_LDRB_POST, K_LDRB_REG};
        base = first[mode];
        o->k = (uint16_t)(base + kind);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------- data processing (register)

static bool dp_reg(Op *o, uint32_t w) {
    unsigned sf = bits(w, 31, 1), rd = bits(w, 0, 5), rn = bits(w, 5, 5), rm = bits(w, 16, 5);
    if (!bits(w, 28, 1)) {
        if (bits(w, 24, 4) == 0xA) {  // logical, shifted register
            unsigned opc = bits(w, 29, 2), sh = bits(w, 22, 2), N = bits(w, 21, 1), amt = bits(w, 10, 6);
            if (!sf && amt >= 32) return false;
            o->d = wz(rd);
            o->n = rz(rn);
            o->m = rz(rm);
            o->x = (uint8_t)sh;
            o->y = (uint8_t)amt;
            static const uint16_t k64[8] = {K_ANDR64, K_BICR64, K_ORRR64, K_ORNR64, K_EORR64, K_EONR64, K_ANDSR64, K_BICSR64};
            static const uint16_t k32[8] = {K_ANDR32, K_BICR32, K_ORRR32, K_ORNR32, K_EORR32, K_EONR32, K_ANDSR32, K_BICSR32};
            o->k = (sf ? k64 : k32)[opc * 2 + N];
            if (opc == 1 && !N && rn == 31 && !amt) o->k = sf ? K_MOVR64 : K_MOVR32;
            return true;
        }
        if (bits(w, 24, 4) == 0xB) {  // add / sub, shifted or extended register
            unsigned op = bits(w, 30, 1), S = bits(w, 29, 1);
            if (!bits(w, 21, 1)) {
                unsigned sh = bits(w, 22, 2), amt = bits(w, 10, 6);
                if (sh == 3) return false;
                o->d = wz(rd);
                o->n = rz(rn);
                o->m = rz(rm);
                o->x = (uint8_t)sh;
                o->y = (uint8_t)amt;
                static const uint16_t k64[4] = {K_ADDR64, K_SUBR64, K_ADDSR64, K_SUBSR64};
                static const uint16_t k32[4] = {K_ADDR32, K_SUBR32, K_ADDSR32, K_SUBSR32};
                o->k = (sf ? k64 : k32)[S * 2 + op];
            } else {
                if (bits(w, 22, 2)) return false;
                o->d = S ? wz(rd) : rsp(rd);
                o->n = rsp(rn);
                o->m = rz(rm);
                o->x = (uint8_t)bits(w, 13, 3);
                o->y = (uint8_t)bits(w, 10, 3);
                static const uint16_t k64[4] = {K_ADDX64, K_SUBX64, K_ADDSX64, K_SUBSX64};
                static const uint16_t k32[4] = {K_ADDX32, K_SUBX32, K_ADDSX32, K_SUBSX32};
                o->k = (sf ? k64 : k32)[S * 2 + op];
            }
            return true;
        }
        return false;
    }
    unsigned op2 = bits(w, 21, 4);
    o->d = wz(rd);
    o->n = rz(rn);
    o->m = rz(rm);
    if (bits(w, 24, 4) == 0xB) {  // three source
        unsigned op54 = bits(w, 29, 2), op31 = bits(w, 21, 3), o0 = bits(w, 15, 1);
        o->a = rz(bits(w, 10, 5));
        if (op54) return false;
        switch (op31) {
        case 0: o->k = sf ? (o0 ? K_MSUB64 : K_MADD64) : (o0 ? K_MSUB32 : K_MADD32); return true;
        case 1: o->k = o0 ? K_SMSUBL : K_SMADDL; return true;
        case 5: o->k = o0 ? K_UMSUBL : K_UMADDL; return true;
        case 2: o->k = K_SMULH; return true;
        case 6: o->k = K_UMULH; return true;
        }
        return false;
    }
    if (op2 == 0) {  // add / subtract with carry
        unsigned op = bits(w, 30, 1), S = bits(w, 29, 1);
        static const uint16_t k64[4] = {K_ADC64, K_ADCS64, K_SBC64, K_SBCS64};
        static const uint16_t k32[4] = {K_ADC32, K_ADCS32, K_SBC32, K_SBCS32};
        o->k = (sf ? k64 : k32)[op * 2 + S];
        return true;
    }
    if (op2 == 2) {  // conditional compare
        unsigned op = bits(w, 30, 1), imm_form = bits(w, 11, 1), cond = bits(w, 12, 4), nzcv = bits(w, 0, 4);
        o->x = (uint8_t)cond;
        o->y = (uint8_t)nzcv;
        o->d = 0;
        if (imm_form) {
            o->imm = rm;
            static const uint16_t k64[2] = {K_CCMNI64, K_CCMPI64}, k32[2] = {K_CCMNI32, K_CCMPI32};
            o->k = (sf ? k64 : k32)[op];
        } else {
            static const uint16_t k64[2] = {K_CCMNR64, K_CCMPR64}, k32[2] = {K_CCMNR32, K_CCMPR32};
            o->k = (sf ? k64 : k32)[op];
        }
        return true;
    }
    if (op2 == 4) {  // conditional select
        unsigned op = bits(w, 30, 1), o2 = bits(w, 10, 2), cond = bits(w, 12, 4);
        if (o2 > 1) return false;
        o->x = (uint8_t)cond;
        static const uint16_t k64[4] = {K_CSEL64, K_CSINC64, K_CSINV64, K_CSNEG64};
        static const uint16_t k32[4] = {K_CSEL32, K_CSINC32, K_CSINV32, K_CSNEG32};
        o->k = (sf ? k64 : k32)[op * 2 + o2];
        return true;
    }
    if (op2 == 6) {
        unsigned opcode = bits(w, 10, 6);
        if (bits(w, 30, 1)) {  // one source
            if (bits(w, 16, 5)) return false;
            switch (opcode) {
            case 0: o->k = sf ? K_RBIT64 : K_RBIT32; return true;
            case 1: o->k = sf ? K_REV16_64 : K_REV16_32; return true;
            case 2: o->k = sf ? K_REV32_64 : K_REV32; return true;
            case 3: if (!sf) return false; o->k = K_REV64; return true;
            case 4: o->k = sf ? K_CLZ64 : K_CLZ32; return true;
            case 5: o->k = sf ? K_CLS64 : K_CLS32; return true;
            }
            return false;
        }
        if (opcode >= 16 && opcode <= 23) {  // CRC32
            unsigned C = bits(w, 12, 1), sz = bits(w, 10, 2);
            if ((sz == 3) != (sf != 0)) return false;
            o->k = K_CRC;
            o->d = wz(rd);
            o->x = (uint8_t)C;
            o->y = (uint8_t)sz;
            return true;
        }
        switch (opcode) {
        case 2: o->k = sf ? K_UDIV64 : K_UDIV32; return true;
        case 3: o->k = sf ? K_SDIV64 : K_SDIV32; return true;
        case 8: o->k = sf ? K_LSLV64 : K_LSLV32; return true;
        case 9: o->k = sf ? K_LSRV64 : K_LSRV32; return true;
        case 10: o->k = sf ? K_ASRV64 : K_ASRV32; return true;
        case 11: o->k = sf ? K_RORV64 : K_RORV32; return true;
        }
        return false;
    }
    return false;
}

// ---------------------------------------------------------------- entry

void interp_decode(const IMod *mod, Op *op, uint64_t pc, uint32_t w) {
    Op t;
    memset(&t, 0, sizeof t);
    t.k = K_UDF;
    bool ok;
    unsigned op0 = bits(w, 25, 4);
    if ((op0 & 0xE) == 0x8) ok = dp_imm(&t, w, pc);
    else if ((op0 & 0xE) == 0xA) ok = branch_sys(mod, &t, w, pc);
    else if ((op0 & 5) == 4) ok = ldst(&t, w, pc);
    else if ((op0 & 7) == 5) ok = dp_reg(&t, w);
    else if ((op0 & 7) == 7) { t.k = K_SIMD; t.imm = w; ok = true; }
    else ok = false;
    if (!ok) memset(&t, 0, sizeof t), t.k = K_UDF;
    uint16_t k = t.k;
    op->d = t.d; op->n = t.n; op->m = t.m; op->a = t.a; op->x = t.x; op->y = t.y;
    op->imm = t.imm;
    __atomic_store_n(&op->k, k, __ATOMIC_RELEASE);
}
