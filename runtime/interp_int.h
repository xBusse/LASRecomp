// Interpreter internals shared by interp.c (the loop) and interp_decode.c (instruction -> Op).
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "cpu.h"

// A decoded instruction. 16 bytes, one per guest instruction word (the array parallels the module's text), all zero
// (k == K_UNDEF) until the instruction first runs. Register fields are indexes into the Cpu's first 34 qwords: 0-30
// x0-x30, 31 the zero register (reads 0, never written), 32 sp, 33 a scratch slot for results that are thrown away
// (it overlays Cpu.pc, which is only meaningful just before a call out). Whether a field means sp or xzr is settled at
// decode time, so the handlers never look.
typedef struct Op {
    uint16_t k;
    uint8_t d, n, m, a, x, y;
    int64_t imm;
} Op;
_Static_assert(sizeof(Op) == 16, "Op");

#define REG_ZR 31
#define REG_SP 32
#define REG_DUMP 33

// op list: UNDEF must be first (k == 0)
#define INTERP_OPS(X) \
    X(UNDEF) X(UDF) X(NOP) X(SIMD) X(HOOK) X(PATCH) \
    X(MOVI) X(MOVK64) X(MOVK32) \
    X(ADDI64) X(ADDI32) X(ADDSI64) X(ADDSI32) X(SUBSI64) X(SUBSI32) \
    X(ANDI64) X(ANDI32) X(ORRI64) X(ORRI32) X(EORI64) X(EORI32) X(ANDSI64) X(ANDSI32) \
    X(UBFM64) X(UBFM32) X(SBFM64) X(SBFM32) X(BFM64) X(BFM32) X(EXTR64) X(EXTR32) \
    X(MOVR64) X(MOVR32) \
    X(ADDR64) X(ADDR32) X(SUBR64) X(SUBR32) X(ADDSR64) X(ADDSR32) X(SUBSR64) X(SUBSR32) \
    X(ANDR64) X(ORRR64) X(EORR64) X(BICR64) X(ORNR64) X(EONR64) X(ANDSR64) X(BICSR64) \
    X(ANDR32) X(ORRR32) X(EORR32) X(BICR32) X(ORNR32) X(EONR32) X(ANDSR32) X(BICSR32) \
    X(ADDX64) X(ADDX32) X(SUBX64) X(SUBX32) X(ADDSX64) X(ADDSX32) X(SUBSX64) X(SUBSX32) \
    X(ADC64) X(ADC32) X(SBC64) X(SBC32) X(ADCS64) X(ADCS32) X(SBCS64) X(SBCS32) \
    X(CCMNR64) X(CCMNR32) X(CCMPR64) X(CCMPR32) X(CCMNI64) X(CCMNI32) X(CCMPI64) X(CCMPI32) \
    X(CSEL64) X(CSINC64) X(CSINV64) X(CSNEG64) X(CSEL32) X(CSINC32) X(CSINV32) X(CSNEG32) \
    X(MADD64) X(MSUB64) X(MADD32) X(MSUB32) X(SMADDL) X(SMSUBL) X(UMADDL) X(UMSUBL) X(SMULH) X(UMULH) \
    X(UDIV64) X(UDIV32) X(SDIV64) X(SDIV32) X(LSLV64) X(LSLV32) X(LSRV64) X(LSRV32) \
    X(ASRV64) X(ASRV32) X(RORV64) X(RORV32) \
    X(RBIT64) X(RBIT32) X(REV16_64) X(REV16_32) X(REV32_64) X(REV32) X(REV64) X(CLZ64) X(CLZ32) X(CLS64) X(CLS32) \
    X(CRC) \
    X(B_NEAR) X(B_FAR) X(BL_NEAR) X(BL_FAR) X(BCC) X(CBZ64) X(CBNZ64) X(CBZ32) X(CBNZ32) X(TBZ) X(TBNZ) \
    X(BR) X(BLR) X(RET) X(SVC) X(BRK) \
    X(MRS_CONST) X(MRS_TPIDR) X(MRS_TPIDRRO) X(MRS_TICKS) X(MRS_NZCV) X(MRS_FPCR) X(MRS_FPSR) \
    X(MSR_TPIDR) X(MSR_NZCV) X(MSR_FPCR) X(MSR_FPSR) X(DCZVA) X(FENCE) X(CLREX) \
    X(LDRW_ABS) X(LDRX_ABS) X(LDRSW_ABS) \
    X(LDXR) X(STXR) X(LDAR) X(STLR) \
    LDST_OPS(X, O) LDST_OPS(X, PRE) LDST_OPS(X, POST) LDST_OPS(X, REG) \
    PAIR_OPS(X, O) PAIR_OPS(X, PRE) PAIR_OPS(X, POST)

// single register loads and stores, per addressing mode: [base + imm], pre-indexed, post-indexed, [base + ext(rm)]
#define LDST_OPS(X, M) \
    X(LDRB_##M) X(LDRH_##M) X(LDRW_##M) X(LDRX_##M) X(LDRSB64_##M) X(LDRSH64_##M) X(LDRSW64_##M) X(LDRSB32_##M) \
    X(LDRSH32_##M) X(STRB_##M) X(STRH_##M) X(STRW_##M) X(STRX_##M)
#define PAIR_OPS(X, M) X(LDPW_##M) X(LDPX_##M) X(LDPSW_##M) X(STPW_##M) X(STPX_##M)

enum {
#define X(n) K_##n,
    INTERP_OPS(X)
#undef X
    K_COUNT
};

// A module whose text is interpreted: ops[i] decodes the word at lo + 4 * i.
typedef struct IMod {
    uint64_t lo, hi;
    Op *ops;
} IMod;

// Fills *op (everything but k first) for instruction w at pc, then sets op->k with release ordering.
// Never leaves k == K_UNDEF: encodings the lifter rejects become K_UDF. mod: for branch targets.
void interp_decode(const IMod *mod, Op *op, uint64_t pc, uint32_t w);

// ---- shared small helpers (also used by interp_simd.c)
static inline uint64_t cond_holds(uint64_t nzcv, unsigned cc) {
    if (cc >= 14) return 1;
    unsigned n = (nzcv >> 31) & 1, z = (nzcv >> 30) & 1, cy = (nzcv >> 29) & 1, v = (nzcv >> 28) & 1;
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
