// The interpreter tier (see interp.h): module setup, the instruction loop, hooks for host replacements.
//
// One `run` call is one guest function activation, like one generated function: `bl` / `blr` run the callee in a
// nested `run`, `ret` returns, `b` / `br` go on in the same loop (a tail call). So the host stack holds the guest's
// call chain exactly as it does in the generated code, and HLE functions, overrides and the generated code (if a
// library is loaded next to the interpreter) mix freely through rt_dispatch.
#include "interp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "interp_int.h"
#ifndef INTERP_TEST
#include "game_addresses.h"
#endif
#include "rt.h"

#ifndef INTERP_TEST
#include "rando.h"
HostFn dispatch_override_lookup(uint64_t addr);  // dispatch.c
#endif
uint32_t interp_crc(int castagnoli, uint32_t acc, uint64_t v, int bytes);  // interp_simd.c

typedef uint64_t __attribute__((may_alias)) ru64;
#define R(i) (*(ru64 *)((char *)c + 8u * (i)))
#define D R(op->d)
#define N R(op->n)
#define M R(op->m)
#define A R(op->a)
#define N32 ((uint32_t)R(op->n))
#define M32 ((uint32_t)R(op->m))
static inline uint64_t ones64(int n) { return n >= 64 ? ~0ull : (1ull << n) - 1; }
#define PCOF(o) (m->lo + ((uint64_t)((o) - m->ops) << 2))

static IMod g_mods[4];
static int g_nmods;
static bool g_active;

#ifdef INTERP_TEST
static int g_step_fail;
#endif

static inline Op *lookup(uint64_t t, IMod **pm) {
    for (int i = 0; i < g_nmods; i++) {
        IMod *m = &g_mods[i];
        if (t - m->lo < m->hi - m->lo) {
            if (t & 3) return NULL;
            *pm = m;
            return &m->ops[(t - m->lo) >> 2];
        }
    }
    return NULL;
}

static void run(Cpu *c, IMod *m, Op *op);

#ifdef INTERP_TRACK  // debugging: the last calls a thread made (a crash report prints them)
#define NCALLS 24
static __thread uint64_t t_calls[NCALLS][2];  // {target, return address}
static __thread unsigned t_ncalls;
#define TRACK_CALL(t, ra) do { t_calls[t_ncalls % NCALLS][0] = (t); t_calls[t_ncalls % NCALLS][1] = (ra); t_ncalls++; } while (0)
void interp_dump_calls(void) {
    extern const char *dispatch_hle_name(uint64_t pc);
    fprintf(stderr, "last guest calls (oldest first; target <- return address):\n");
    for (unsigned i = t_ncalls > NCALLS ? t_ncalls - NCALLS : 0; i < t_ncalls; i++) {
        uint64_t t = t_calls[i % NCALLS][0], ra = t_calls[i % NCALLS][1], off = 0, roff = 0;
        const char *h = dispatch_hle_name(t), *s = h ? h : loader_symbol_at(t, &off), *rs = loader_symbol_at(ra, &roff);
        fprintf(stderr, "  %llx %s+%llx  <- %llx %s+%llx\n", (unsigned long long)t, s ? s : "?", (unsigned long long)off,
                (unsigned long long)ra, rs ? rs : "?", (unsigned long long)roff);
    }
    // the first words of the last few call targets (to read them with an AArch64 disassembler)
    for (unsigned i = t_ncalls > 6 ? t_ncalls - 6 : 0; i < t_ncalls; i++) {
        uint64_t t = t_calls[i % NCALLS][0];
        if (t < 0x7000000000ull || t >= 0x7008000000ull) continue;
        fprintf(stderr, "  words@%llx:", (unsigned long long)t);
        for (int k = 0; k < 6; k++) fprintf(stderr, " %08x", *(const uint32_t *)(uintptr_t)(t + 4 * (uint64_t)k));
        fprintf(stderr, "\n");
    }
}
#else
#define TRACK_CALL(t, ra) ((void)0)
void interp_dump_calls(void) {}
#endif

static inline void call_pc(Cpu *c, uint64_t t) {
    IMod *m2;
    Op *o2 = lookup(t, &m2);
    if (o2) {
        run(c, m2, o2);
    } else {
        c->pc = t;
        rt_dispatch(c);
    }
}

// ---------------------------------------------------------------- small operations

static inline uint64_t sh64(uint64_t v, unsigned type, unsigned amt) {
    switch (type) {
    case 0: return v << amt;
    case 1: return v >> amt;
    case 2: return (uint64_t)((int64_t)v >> amt);
    default: return ROR64(v, amt);
    }
}
static inline uint32_t sh32(uint32_t v, unsigned type, unsigned amt) {
    amt &= 31;
    switch (type) {
    case 0: return v << amt;
    case 1: return v >> amt;
    case 2: return (uint32_t)((int32_t)v >> amt);
    default: return ROR32(v, amt);
    }
}
static inline uint64_t ext64(uint64_t v, unsigned option, unsigned sh) {
    uint64_t e;
    switch (option) {
    case 0: e = (uint8_t)v; break;
    case 1: e = (uint16_t)v; break;
    case 2: e = (uint32_t)v; break;
    case 4: e = (uint64_t)(int64_t)(int8_t)v; break;
    case 5: e = (uint64_t)(int64_t)(int16_t)v; break;
    case 6: e = (uint64_t)(int64_t)(int32_t)v; break;
    default: e = v; break;
    }
    return e << sh;
}

// ---------------------------------------------------------------- the loop

#ifdef INTERP_TEST
#define NEXT() do { c->pc = PCOF(op) + 4; return; } while (0)
#define TAKEN() do { c->pc = PCOF(op) + 4 * (uint64_t)op->imm; return; } while (0)
#define JUMP_PC(t) do { c->pc = (t); return; } while (0)
#define TRAP_UDF() do { g_step_fail = 1; return; } while (0)
#else
#ifdef INTERP_TRACK  // debugging: c->pc is always the instruction being run (a crash report names it)
#define DISPATCH() do { c->pc = PCOF(op); goto *labels[__atomic_load_n(&op->k, __ATOMIC_ACQUIRE)]; } while (0)
#else
#define DISPATCH() goto *labels[__atomic_load_n(&op->k, __ATOMIC_ACQUIRE)]
#endif
#define NEXT() do { ++op; DISPATCH(); } while (0)
#define TAKEN() do { op += op->imm; DISPATCH(); } while (0)
#define JUMP_PC(t) do {                                  \
        uint64_t jt_ = (t);                              \
        IMod *jm_;                                       \
        Op *jo_ = lookup(jt_, &jm_);                     \
        if (!jo_) { c->pc = jt_; rt_dispatch(c); return; } \
        m = jm_;                                         \
        op = jo_;                                        \
        DISPATCH();                                      \
    } while (0)
#define TRAP_UDF() rt_udf(c, PCOF(op))
#endif

#ifdef INTERP_TEST
#define CALL_NEAR() do { R(30) = PCOF(op) + 4; c->pc = PCOF(op) + 4 * (uint64_t)op->imm; return; } while (0)
#define CALL_FAR() do { R(30) = PCOF(op) + 4; c->pc = (uint64_t)op->imm; return; } while (0)
#define CALL_REG() do { uint64_t t_ = N; R(30) = PCOF(op) + 4; c->pc = t_; return; } while (0)
#define RETURN() do { c->pc = R(30); return; } while (0)
#else
#define CALL_NEAR() do { R(30) = PCOF(op) + 4; c->pc = PCOF(op + op->imm); TRACK_CALL(c->pc, R(30)); run(c, m, op + op->imm); NEXT(); } while (0)  // (c->pc: the callee, for crash reports)
#define CALL_FAR() do { R(30) = PCOF(op) + 4; c->pc = (uint64_t)op->imm; TRACK_CALL(c->pc, R(30)); rt_dispatch(c); NEXT(); } while (0)
#define CALL_REG() do { uint64_t t_ = N; R(30) = PCOF(op) + 4; c->pc = t_; TRACK_CALL(t_, R(30)); call_pc(c, t_); NEXT(); } while (0)
#define RETURN() return
#endif

// Decoding of the instruction at pc (first time it runs): host replacements first, then the word itself.
#ifndef INTERP_TEST
static const InterpPatch *patch_sorted;  // the selected game version's replacements, by address
static int patch_n;
static const InterpHook *hook_sorted;  // and its hook sites
static int hook_n;
static uint16_t *hook_k0;  // per hook site: the kind its own instruction decoded to (run after rt_hook says 0)
static int cmp_patch(const void *a, const void *b) {
    uint64_t x = ((const InterpPatch *)a)->addr, y = ((const InterpPatch *)b)->addr;
    return x < y ? -1 : x > y;
}
static int cmp_hook(const void *a, const void *b) {
    uint64_t x = ((const InterpHook *)a)->addr, y = ((const InterpHook *)b)->addr;
    return x < y ? -1 : x > y;
}
static const InterpPatch *find_patch(uint64_t addr) {
    int lo = 0, hi = patch_n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (patch_sorted[mid].addr < addr) lo = mid + 1;
        else hi = mid;
    }
    return lo < patch_n && patch_sorted[lo].addr == addr ? &patch_sorted[lo] : NULL;
}
static int find_hook(uint64_t addr) {
    int lo = 0, hi = hook_n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (hook_sorted[mid].addr < addr) lo = mid + 1;
        else hi = mid;
    }
    return lo < hook_n && hook_sorted[lo].addr == addr ? lo : -1;
}
static const InterpRandoSite *find_rando(uint64_t addr) {
    for (int i = 0; i < interp_rando_count; i++)
        if (interp_rando_sites[i].addr == addr) return &interp_rando_sites[i];
    return NULL;
}
static void decode_at(IMod *m, Op *op, uint64_t pc) {
    HostFn hook = dispatch_override_lookup(pc);
    if (hook) {
        op->imm = (int64_t)(intptr_t)hook;
        __atomic_store_n(&op->k, K_HOOK, __ATOMIC_RELEASE);
        return;
    }
    const InterpPatch *p = find_patch(pc);
    if (p) {
        op->imm = (int64_t)(intptr_t)p->fn;
        __atomic_store_n(&op->k, K_PATCH, __ATOMIC_RELEASE);
        return;
    }
    uint32_t w = *(const uint32_t *)(uintptr_t)pc;
    const InterpRandoSite *rs = find_rando(pc);
    if (rs) {  // a randomizer patch site: the active seed picks the instruction (0 = the original)
        uint32_t rw = rando_word(pc);
        for (int i = 0; i < rs->nwords; i++)
            if (rs->words[i] == rw) w = rw;
    }
    int hi = find_hook(pc);
    if (hi >= 0) {  // a hook site: the instruction decodes as it is, but rt_hook is asked first (L_HSITE)
        if (w != hook_sorted[hi].word) {
            static int warned;
            if (warned++ < 8)
                LOG("[interp] hook site %llx holds %08x, not %08x: the game files do not match this version's table\n",
                    (unsigned long long)pc, w, hook_sorted[hi].word);
            interp_decode(m, op, pc, w);
            return;
        }
        Op t;
        memset(&t, 0, sizeof t);
        interp_decode(m, &t, pc, w);  // (on a private copy: nobody runs the plain instruction meanwhile)
        hook_k0[hi] = t.k;
        op->d = t.d; op->n = t.n; op->m = t.m; op->a = t.a; op->x = t.x; op->y = t.y;
        op->imm = t.imm;
        __atomic_store_n(&op->k, K_HSITE, __ATOMIC_RELEASE);
        return;
    }
    interp_decode(m, op, pc, w);
}
#endif

static void run(Cpu *c, IMod *m, Op *op) {
#ifndef INTERP_TEST
#define X(n) [K_##n] = &&L_##n,
    static const void *const labels[K_COUNT] = {INTERP_OPS(X)};
#undef X
    DISPATCH();
#else
    switch (op->k) {
#define X(n) case K_##n: goto L_##n;
    INTERP_OPS(X)
#undef X
    }
#endif

L_UNDEF:
#ifndef INTERP_TEST
    decode_at(m, op, PCOF(op));
    DISPATCH();
#else
    return;
#endif
L_UDF: TRAP_UDF(); return;
L_NOP: NEXT();
L_HOOK:
#ifndef INTERP_TEST
    ((HostFn)(uintptr_t)op->imm)(c);
    return;
#else
    return;
#endif
L_PATCH: {
#ifndef INTERP_TEST
    uint64_t t = ((InterpPatchFn)(uintptr_t)op->imm)(c);
    if (!t) NEXT();
    if (t == 1) RETURN();
    JUMP_PC(t);
#else
    return;
#endif
}
L_HSITE: {
#ifndef INTERP_TEST
    int hi_ = find_hook(PCOF(op));
    int h = rt_hook(c, hook_sorted[hi_].addr);
    if (h == 0) goto *labels[hook_k0[hi_]];  // the site's own instruction (op still holds its operands)
    if (h == 1) NEXT();                      // the hook did it all
    JUMP_PC(hook_sorted[hi_].targets[h - 2]);
#else
    return;
#endif
}
L_SIMD:
    if (!interp_simd(c, (uint32_t)op->imm, PCOF(op))) TRAP_UDF();
    NEXT();

    // ---- immediates
L_MOVI: D = (uint64_t)op->imm; NEXT();
L_MOVK64: D = (D & ~(0xFFFFull << op->x)) | (uint64_t)op->imm; NEXT();
L_MOVK32: D = (uint32_t)((D & ~(0xFFFFull << op->x)) | (uint64_t)op->imm); NEXT();
L_ADDI64: D = N + (uint64_t)op->imm; NEXT();
L_ADDI32: D = (uint32_t)(N + (uint64_t)op->imm); NEXT();
L_ADDSI64: D = adds64(c, N, (uint64_t)op->imm); NEXT();
L_ADDSI32: D = adds32(c, N32, (uint32_t)op->imm); NEXT();
L_SUBSI64: D = subs64(c, N, (uint64_t)op->imm); NEXT();
L_SUBSI32: D = subs32(c, N32, (uint32_t)op->imm); NEXT();
L_ANDI64: D = N & (uint64_t)op->imm; NEXT();
L_ANDI32: D = (uint32_t)(N & (uint64_t)op->imm); NEXT();
L_ORRI64: D = N | (uint64_t)op->imm; NEXT();
L_ORRI32: D = (uint32_t)(N | (uint64_t)op->imm); NEXT();
L_EORI64: D = N ^ (uint64_t)op->imm; NEXT();
L_EORI32: D = (uint32_t)(N ^ (uint64_t)op->imm); NEXT();
L_ANDSI64: D = logic64(c, N & (uint64_t)op->imm); NEXT();
L_ANDSI32: D = logic32(c, N32 & (uint32_t)op->imm); NEXT();

    // ---- bitfield / extract
L_UBFM64: D = ROR64(N, op->x) & (uint64_t)op->imm; NEXT();
L_UBFM32: D = (uint32_t)(ROR32(N32, op->x) & (uint32_t)op->imm); NEXT();
L_SBFM64: {
    uint64_t src = N, rot = ROR64(src, op->x), tm = ones64(((op->y - op->x) & 63) + 1);
    uint64_t top = (uint64_t)(-(int64_t)((src >> op->y) & 1));
    D = (top & ~tm) | (rot & (uint64_t)op->imm);
    NEXT();
}
L_SBFM32: {
    uint32_t src = N32, rot = ROR32(src, op->x), tm = (uint32_t)ones64(((op->y - op->x) & 31) + 1);
    uint32_t top = (uint32_t)(-(int32_t)((src >> op->y) & 1));
    D = (uint32_t)((top & ~tm) | (rot & (uint32_t)op->imm));
    NEXT();
}
L_BFM64: {
    uint64_t src = N, dst = A, rot = ROR64(src, op->x), wm = (uint64_t)op->imm, tm = ones64(((op->y - op->x) & 63) + 1);
    uint64_t bot = (dst & ~wm) | (rot & wm);
    D = (dst & ~tm) | (bot & tm);
    NEXT();
}
L_BFM32: {
    uint32_t src = N32, dst = (uint32_t)A, rot = ROR32(src, op->x), wm = (uint32_t)op->imm,
             tm = (uint32_t)ones64(((op->y - op->x) & 31) + 1);
    uint32_t bot = (dst & ~wm) | (rot & wm);
    D = (uint32_t)((dst & ~tm) | (bot & tm));
    NEXT();
}
L_EXTR64: D = op->x ? (M >> op->x) | (N << (64 - op->x)) : M; NEXT();
L_EXTR32: D = (uint32_t)(op->x ? (M32 >> (op->x & 31)) | (N32 << ((32 - op->x) & 31)) : M32); NEXT();

    // ---- register operands
L_MOVR64: D = M; NEXT();
L_MOVR32: D = M32; NEXT();
L_ADDR64: D = N + sh64(M, op->x, op->y); NEXT();
L_ADDR32: D = (uint32_t)(N32 + sh32(M32, op->x, op->y)); NEXT();
L_SUBR64: D = N - sh64(M, op->x, op->y); NEXT();
L_SUBR32: D = (uint32_t)(N32 - sh32(M32, op->x, op->y)); NEXT();
L_ADDSR64: D = adds64(c, N, sh64(M, op->x, op->y)); NEXT();
L_ADDSR32: D = adds32(c, N32, sh32(M32, op->x, op->y)); NEXT();
L_SUBSR64: D = subs64(c, N, sh64(M, op->x, op->y)); NEXT();
L_SUBSR32: D = subs32(c, N32, sh32(M32, op->x, op->y)); NEXT();
L_ANDR64: D = N & sh64(M, op->x, op->y); NEXT();
L_ORRR64: D = N | sh64(M, op->x, op->y); NEXT();
L_EORR64: D = N ^ sh64(M, op->x, op->y); NEXT();
L_BICR64: D = N & ~sh64(M, op->x, op->y); NEXT();
L_ORNR64: D = N | ~sh64(M, op->x, op->y); NEXT();
L_EONR64: D = N ^ ~sh64(M, op->x, op->y); NEXT();
L_ANDSR64: D = logic64(c, N & sh64(M, op->x, op->y)); NEXT();
L_BICSR64: D = logic64(c, N & ~sh64(M, op->x, op->y)); NEXT();
L_ANDR32: D = (uint32_t)(N32 & sh32(M32, op->x, op->y)); NEXT();
L_ORRR32: D = (uint32_t)(N32 | sh32(M32, op->x, op->y)); NEXT();
L_EORR32: D = (uint32_t)(N32 ^ sh32(M32, op->x, op->y)); NEXT();
L_BICR32: D = (uint32_t)(N32 & ~sh32(M32, op->x, op->y)); NEXT();
L_ORNR32: D = (uint32_t)(N32 | ~sh32(M32, op->x, op->y)); NEXT();
L_EONR32: D = (uint32_t)(N32 ^ ~sh32(M32, op->x, op->y)); NEXT();
L_ANDSR32: D = logic32(c, N32 & sh32(M32, op->x, op->y)); NEXT();
L_BICSR32: D = logic32(c, N32 & ~sh32(M32, op->x, op->y)); NEXT();
L_ADDX64: D = N + ext64(M, op->x, op->y); NEXT();
L_ADDX32: D = (uint32_t)(N32 + (uint32_t)ext64(M, op->x, op->y)); NEXT();
L_SUBX64: D = N - ext64(M, op->x, op->y); NEXT();
L_SUBX32: D = (uint32_t)(N32 - (uint32_t)ext64(M, op->x, op->y)); NEXT();
L_ADDSX64: D = adds64(c, N, ext64(M, op->x, op->y)); NEXT();
L_ADDSX32: D = adds32(c, N32, (uint32_t)ext64(M, op->x, op->y)); NEXT();
L_SUBSX64: D = subs64(c, N, ext64(M, op->x, op->y)); NEXT();
L_SUBSX32: D = subs32(c, N32, (uint32_t)ext64(M, op->x, op->y)); NEXT();
L_ADC64: D = adc64(c, N, M); NEXT();
L_ADC32: D = adc32(c, N32, M32); NEXT();
L_SBC64: D = sbc64(c, N, M); NEXT();
L_SBC32: D = sbc32(c, N32, M32); NEXT();
L_ADCS64: D = adcs64(c, N, M); NEXT();
L_ADCS32: D = adcs32(c, N32, M32); NEXT();
L_SBCS64: D = sbcs64(c, N, M); NEXT();
L_SBCS32: D = sbcs32(c, N32, M32); NEXT();

    // conditional compare
#define CCMP(NAME, EXPR)                                                  \
    L_##NAME:                                                             \
    if (cond_holds(c->nzcv, op->x)) { (void)(EXPR); }                     \
    else c->nzcv = (uint64_t)op->y << 28;                                 \
    NEXT();
    CCMP(CCMNR64, adds64(c, N, M))
    CCMP(CCMNR32, adds32(c, N32, M32))
    CCMP(CCMPR64, subs64(c, N, M))
    CCMP(CCMPR32, subs32(c, N32, M32))
    CCMP(CCMNI64, adds64(c, N, (uint64_t)op->imm))
    CCMP(CCMNI32, adds32(c, N32, (uint32_t)op->imm))
    CCMP(CCMPI64, subs64(c, N, (uint64_t)op->imm))
    CCMP(CCMPI32, subs32(c, N32, (uint32_t)op->imm))
#undef CCMP

    // conditional select
L_CSEL64: D = cond_holds(c->nzcv, op->x) ? N : M; NEXT();
L_CSINC64: D = cond_holds(c->nzcv, op->x) ? N : M + 1; NEXT();
L_CSINV64: D = cond_holds(c->nzcv, op->x) ? N : ~M; NEXT();
L_CSNEG64: D = cond_holds(c->nzcv, op->x) ? N : -M; NEXT();
L_CSEL32: D = cond_holds(c->nzcv, op->x) ? N32 : M32; NEXT();
L_CSINC32: D = cond_holds(c->nzcv, op->x) ? N32 : (uint32_t)(M32 + 1); NEXT();
L_CSINV32: D = cond_holds(c->nzcv, op->x) ? N32 : (uint32_t)~M32; NEXT();
L_CSNEG32: D = cond_holds(c->nzcv, op->x) ? N32 : (uint32_t)-M32; NEXT();

    // multiply / divide / shifts by register
L_MADD64: D = A + N * M; NEXT();
L_MSUB64: D = A - N * M; NEXT();
L_MADD32: D = (uint32_t)((uint32_t)A + N32 * M32); NEXT();
L_MSUB32: D = (uint32_t)((uint32_t)A - N32 * M32); NEXT();
L_SMADDL: D = A + (uint64_t)((int64_t)(int32_t)N32 * (int64_t)(int32_t)M32); NEXT();
L_SMSUBL: D = A - (uint64_t)((int64_t)(int32_t)N32 * (int64_t)(int32_t)M32); NEXT();
L_UMADDL: D = A + (uint64_t)N32 * (uint64_t)M32; NEXT();
L_UMSUBL: D = A - (uint64_t)N32 * (uint64_t)M32; NEXT();
L_SMULH: D = (uint64_t)(((__int128)(int64_t)N * (__int128)(int64_t)M) >> 64); NEXT();
L_UMULH: D = (uint64_t)(((unsigned __int128)N * M) >> 64); NEXT();
L_UDIV64: D = udiv64(N, M); NEXT();
L_UDIV32: D = udiv32(N32, M32); NEXT();
L_SDIV64: D = sdiv64(N, M); NEXT();
L_SDIV32: D = sdiv32(N32, M32); NEXT();
L_LSLV64: D = N << (M & 63); NEXT();
L_LSLV32: D = (uint32_t)(N32 << (M & 31)); NEXT();
L_LSRV64: D = N >> (M & 63); NEXT();
L_LSRV32: D = (uint32_t)(N32 >> (M & 31)); NEXT();
L_ASRV64: D = (uint64_t)((int64_t)N >> (M & 63)); NEXT();
L_ASRV32: D = (uint32_t)((int32_t)N32 >> (M & 31)); NEXT();
L_RORV64: D = ROR64(N, M & 63); NEXT();
L_RORV32: D = ROR32(N32, M & 31); NEXT();

    // one source
L_RBIT64: D = rbit64(N); NEXT();
L_RBIT32: D = rbit32(N32); NEXT();
L_REV16_64: D = rev16_64(N); NEXT();
L_REV16_32: D = rev16_32(N32); NEXT();
L_REV32_64: D = rev32_64(N); NEXT();
L_REV32: D = __builtin_bswap32(N32); NEXT();
L_REV64: D = __builtin_bswap64(N); NEXT();
L_CLZ64: D = clz64(N); NEXT();
L_CLZ32: D = clz32(N32); NEXT();
L_CLS64: D = cls64(N); NEXT();
L_CLS32: D = cls32(N32); NEXT();
L_CRC: D = interp_crc(op->x, N32, op->y == 3 ? M : (uint64_t)M32, 1 << op->y); NEXT();

    // ---- branches
L_B_NEAR: TAKEN();
L_B_FAR: JUMP_PC((uint64_t)op->imm);
L_BL_NEAR: CALL_NEAR();
L_BL_FAR: CALL_FAR();
L_BCC: if (cond_holds(c->nzcv, op->x)) TAKEN(); NEXT();
L_CBZ64: if (N == 0) TAKEN(); NEXT();
L_CBNZ64: if (N != 0) TAKEN(); NEXT();
L_CBZ32: if (N32 == 0) TAKEN(); NEXT();
L_CBNZ32: if (N32 != 0) TAKEN(); NEXT();
L_TBZ: if (!((N >> op->x) & 1)) TAKEN(); NEXT();
L_TBNZ: if ((N >> op->x) & 1) TAKEN(); NEXT();
L_BR: JUMP_PC(N);
L_BLR: CALL_REG();
L_RET: RETURN();
L_SVC: rt_svc(c, (uint32_t)op->imm, PCOF(op)); NEXT();
L_BRK:
#ifdef INTERP_TEST
    g_step_fail = 1;
    return;
#else
    rt_brk(c, (uint32_t)op->imm, PCOF(op));
#endif

    // ---- system registers
L_MRS_CONST: D = (uint64_t)op->imm; NEXT();
L_MRS_TPIDR: D = c->tpidr; NEXT();
L_MRS_TPIDRRO: D = c->tpidrro; NEXT();
L_MRS_TICKS: D = rt_ticks(); NEXT();
L_MRS_NZCV: D = c->nzcv; NEXT();
L_MRS_FPCR: D = c->fpcr; NEXT();
L_MRS_FPSR: D = c->fpsr; NEXT();
L_MSR_TPIDR: c->tpidr = N; NEXT();
L_MSR_NZCV: c->nzcv = N & 0xf0000000ull; NEXT();
L_MSR_FPCR: c->fpcr = N & 0x7c00000ull; NEXT();
L_MSR_FPSR: c->fpsr = N & 0xf800009full; NEXT();
L_DCZVA: memset((void *)(uintptr_t)(N & ~63ull), 0, 64); NEXT();
L_FENCE: __atomic_thread_fence(__ATOMIC_SEQ_CST); NEXT();
L_CLREX: c->excl_addr = ~0ull; NEXT();

    // ---- loads and stores
L_LDRW_ABS: D = LD32((uint64_t)op->imm); NEXT();
L_LDRX_ABS: D = LD64((uint64_t)op->imm); NEXT();
L_LDRSW_ABS: D = (uint64_t)(int64_t)(int32_t)LD32((uint64_t)op->imm); NEXT();

L_LDXR: {
    uint64_t a = N, v;
    switch (op->y) {
    case 0: v = ldx8(c, a); break;
    case 1: v = ldx16(c, a); break;
    case 2: v = ldx32(c, a); break;
    default: v = ldx64(c, a); break;
    }
    D = v;
    NEXT();
}
L_STXR: {
    uint64_t a = N, v = D;
    uint32_t st;
    switch (op->y) {
    case 0: st = stx8(c, a, v); break;
    case 1: st = stx16(c, a, v); break;
    case 2: st = stx32(c, a, v); break;
    default: st = stx64(c, a, v); break;
    }
    M = st;
    NEXT();
}
L_LDAR: {
    uint64_t a = N, v;
    switch (op->y) {
    case 0: v = lda8(a); break;
    case 1: v = lda16(a); break;
    case 2: v = lda32(a); break;
    default: v = lda64(a); break;
    }
    D = v;
    NEXT();
}
L_STLR: {
    uint64_t a = N, v = D;
    switch (op->y) {
    case 0: stl8(a, v); break;
    case 1: stl16(a, v); break;
    case 2: stl32(a, v); break;
    default: stl64(a, v); break;
    }
    NEXT();
}

#define EA_O uint64_t ea = N + (uint64_t)op->imm
#define EA_PRE uint64_t ea = N + (uint64_t)op->imm
#define EA_POST uint64_t ea = N
#define EA_REG uint64_t ea = N + ext64(M, op->x, op->y)
#define WB_O (void)0
#define WB_PRE N = ea
#define WB_POST N = ea + (uint64_t)op->imm
#define WB_REG (void)0

#define LDOP(NAME, VAL)                                         \
    L_##NAME##_O: { EA_O; uint64_t v = (VAL); WB_O; D = v; NEXT(); }     \
    L_##NAME##_PRE: { EA_PRE; uint64_t v = (VAL); WB_PRE; D = v; NEXT(); } \
    L_##NAME##_POST: { EA_POST; uint64_t v = (VAL); WB_POST; D = v; NEXT(); } \
    L_##NAME##_REG: { EA_REG; uint64_t v = (VAL); WB_REG; D = v; NEXT(); }
#define STOP(NAME, STORE)                                       \
    L_##NAME##_O: { EA_O; STORE; WB_O; NEXT(); }                \
    L_##NAME##_PRE: { EA_PRE; STORE; WB_PRE; NEXT(); }          \
    L_##NAME##_POST: { EA_POST; STORE; WB_POST; NEXT(); }       \
    L_##NAME##_REG: { EA_REG; STORE; WB_REG; NEXT(); }

    LDOP(LDRB, LD8(ea))
    LDOP(LDRH, LD16(ea))
    LDOP(LDRW, LD32(ea))
    LDOP(LDRX, LD64(ea))
    LDOP(LDRSB64, (uint64_t)(int64_t)(int8_t)LD8(ea))
    LDOP(LDRSH64, (uint64_t)(int64_t)(int16_t)LD16(ea))
    LDOP(LDRSW64, (uint64_t)(int64_t)(int32_t)LD32(ea))
    LDOP(LDRSB32, (uint32_t)(int32_t)(int8_t)LD8(ea))
    LDOP(LDRSH32, (uint32_t)(int32_t)(int16_t)LD16(ea))
    STOP(STRB, ST8(ea, D))
    STOP(STRH, ST16(ea, D))
    STOP(STRW, ST32(ea, D))
    STOP(STRX, ST64(ea, D))

#define PAIR_LD(NAME, A_, B_, SZ)                                                                    \
    L_##NAME##_O: { EA_O; uint64_t a = (A_), b = (B_); WB_O; D = a; M = b; NEXT(); }                  \
    L_##NAME##_PRE: { EA_PRE; uint64_t a = (A_), b = (B_); WB_PRE; D = a; M = b; NEXT(); }            \
    L_##NAME##_POST: { EA_POST; uint64_t a = (A_), b = (B_); WB_POST; D = a; M = b; NEXT(); }
#define PAIR_ST(NAME, STORE)                                            \
    L_##NAME##_O: { EA_O; STORE; WB_O; NEXT(); }                        \
    L_##NAME##_PRE: { EA_PRE; STORE; WB_PRE; NEXT(); }                  \
    L_##NAME##_POST: { EA_POST; STORE; WB_POST; NEXT(); }
    PAIR_LD(LDPW, LD32(ea), LD32(ea + 4), 4)
    PAIR_LD(LDPX, LD64(ea), LD64(ea + 8), 8)
    PAIR_LD(LDPSW, (uint64_t)(int64_t)(int32_t)LD32(ea), (uint64_t)(int64_t)(int32_t)LD32(ea + 4), 4)
    PAIR_ST(STPW, (ST32(ea, D), ST32(ea + 4, M)))
    PAIR_ST(STPX, (ST64(ea, D), ST64(ea + 8, M)))
}

// ---------------------------------------------------------------- setup

#ifndef INTERP_TEST
void interp_init(void) {
    if (!g_nmods) {  // (first call; a reboot keeps the same game)
        const GameVersion *gv = game_version();
        const InterpVersion *iv = NULL;
        for (int i = 0; gv && i < interp_version_count; i++)
            if (!strcmp(interp_versions[i].key, gv->key)) iv = &interp_versions[i];
        if (!iv) FATAL("interp: no patch table for this game version (%s)\n", gv ? gv->key : "unknown");
        patch_n = iv->npatches;
        patch_sorted = malloc(sizeof(InterpPatch) * (size_t)(patch_n + 1));
        memcpy((void *)patch_sorted, iv->patches, sizeof(InterpPatch) * (size_t)patch_n);
        qsort((void *)patch_sorted, (size_t)patch_n, sizeof(InterpPatch), cmp_patch);
        hook_n = iv->nhooks;
        hook_sorted = malloc(sizeof(InterpHook) * (size_t)(hook_n + 1));
        memcpy((void *)hook_sorted, iv->hooks, sizeof(InterpHook) * (size_t)hook_n);
        qsort((void *)hook_sorted, (size_t)hook_n, sizeof(InterpHook), cmp_hook);
        hook_k0 = calloc((size_t)hook_n + 1, sizeof *hook_k0);
        LOG("[interp] game version %s: %d replacements, %d hook sites\n", iv->key, patch_n, hook_n);
    }
    for (int i = 0; i < g_nmodules; i++) {
        Module *mod = &g_modules[i];
        uint64_t lo = mod->base + mod->text_off, hi = lo + mod->text_size;
        int j;
        for (j = 0; j < g_nmods; j++)
            if (g_mods[j].lo == lo && g_mods[j].hi == hi) break;
        if (j < g_nmods) {  // a reboot reloads the same text: decode it afresh (seed patches may have changed it)
            size_t n = (size_t)((hi - lo) / 4) * sizeof(Op);
            if (mmap(g_mods[j].ops, n, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE | MAP_FIXED, -1, 0) == MAP_FAILED)
                FATAL("interp: cannot reset the decoded code\n");
            continue;
        }
        if (g_nmods == 4) FATAL("interp: too many modules\n");
        size_t n = (size_t)((hi - lo) / 4);
        void *ops = mmap(NULL, n * sizeof(Op), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
        if (ops == MAP_FAILED) FATAL("interp: no memory for the decoded code of %s\n", mod->name);
        g_mods[g_nmods++] = (IMod){lo, hi, ops};
        LOG("[interp] %s text %llx-%llx\n", mod->name, (unsigned long long)lo, (unsigned long long)hi);
    }
    g_active = g_nmods > 0;
}

bool interp_active(void) { return g_active; }

void interp_enter(Cpu *c) {
    IMod *m;
    Op *op = lookup(c->pc, &m);
    if (!op) {
        uint64_t off;
        const char *s = loader_symbol_at(c->x[30], &off);
        FATAL("no function at %llx (lr=%llx %s+%llx)\n", (unsigned long long)c->pc, (unsigned long long)c->x[30],
              s ? s : "?", (unsigned long long)off);
    }
    c->x[31] = 0;
    run(c, m, op);
}

static void forget(uint64_t addr) {
    IMod *m;
    Op *op = lookup(addr, &m);
    if (op) __atomic_store_n(&op->k, K_UNDEF, __ATOMIC_RELEASE);
}
void interp_invalidate(uint64_t addr) { forget(addr); }
void interp_reset_rando(void) {
    for (int i = 0; i < interp_rando_count; i++) forget(interp_rando_sites[i].addr);
}
#else
bool interp_step_word(Cpu *c, uint32_t w, uint64_t pc) {
    IMod dm = {0, ~0ull, NULL};
    Op op;
    memset(&op, 0, sizeof op);
    interp_decode(&dm, &op, pc, w);
    if (op.k == K_UDF) return false;
    IMod rm = {pc, pc + 4, &op};
    g_step_fail = 0;
    c->x[31] = 0;
    run(c, &rm, &op);
    return !g_step_fail;
}
#endif
