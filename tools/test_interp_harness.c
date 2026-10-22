// Runs the lifted one-instruction functions (oracle_*.c) and the interpreter on the same random state and compares.
// Built and driven by tools/test_interp.py.
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cpu.h"
#include "interp.h"

extern GuestFn *const test_chunks[];
extern const int test_chunk_size, ntests, nbad;
extern const unsigned test_words[], bad_words[];
extern const unsigned long long test_pcs[], bad_pc;
extern const unsigned char test_kinds[];

#define BUF 0x7200000000ull
#define BUF_SIZE 0x30000
#define BUF_MID (BUF + 0x10000)

void rt_dispatch(Cpu *c) { (void)c; }
void rt_svc(Cpu *c, uint32_t imm, uint64_t pc) { (void)c; (void)imm; (void)pc; }
void rt_brk(Cpu *c, uint32_t imm, uint64_t pc) { (void)c; (void)imm; (void)pc; abort(); }
void rt_udf(Cpu *c, uint64_t pc) { (void)c; (void)pc; abort(); }
uint64_t rt_ticks(void) { return 0; }
volatile int g_fnprof;
void fnprof_hit(uint64_t a, uint64_t b, uint64_t c) { (void)a; (void)b; (void)c; }

static sigjmp_buf jb;
static volatile int in_run;
static void on_segv(int s) { (void)s; if (in_run) siglongjmp(jb, 1); _exit(99); }

static uint64_t rs;
static uint64_t rnd(void) {
    rs ^= rs >> 12; rs ^= rs << 25; rs ^= rs >> 27;
    return rs * 0x2545F4914F6CDD1Dull;
}
static const uint64_t SPECIAL[] = {0, 1, 2, ~0ull, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull, 0xFFFFFFFFull, 0x7FFFFFFFull,
                                   0x80000000ull, 0x100000000ull, 0x3F800000ull, 0x3FF0000000000000ull, 0xFFull, 0xFFFFull,
                                   63, 64, 31, 32};
static uint64_t rand64(void) {
    unsigned k = (unsigned)(rnd() % 100);
    if (k < 30) return SPECIAL[rnd() % (sizeof SPECIAL / sizeof *SPECIAL)];
    if (k < 50) return rnd() & 0xFF;
    if (k < 60) {
        float f = (float)((double)(rnd() % 2000000) / 1000.0 - 1000.0);
        uint32_t u;
        memcpy(&u, &f, 4);
        return u | (rnd() << 32);
    }
    return rnd();
}
static __uint128_t rand_v(void) {
    unsigned k = (unsigned)(rnd() % 100);
    if (k < 50) {
        float fs[4];
        for (int i = 0; i < 4; i++) fs[i] = (float)((double)(rnd() % 200000) / 1000.0 - 100.0);
        if (rnd() % 10 == 0) {
            static const float sp[] = {0.0f, -0.0f, __builtin_inff(), __builtin_nanf(""), 1e-40f};
            fs[rnd() % 4] = sp[rnd() % 5];
        }
        __uint128_t v;
        memcpy(&v, fs, 16);
        return v;
    }
    if (k < 70) {
        double ds[2];
        for (int i = 0; i < 2; i++) ds[i] = (double)(rnd() % 2000000000) / 1000.0 - 1e6;
        __uint128_t v;
        memcpy(&v, ds, 16);
        return v;
    }
    return ((__uint128_t)rnd() << 64) | rnd();
}

static int is_mem(unsigned w) { return !((w >> 25) & 1) && ((w >> 27) & 1); }

static void make_state(Cpu *c, unsigned w) {
    memset(c, 0, sizeof *c);
    int mem = is_mem(w) || (w & 0xFFFFFFE0u) == 0xD50B7420u;
    for (int i = 0; i < 31; i++) c->x[i] = mem ? BUF_MID + (rnd() % 0x100 & ~7ull) : rand64();
    c->sp = BUF_MID + (rnd() % 0x100 & ~15ull);
    if (mem) {
        unsigned rm = (w >> 16) & 31, rn = (w >> 5) & 31;
        if (((w >> 21) & 1 && ((w >> 10) & 3) == 2) || (w & 0xBFA00000u) == 0x0C800000u) {
            if (rm != rn && rm < 31) c->x[rm] = rnd() % 0x100;
        }
    }
    c->nzcv = (uint64_t)(rnd() & 15) << 28;
    for (int i = 0; i < 32; i++) c->v[i] = rand_v();
    c->excl_addr = ~0ull;
    // FPCR: mostly default, sometimes DN / FZ / rounding modes
    unsigned f = (unsigned)(rnd() % 8);
    c->fpcr = f < 5 ? 0 : (uint64_t)(rnd() & 0x7) << 22;
}

static const char *diff(const Cpu *a, const Cpu *b, int cmp_pc) {
    static char msg[128];
    for (int i = 0; i < 31; i++)
        if (a->x[i] != b->x[i]) { snprintf(msg, sizeof msg, "x%d %llx vs %llx", i, (unsigned long long)a->x[i], (unsigned long long)b->x[i]); return msg; }
    if (a->sp != b->sp) return "sp";
    if (a->nzcv != b->nzcv) { snprintf(msg, sizeof msg, "nzcv %llx vs %llx", (unsigned long long)a->nzcv, (unsigned long long)b->nzcv); return msg; }
    if (a->fpcr != b->fpcr) return "fpcr";
    if (a->fpsr != b->fpsr) return "fpsr";
    if (a->excl_addr != b->excl_addr) return "excl_addr";
    for (int i = 0; i < 32; i++)
        if (a->v[i] != b->v[i]) {
            snprintf(msg, sizeof msg, "v%d %016llx%016llx vs %016llx%016llx", i, (unsigned long long)(a->v[i] >> 64),
                     (unsigned long long)a->v[i], (unsigned long long)(b->v[i] >> 64), (unsigned long long)b->v[i]);
            return msg;
        }
    if (cmp_pc && a->pc != b->pc) { snprintf(msg, sizeof msg, "pc %llx vs %llx", (unsigned long long)a->pc, (unsigned long long)b->pc); return msg; }
    return NULL;
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 3;
    rs = (argc > 2 ? strtoull(argv[2], 0, 10) : 1) * 0x9E3779B97F4A7C15ull | 1;
    void *p = mmap((void *)BUF, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
    if (p != (void *)BUF) { perror("mmap"); return 1; }
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_segv;
    sa.sa_flags = SA_NODEFER;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    uint8_t *pristine = malloc(BUF_SIZE), *snap = malloc(BUF_SIZE);
    for (int i = 0; i < BUF_SIZE; i++) pristine[i] = (uint8_t)rnd();
    static Cpu c0, c1, c2;
    long fails = 0, crashes = 0, runs = 0;
    for (int k = 0; k < ntests; k++) {
        unsigned w = test_words[k];
        uint64_t pc = test_pcs[k];
        GuestFn f = test_chunks[k / test_chunk_size][k % test_chunk_size];
        for (int rep = 0; rep < reps; rep++) {
            make_state(&c0, w);
            // oracle
            memcpy((void *)BUF, pristine, BUF_SIZE);
            c1 = c0;
            in_run = 1;
            if (sigsetjmp(jb, 1)) { in_run = 0; crashes++; printf("CRASH(oracle) %08x\n", w); break; }
            f(&c1);
            in_run = 0;
            memcpy(snap, (void *)BUF, BUF_SIZE);
            // interpreter
            memcpy((void *)BUF, pristine, BUF_SIZE);
            c2 = c0;
            in_run = 1;
            if (sigsetjmp(jb, 1)) { in_run = 0; crashes++; printf("CRASH(interp) %08x\n", w); break; }
            bool ok = interp_step_word(&c2, w, pc);
            in_run = 0;
            runs++;
            const char *d = NULL;
            if (!ok) d = "interpreter rejects a word the lifter accepts";
            else d = diff(&c1, &c2, test_kinds[k]);
            if (!d && memcmp(snap, (void *)BUF, BUF_SIZE)) d = "memory";
            if (d) {
                fails++;
                if (fails <= 400) printf("MISMATCH %08x (rep %d): %s\n", w, rep, d);
                break;
            }
        }
    }
    long accepted = 0;
    for (int k = 0; k < nbad; k++) {
        unsigned w = bad_words[k];
        make_state(&c0, w);
        memcpy((void *)BUF, pristine, BUF_SIZE);
        c2 = c0;
        in_run = 1;
        if (sigsetjmp(jb, 1)) { in_run = 0; accepted++; if (accepted <= 40) printf("REJECT-MISMATCH %08x: interpreter faults on a word the lifter rejects\n", w); continue; }
        bool ok = interp_step_word(&c2, w, bad_pc);
        in_run = 0;
        if (ok) { accepted++; if (accepted <= 40) printf("REJECT-MISMATCH %08x: interpreter runs a word the lifter rejects\n", w); }
    }
    printf("%d words, %ld runs, %ld mismatches, %ld crashes; %d lifter-rejected words, %ld accepted by the interpreter\n",
           ntests, runs, fails, crashes, nbad, accepted);
    return fails || crashes || accepted ? 1 : 0;
}
