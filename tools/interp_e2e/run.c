// Runs the functions of prog.c (linked for AArch64 at E2E_BASE) in the interpreter. Built by tools/test_interp_e2e.sh.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "cpu.h"
#include "interp.h"
#include "rt.h"

Module g_modules[4];
int g_nmodules;
void rt_dispatch(Cpu *c) { interp_enter(c); }
void rt_svc(Cpu *c, uint32_t imm, uint64_t pc) { (void)c; fprintf(stderr, "svc #%u at %llx\n", imm, (unsigned long long)pc); exit(3); }
void rt_brk(Cpu *c, uint32_t imm, uint64_t pc) { (void)c; fprintf(stderr, "brk #%u at %llx\n", imm, (unsigned long long)pc); exit(3); }
void rt_udf(Cpu *c, uint64_t pc) { (void)c; fprintf(stderr, "undefined instruction at %llx\n", (unsigned long long)pc); exit(3); }
const char *loader_symbol_at(uint64_t a, uint64_t *off) { (void)a; if (off) *off = 0; return "?"; }
uint64_t rt_ticks(void) { return 0; }
volatile int g_fnprof;
void fnprof_hit(uint64_t a, uint64_t b, uint64_t c) { (void)a; (void)b; (void)c; }
uint32_t rando_word(uint64_t a) { (void)a; return 0; }
void rt_fatal_trace(void) {}
void rt_fatal_msg(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap); }
// hook-site test (hk_fn in prog.c): what rt_hook answers, and what it saw
extern InterpHook stub_hooks[2];
static int g_hk_ret;
static uint64_t g_hk_site, g_hk_x0;
int rt_hook(Cpu *c, uint64_t site) { g_hk_site = site; g_hk_x0 = c->x[0]; return g_hk_ret; }
const char *rt_log_prefix(void) { return ""; }

// usage: run <image.bin> <base> <text_off> <text_size> <name=addr>...
int main(int argc, char **argv) {
    if (argc < 6) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    uint64_t base = strtoull(argv[2], 0, 16), toff = strtoull(argv[3], 0, 16), tsz = strtoull(argv[4], 0, 16);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t map = ((size_t)n + 0x200000) & ~0xFFFFull;
    void *img = mmap((void *)base, map, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANON | MAP_PRIVATE | MAP_FIXED, -1, 0);
    if (img == MAP_FAILED || fread(img, 1, (size_t)n, f) != (size_t)n) return 2;
    g_modules[0] = (Module){"e2e", base, map, toff, tsz, 0};
    g_nmodules = 1;
    static uint64_t hk_targets[1];
    for (int i = 5; i < argc; i++)  // hk_fn: the hook site is its second instruction, the target its `ret`
        if (!strncmp(argv[i], "hk_fn=", 6)) {
            uint64_t a = strtoull(argv[i] + 6, 0, 16);
            hk_targets[0] = a + 12;
            stub_hooks[0] = (InterpHook){a + 4, 0x91002800 /* add x0, x0, #10 */, hk_targets, 1};
        }
    interp_init();
    static char stack[1 << 20];
    int fails = 0, ran = 0;
    for (int i = 5; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (!eq) continue;
        *eq = 0;
        uint64_t addr = strtoull(eq + 1, 0, 16);
        if (!strcmp(argv[i], "hk_fn")) {  // rt_hook 0: the instruction runs; 1: it does not; 2: jump to the target
            static const int ret[3] = {0, 1, 2};
            static const uint64_t want[3] = {111, 101, 1};
            for (int r = 0; r < 3; r++) {
                g_hk_ret = ret[r];
                Cpu *h = calloc(1, sizeof(Cpu));
                h->sp = ((uint64_t)(uintptr_t)stack + sizeof stack - 64) & ~15ull;
                h->pc = addr;
                interp_enter(h);
                int ok = h->x[0] == want[r] && g_hk_site == addr + 4 && g_hk_x0 == 1;
                printf("%-18s hook returns %d: %s (x0=%llu, site %llx saw x0=%llu)\n", "hk_fn", r, ok ? "ok" : "FAIL",
                       (unsigned long long)h->x[0], (unsigned long long)g_hk_site, (unsigned long long)g_hk_x0);
                ran++;
                fails += !ok;
                free(h);
            }
            continue;
        }
        Cpu *c = calloc(1, sizeof(Cpu));
        c->sp = ((uint64_t)(uintptr_t)stack + sizeof stack - 64) & ~15ull;
        c->pc = addr;
        c->x[30] = 0;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        interp_enter(c);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        ran++;
        printf("%-18s %s (x0=%llu)  %.1f ms\n", argv[i], c->x[0] ? "FAIL" : "ok", (unsigned long long)c->x[0], ms);
        fails += c->x[0] != 0;
        free(c);
    }
    printf("%d functions, %d failed\n", ran, fails);
    return fails != 0;
}
