// Runs the functions of prog.c (linked for AArch64 at E2E_BASE) in the interpreter. Built by tools/test_interp_e2e.sh.
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
    interp_init();
    static char stack[1 << 20];
    int fails = 0, ran = 0;
    for (int i = 5; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (!eq) continue;
        *eq = 0;
        uint64_t addr = strtoull(eq + 1, 0, 16);
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
