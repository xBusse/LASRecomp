// The interpreter tier: runs the guest's AArch64 code directly, without the recompiled game library.
//
// Needed where nothing can be built or loaded at run time (iOS: no JIT, no loading of code made on the device), and
// useful anywhere as a fallback: the game files alone are enough, no `make gen`. It reads the instructions out of the
// loaded modules' text (guest memory), decodes each one the first time it runs into a compact `Op`, and runs those.
// It does the same as the code tools/gen.py makes (tools/lifter.py, tools/simd_c.py, tools/patches.py): the same
// instruction subset, the same portable FP/SIMD semantics (runtime/simd.h), the same per-instruction replacements
// (runtime/interp_patches.c). tools/test_interp.py checks it against the lifter, instruction by instruction.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the loaded modules' text (call after loader_load; again after a reboot reloaded them).
void interp_init(void);
bool interp_active(void);

// GuestFn: runs the guest function at c->pc, returning when it returns.
void interp_enter(Cpu *c);

// A host replacement for a guest function (hle_override) or an instruction (interp_patches.c) was added or changed at
// addr: forget what was decoded there.
void interp_invalidate(uint64_t addr);
// The randomizer seed changed: forget the instructions at its patch sites.
void interp_reset_rando(void);

// Replacement of one instruction: runs instead of it; returns 0 to go on with the next one, or the guest address to
// continue at.
typedef uint64_t (*InterpPatchFn)(Cpu *c);
typedef struct { uint64_t addr; InterpPatchFn fn; } InterpPatch;
extern const InterpPatch interp_patches[];
extern const int interp_patch_count;
// Instructions the randomizer seed may replace (tools/rando/patch_variants.json): the address, and the words it can use.
typedef struct { uint64_t addr; const uint32_t *words; int nwords; } InterpRandoSite;
extern const InterpRandoSite interp_rando_sites[];
extern const int interp_rando_count;

// Single-step interface, for the tests (tools/test_interp.py): runs the one instruction w at pc against c. Returns
// false for an encoding the lifter rejects too (the generated code traps on it). A branch leaves its target in c->pc
// (and, for calls, the return address in x30); other instructions leave c->pc alone.
bool interp_step_word(Cpu *c, uint32_t w, uint64_t pc);

// Implemented in interp_simd.c: FP / AdvSIMD / crypto and their loads and stores. false: encoding not supported.
bool interp_simd(Cpu *c, uint32_t w, uint64_t pc);

#ifdef __cplusplus
}
#endif
