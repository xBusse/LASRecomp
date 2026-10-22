// No game, no replacements: the e2e test runs plain code.
#include "interp.h"
#include "rt.h"
const InterpPatch interp_patches[] = {{0, 0}};
const int interp_patch_count = 0;
const InterpRandoSite interp_rando_sites[] = {{0, 0, 0}};
const int interp_rando_count = 0;
HostFn dispatch_override_lookup(uint64_t addr) { (void)addr; return 0; }
