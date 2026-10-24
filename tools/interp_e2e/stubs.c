// No game, no replacements: the e2e test runs plain code.
#include "game_addresses.h"
#include "interp.h"
#include "rt.h"
static const InterpPatch no_patches[] = {{0, 0}};
InterpHook stub_hooks[2];  // (run.c fills one in before interp_init: the hook-site test)
const InterpVersion interp_versions[] = {{"stub", no_patches, 0, stub_hooks, 2}};
const int interp_version_count = 1;
const InterpRandoSite interp_rando_sites[] = {{0, 0, 0}};
const int interp_rando_count = 0;
static const GameVersion stub_version = {"stub", "stub", "", 0};
const GameVersion *game_version(void) { return &stub_version; }
HostFn dispatch_override_lookup(uint64_t addr) { (void)addr; return 0; }
