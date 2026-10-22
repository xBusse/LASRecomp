# LAS Recomp on iOS (sideloaded)

**Status: untested.** Written and reviewed off-device; nothing here has been compiled for iOS or run on an iPhone yet.
Expect to fix compile errors on first build (SDL hint names in `runtime/nvn/metal.m` are the likeliest).

## How it differs from desktop
- No JIT, no spawning processes, no loading locally built code. So the game code (`lasgame.dylib`) is **built on a Mac and
  bundled in the app**; the on-device launcher only imports your game files. Building the game code on the phone
  itself is not possible without an interpreter tier (not implemented).
- Touch controls (`runtime/ui/touch_pad.mm`): left stick, A/B/X/Y, L/R, +/-, and a menu button.
- Data lives in the app's Documents folder (visible in Files > LAS Recomp).

## Build (on a Mac with Xcode)
1. `make ios-deps`        # static SDL3 + FFmpeg for arm64 iOS -> build/ios-deps
2. put your extracted game (exefs/romfs) where the desktop build expects it, then
   `make ios-gen`        # generates the C and builds build/ios/lasgame.dylib
3. `make ios`            # builds the app and `build/ios/LAS Recomp.ipa`
   (override with `IOS_MIN=16.0`, `BUNDLE_ID=...`)

## Install
Sideload the IPA with AltStore / SideStore / Sideloadly. The IPA is only ad-hoc signed to carry
`tools/ios/entitlements.plist`; your sideloader re-signs it.

The game reserves a ~12.5 GB virtual window, which needs the **extended-virtual-addressing** and
**increased-memory-limit** entitlements. A free Apple ID may not grant them; if the app shows the "could not reserve
guest memory" box, that is why (a paid developer account, or a shifted/smaller window, is the fix).
Needs a device with 6 GB+ RAM.

## First run
Copy `exefs` and `romfs` (and optionally `fonts/`) into Files > LAS Recomp > Import, then open the app.

## Not done
- Unicorn removal from `tools/gen.py` (only matters if gen is to run somewhere without a JIT).
- Interpreter tier for on-device code generation: realistic speed on a phone is likely 10-20 fps at best.
- Nintendo shared fonts: falls back to the bundled Baloo2 (Latin only) unless you provide `fonts/`.
