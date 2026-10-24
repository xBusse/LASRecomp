#!/bin/bash
# tools/ios/bundle_ios.sh <exe> public|private: build/ios/LAS Recomp.app and build/ios/LASRecomp-<mode>.ipa from <exe>
# (make ios-public / ios-private).
#   public   never any game code: a build/ios/lasgame.dylib left by an earlier `make ios-gen` is ignored, and the finished
#            .ipa is checked to hold none. The app runs the game's code in its interpreter from the player's own files.
#   private  with build/ios/lasgame.dylib (make ios-gen): game code compiled from YOUR game files. For your own phone only.
#   LASRecomp.app/LASRecomp                the executable (static SDL3 and FFmpeg inside: build/ios-deps)
#   LASRecomp.app/Frameworks/lasgame.dylib the game code, private mode only (the app can't build it: an iOS app can't run a
#                                          compiler or load code it made itself, docs/ios.md)
#   LASRecomp.app/*.png, Baloo2*           icons, the start screen's font (OFL)
# No game files, and game code only in private mode. The .ipa is ad-hoc signed with tools/ios/entitlements.plist
# only so the entitlements travel with it: AltStore / SideStore / Sideloadly sign it again with the player's own
# certificate when they install it.
set -euo pipefail
cd "$(dirname "$0")/../.."
EXE=${1:-build/ios/lasrecomp}
MODE=${2:-public}
case "$MODE" in public|private) ;; *) echo "usage: $0 <exe> public|private"; exit 1 ;; esac
OUT=build/ios
APP="$OUT/LAS Recomp.app"
IPA="$OUT/LASRecomp-$MODE.ipa"
IOS_MIN=${IOS_MIN:-16.0}
BUNDLE_ID=${BUNDLE_ID:-$(sed -n 's/^#define RT_BUNDLE_ID "\(.*\)"/\1/p' runtime/paths.h)}
VERSION=$(sed -n 's/^#define LARECOMP_VERSION "\(.*\)"/\1/p' runtime/ui/launcher.mm)
case "$VERSION" in  # (CFBundleShortVersionString must be digits: beta_<n> -> 0.9.<n>, v1.2.3 -> 1.2.3)
    beta_*) VERSION="0.9.${VERSION#beta_}" ;;
    v*) VERSION="${VERSION#v}" ;;
esac
[ -f "$EXE" ] || { echo "no $EXE (make ios)"; exit 1; }
# no game code inside the executable: it is built on the player's own files and kept in Frameworks/
if nm "$EXE" | grep ' _f_70' >/dev/null; then echo "$EXE has game code linked in"; exit 1; fi

rm -rf "$APP" "$IPA" "$OUT/Payload" "$OUT/LAS Recomp.ipa"
mkdir -p "$APP/Frameworks"
cp "$EXE" "$APP/LASRecomp"
strip -S "$APP/LASRecomp"
sed -e "s/@BUNDLE_ID@/$BUNDLE_ID/" -e "s/@VERSION@/$VERSION/" -e "s/@IOS_MIN@/$IOS_MIN/" tools/ios/Info.plist > "$APP/Info.plist"
plutil -lint "$APP/Info.plist" >/dev/null

# icons: iPhone 60pt @2x/@3x, iPad 76pt @2x and 83.5pt @2x, from the 1024 px one (no transparency: iOS icons are opaque)
for spec in "AppIcon60x60@2x:120" "AppIcon60x60@3x:180" "AppIcon76x76@2x~ipad:152" "AppIcon83.5x83.5@2x~ipad:167"; do
    sips -z "${spec#*:}" "${spec#*:}" docs/icon/icon_1024.png --out "$APP/${spec%%:*}.png" >/dev/null
done
cp docs/icon/icon_1024.png "$APP/icon.png"  # (the window icon at run time: runtime/paths.c rt_window_icon)
cp third_party/fonts/Baloo2/Baloo2-SemiBold.ttf "$APP/"
cp third_party/fonts/Baloo2/OFL.txt "$APP/Baloo2-OFL.txt"
cp third_party/fonts/DejaVu/DejaVuSans.ttf "$APP/"  # (the game's font engine needs a TrueType font with a Unicode cmap: pl.c)
cp third_party/fonts/DejaVu/LICENSE "$APP/DejaVu-LICENSE.txt"

cp third_party/fonts/DejaVu/DejaVuSans-Bold.ttf "$APP/" 2>/dev/null || true  # (the start screen's font, launcher.mm)
if [ "$MODE" = private ]; then
    [ -f "$OUT/lasgame.dylib" ] || { echo "no $OUT/lasgame.dylib: make ios-gen first (make ios-private does)"; exit 1; }
    cp "$OUT/lasgame.dylib" "$APP/Frameworks/lasgame.dylib"
    echo "game code: included ($(du -h "$OUT/lasgame.dylib" | cut -f1)), id $(cat "$OUT/lasgame.dylib.id" 2>/dev/null || echo '?')"
    echo "PRIVATE build: it holds code made from your game files. Do not share this .ipa."
else
    rmdir "$APP/Frameworks"
    echo "public build: no game code inside; the app runs the game's own code in its interpreter"
fi

# entitlements ride in an ad-hoc signature (every Mach-O, the library first)
for f in "$APP"/Frameworks/*.dylib "$APP/LASRecomp"; do
    [ -f "$f" ] && codesign --force --sign - --entitlements tools/ios/entitlements.plist "$f"
done

mkdir -p "$OUT/Payload"
cp -R "$APP" "$OUT/Payload/"
(cd "$OUT" && zip -qry "$(basename "$IPA")" Payload)
rm -rf "$OUT/Payload"
if [ "$MODE" = public ] && unzip -l "$IPA" | grep -qi 'lasgame'; then
    rm -f "$IPA"
    echo "REFUSED: the public .ipa held game code (lasgame)"; exit 1
fi
echo "$IPA"
