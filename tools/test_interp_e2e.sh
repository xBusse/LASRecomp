#!/bin/sh
# End-to-end test of the interpreter: tools/interp_e2e/prog.c is compiled for AArch64 (clang, freestanding), loaded at
# a guest-like address and its t_* functions run by the interpreter; each checks itself. Needs clang with the AArch64
# target and lld. Usage: tools/test_interp_e2e.sh [opt level, default -O2]
set -e
cd "$(dirname "$0")/.."
OPT=${1:--O2}
W=build/test_interp_e2e
mkdir -p $W
BASE=0x7000100000
clang --target=aarch64-linux-gnu -march=armv8-a+crypto+crc -ffreestanding -fno-builtin -fno-stack-protector -fno-pic \
    -ffp-contract=off -fno-math-errno $OPT -c tools/interp_e2e/prog.c -o $W/prog.o
ld.lld -static -Ttext=$BASE --no-rosegment -z norelro -e t_fib $W/prog.o -o $W/prog.elf
llvm-objcopy -O binary $W/prog.elf $W/prog.bin
TEXT=$(llvm-objdump -h $W/prog.elf | awk '$2==".text"{print $4, $3}')
TOFF=$(printf '%x' $(( 0x$(echo $TEXT | cut -d' ' -f1) - BASE )))
TSZ=$(echo $TEXT | cut -d' ' -f2)
FUNCS=$(llvm-nm $W/prog.elf | awk '$2=="T" && $3 ~ /^t_/{printf "%s=%s ", $3, $1}')
clang -O1 -w -mfma -march=native -Iruntime -DINTERP_E2E tools/interp_e2e/run.c runtime/interp.c runtime/interp_decode.c \
    runtime/interp_simd.c tools/interp_e2e/stubs.c -o $W/run -lm -lpthread
$W/run $W/prog.bin ${BASE#0x} $TOFF $TSZ $FUNCS
