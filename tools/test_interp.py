"""Differential test of the interpreter (runtime/interp*.c) against the lifter, one instruction at a time.

For many AArch64 instruction words it lifts each word with tools/lifter.py (portable C, --simd=c), compiles those as
functions, and runs both them and the interpreter (`interp_step_word`) on identical random register / memory state. Any
difference in registers, flags, V registers, memory or branch target is reported. Words the lifter rejects must be
rejected by the interpreter too.

Needs only Python 3 and clang. The lifter is itself checked against Unicorn by tools/test_lift.py / test_simd.py; this
test makes the interpreter agree with the lifter, instruction for instruction.

Usage: test_interp.py [--words N] [--reps R] [--seed S] [--jobs J] [--only integer|simd|all] [--keep]
"""
import argparse
import os
import random
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifter import Lifter, Unsupported, bits  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 0x7000000000
BUF = 0x7200000000
BUF_SIZE = 0x30000
BUF_MID = BUF + 0x10000


def is_mem(w):
    return bits(w, 25, 1) == 0 and bits(w, 27, 1) == 1


def skip(w):
    """Words not worth comparing: calls out of the instruction (svc, brk), or encodings the architecture leaves
    unallocated that the lifter turns into C with an undefined shift (whatever the host compiler makes of it)."""
    if (w & 0xFF000000) == 0xD4000000:
        return True
    sf = bits(w, 31, 1)
    if not sf:
        op0 = bits(w, 25, 4)
        if (op0 & 0xE) == 0x8 and bits(w, 23, 3) == 6 and (bits(w, 16, 6) >= 32 or bits(w, 10, 6) >= 32):
            return True  # bitfield, 32-bit with immr / imms >= 32
        if (op0 & 0xE) == 0x8 and bits(w, 23, 3) == 7 and bits(w, 10, 6) >= 32:
            return True  # extr
        if (op0 & 7) == 5 and not bits(w, 28, 1) and bits(w, 24, 4) == 0xB and not bits(w, 21, 1) and bits(w, 10, 6) >= 32:
            return True  # add / sub shifted register, amount >= 32
    return False


def test_body(ins, pc):
    """C for one lifted instruction as a straight-line function body; returns (body, kind) kind 1 = branch."""
    nxt = f'0x{pc + 4:x}ull'
    k = ins.kind
    if k in ('op', 'trap'):
        return ins.code, 0
    if k == 'asm':
        raise Unsupported('asm')
    if k == 'b':
        return f'c->pc = 0x{ins.target:x}ull;', 1
    if k == 'bl':
        return f'c->x[30] = {nxt}; c->pc = 0x{ins.target:x}ull;', 1
    if k == 'bcond':
        return f'c->pc = ({ins.cond}) ? 0x{ins.target:x}ull : {nxt};', 1
    if k == 'br':
        return f'c->pc = {ins.target};', 1
    if k == 'blr':
        return f'{{ uint64_t t = {ins.target}; c->x[30] = {nxt}; c->pc = t; }}', 1
    if k == 'ret':
        return 'c->pc = c->x[30];', 1
    raise Unsupported(k)


def gen_words(rng, n, only):
    """Random words, then mutations of the ones the lifter accepts (to reach into the narrow valid encodings)."""
    lifter = Lifter(BASE, 'c')

    def classify(w):
        top = bits(w, 25, 4)
        simd = (top & 7) == 7 or (bits(w, 26, 1) and (top & 5) == 4)
        if only == 'integer' and simd:
            return None
        if only == 'simd' and not simd:
            return None
        return w

    ok, bad = {}, {}
    def consider(w, pc):
        if bits(w, 27, 3) == 0b011 and bits(w, 24, 2) == 0 and not bits(w, 25, 1):  # ldr literal: keep it in the buffer
            w = (w & ~(0x7FFFF << 5)) | ((rng.randrange(-0x100, 0x100) & 0x7FFFF) << 5)
        if is_mem(w) and (bits(w, 27, 3) == 0b111 and bits(w, 24, 2) == 0 and bits(w, 21, 1) and bits(w, 10, 2) == 2
                          or (w & 0xBFA00000) == 0x0C800000):
            rm, rn = bits(w, 16, 5), bits(w, 5, 5)
            if rm == rn and rm < 31:  # (the test gives rm a small value, which as the base would fault)
                return
        if classify(w) is None or w in ok or w in bad or skip(w):
            return
        try:
            ins = lifter.lift(w, pc)
            if ins.kind == 'trap':
                bad[w] = 1
                return
            body, kind = test_body(ins, pc)
        except Unsupported:
            bad[w] = 1
            return
        ok[w] = (pc, body, kind)

    def pc_of(i):
        return BUF_MID + 0x1000 + (i * 4) % 0x2000

    i = 0
    while len(ok) < n // 3 and i < n * 4:
        consider(rng.getrandbits(32), pc_of(i))
        i += 1
    seeds = list(ok)
    gen = 0
    while len(ok) < n and seeds and gen < 8:
        new = []
        for w in seeds:
            for _ in range(3):
                x = w
                for _ in range(rng.choice((1, 1, 2, 3))):
                    x ^= 1 << rng.randrange(32)
                before = len(ok)
                consider(x, pc_of(i))
                i += 1
                if len(ok) > before:
                    new.append(x)
            if len(ok) >= n:
                break
        seeds = new
        gen += 1
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--words', type=int, default=40000)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 2)
    ap.add_argument('--only', choices=('integer', 'simd', 'all'), default='all')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--chunk', type=int, default=2500)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ok, bad = gen_words(rng, args.words, args.only)
    words = sorted(ok)
    print(f'{len(words)} supported words, {len(bad)} unsupported sampled', file=sys.stderr)

    work = os.path.join(ROOT, 'build', 'test_interp')
    os.makedirs(work, exist_ok=True)
    chunks = [words[i:i + args.chunk] for i in range(0, len(words), args.chunk)]
    cfiles = []
    for ci, ch in enumerate(chunks):
        path = os.path.join(work, f'oracle_{ci:03d}.c')
        with open(path, 'w') as f:
            f.write('#include "cpu.h"\n')
            for k, w in enumerate(ch):
                pc, body, kind = ok[w]
                f.write(f'void t{ci}_{k}(Cpu *c) {{ {body} }}\n')
            f.write(f'GuestFn tests{ci}[] = {{' + ','.join(f't{ci}_{k}' for k in range(len(ch))) + '};\n')
        cfiles.append(path)
    with open(os.path.join(work, 'table.c'), 'w') as f:
        f.write('#include "cpu.h"\n')
        for ci in range(len(chunks)):
            f.write(f'extern GuestFn tests{ci}[];\n')
        f.write('GuestFn *const test_chunks[] = {' + ','.join(f'tests{ci}' for ci in range(len(chunks))) + '};\n')
        f.write(f'const int test_chunk_size = {args.chunk};\n')
        f.write(f'const int ntests = {len(words)};\n')
        f.write('const unsigned test_words[] = {' + ','.join(f'0x{w:08x}u' for w in words) + '};\n')
        f.write('const unsigned long long test_pcs[] = {' + ','.join(f'0x{ok[w][0]:x}ull' for w in words) + '};\n')
        f.write('const unsigned char test_kinds[] = {' + ','.join(str(ok[w][2]) for w in words) + '};\n')
        bw = sorted(bad)[:20000]
        f.write(f'const int nbad = {len(bw)};\n')
        f.write('const unsigned bad_words[] = {' + ','.join(f'0x{w:08x}u' for w in bw) + ',0};\n')
        f.write('const unsigned long long bad_pc = 0x%xull;\n' % (BUF_MID + 0x1000))

    cflags = ['-O1', '-fno-strict-aliasing', '-mfma', '-Wno-deprecated-declarations', '-w', '-I', os.path.join(ROOT, 'runtime')]
    objs = []
    procs = []
    srcs = cfiles + [os.path.join(work, 'table.c'), os.path.join(ROOT, 'tools', 'test_interp_harness.c'),
                     os.path.join(ROOT, 'runtime', 'interp.c'), os.path.join(ROOT, 'runtime', 'interp_decode.c'),
                     os.path.join(ROOT, 'runtime', 'interp_simd.c')]
    todo = []
    for s in srcs:
        o = os.path.join(work, os.path.basename(s) + ('.itest.o' if 'runtime' in s else '.o'))
        flags = cflags + (['-DINTERP_TEST'] if os.path.basename(s) in ('interp.c',) else [])
        todo.append((s, o, flags))
        objs.append(o)
    running = []
    for s, o, flags in todo:
        while len(running) >= args.jobs:
            p = running.pop(0)
            if p.wait():
                sys.exit('compile failed')
        running.append(subprocess.Popen(['clang', *flags, '-c', s, '-o', o]))
    for p in running:
        if p.wait():
            sys.exit('compile failed')
    exe = os.path.join(work, 'harness')
    subprocess.check_call(['clang', '-o', exe, *objs, '-lm', '-lpthread'])
    r = subprocess.run([exe, str(args.reps), str(args.seed)])
    if not args.keep:
        for o in objs:
            pass
    sys.exit(r.returncode)


if __name__ == '__main__':
    main()
