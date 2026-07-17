#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Emit soundmodem/csrc/tm_tables.c from the ASRTU-1 format sheets.
#
# The generated C is the maintained artifact (readable, hand-editable); this
# script keeps it in sync with the CSV sheets and is a development tool only,
# not part of the crate build. test/tm_verify.c independently re-parses the
# CSVs and cross-checks the compiled tables, so a bug here or a manual edit
# there is caught rather than trusted.
#
# Usage: gen_tm_tables.py <ASRTU-1 dir> > csrc/tm_tables.c

import csv
import re
import sys

BASE = {"UX": "TM_UX", "SX": "TM_SX", "FX": "TM_FX", "HX": "TM_HX", "X": "TM_RSVD"}


def parse_offset(full_code):
    # <PKT>-C<sub><byte4>.<bit>.<width>; the leading digit is the sub type.
    tok = full_code.split("-C", 1)[1]
    digits, bit, width = tok.split(".")
    return int(digits) % 10000, int(bit), int(width)


def parse_unit(func):
    m = re.search(r"<([^>]*)>", func)
    if not m:
        return func, ""
    return func[: m.start()] + func[m.end():], m.group(1)


def parse_enum_key(k):
    k = k.strip()
    if k.endswith("H") or k.endswith("h"):
        return int(k[:-1], 16)
    if k and all(c in "01" for c in k):
        return int(k, 2)  # binary code (e.g. 3-bit status)
    return int(k)


def parse_enum(body):
    # body is "BASE|entry,entry,..." (half-width parens) or with full-width
    # colon/semicolon separators. Returns list of (key, desc); catch-all
    # ("other"/"其他") entries without a numeric key are dropped.
    base, items = body.split("|", 1)
    out = []
    paren = re.findall(r"([0-9A-Fa-f]+H?|[01]+)\s*[\(（]([^\)）]*)[\)）]", items)
    if paren:
        for k, desc in paren:
            out.append((parse_enum_key(k), desc.strip()))
        return BASE.get(base.strip(), "TM_UX"), out
    for seg in re.split(r"[;；]", items):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"([0-9A-Fa-f]+H?|[01]+)\s*[:：]\s*(.*)", seg)
        if not m:
            continue
        try:
            key = parse_enum_key(m.group(1))
        except ValueError:
            continue
        out.append((key, m.group(2).strip()))
    return BASE.get(base.strip(), "TM_UX"), out


def parse_affine(expr, base):
    # expr is affine in the base token: value = A * base + B.
    i = expr.index(base)
    left, right = expr[:i], expr[i + len(base):]
    a = 1.0
    if left:
        a *= float(left.rstrip("*"))
    m = re.match(r"^((?:[*/][0-9.]+)*)([+-][0-9.]+)?$", right)
    mult, add = m.group(1), m.group(2)
    for tok in re.findall(r"[*/][0-9.]+", mult):
        if tok[0] == "*":
            a *= float(tok[1:])
        else:
            a /= float(tok[1:])
    return a, (float(add) if add else 0.0)


def parse_func(func):
    # Returns (base_c, dec, mul, off, unit, enum_list).
    func = func.strip().strip('"').strip()
    func, unit = parse_unit(func)
    dec_prefix = re.match(r"F\.(\d+)=", func)
    decimals = int(dec_prefix.group(1)) if dec_prefix else None
    body = func.split("=", 1)[1] if "=" in func else func

    if "{" in body:
        inner = body[body.index("{") + 1: body.rindex("}")]
        base_c, enums = parse_enum(inner)
        return base_c, -1, 1.0, 0.0, unit, enums

    base = next((b for b in ("UX", "SX", "FX", "HX", "X") if b in body), "X")
    if base in ("HX", "X"):
        return BASE[base], -1, 1.0, 0.0, unit, []
    mul, off = parse_affine(body, base)
    scaled = mul != 1.0 or off != 0.0
    if decimals is None:
        decimals = 6 if (scaled or base == "FX") else -1
    return BASE[base], decimals, mul, off, unit, []


def cstr(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    root = sys.argv[1]
    with open(f"{root}/system.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]

    packets = []  # (main, sub, code, fields)
    enum_pool = {}  # key: tuple of (k,desc) -> name
    enum_order = []

    for r in rows:
        if len(r) < 5 or not r[4].strip():
            continue
        main_t = int(r[1].strip().rstrip("Hh"), 16)
        sub_t = int(r[2].strip().rstrip("Hh"), 16)
        code = r[4].strip()
        fields = []
        with open(f"{root}/{code}.csv", encoding="utf-8") as cf:
            for fr in list(csv.reader(cf))[1:]:
                # Some sheets omit the trailing english-name column.
                if len(fr) < 4 or not fr[0].strip():
                    continue
                byte, bit, width = parse_offset(fr[0].strip())
                if width <= 0 or width > 4096:
                    continue  # malformed offset (e.g. a negative tail width)
                en = fr[4].strip() if len(fr) > 4 else ""
                base_c, dec, mul, off, unit, enums = parse_func(fr[3])
                ename = "NULL"
                nmap = 0
                if enums:
                    tup = tuple(enums)
                    if tup not in enum_pool:
                        name = f"en_{len(enum_order)}"
                        enum_pool[tup] = name
                        enum_order.append((name, enums))
                    ename = "&%s[0]" % enum_pool[tup]
                    nmap = len(enums)
                fields.append((byte, bit, width, base_c, dec, mul, off,
                               unit, ename, nmap, fr[2].strip(), en))
        packets.append((main_t, sub_t, code, fields))

    out = []
    w = out.append
    w("/* SPDX-License-Identifier: GPL-3.0-or-later */")
    w("/*")
    w(" * ASRTU-1 telemetry field tables.")
    w(" *")
    w(" * GENERATED from the ASRTU-1 format sheets by tools/gen_tm_tables.py and")
    w(" * committed as the maintained source of truth. test/tm_verify.c re-parses")
    w(" * the sheets independently and cross-checks these tables. Edit the sheets")
    w(" * (or this file) and re-run the generator; never let the two drift.")
    w(" *")
    w(" * Field framing reverse-engineered by BG4QBF (archived Go tool).")
    w(" */")
    w("")
    w('#include "tm.h"')
    w("")
    w("#define NELEM(a)\t((unsigned short)(sizeof(a) / sizeof((a)[0])))")
    w("")
    for name, enums in enum_order:
        w(f"static const struct tm_enum {name}[] = {{")
        for k, desc in enums:
            w(f"\t{{ 0x{k:X}, {cstr(desc)} }},")
        w("};")
    w("")
    for main_t, sub_t, code, fields in packets:
        arr = code.lower().replace("-", "_")
        w(f"static const struct tm_field {arr}[] = {{")
        for (byte, bit, width, base_c, dec, mul, off, unit, ename, nmap,
             cn, en) in fields:
            w(f"\t{{ {byte}, {bit}, {width}, {base_c}, {dec}, "
              f"{mul!r}, 1.0, {off!r}, {cstr(unit)}, {ename}, {nmap}, "
              f"{cstr(cn)}, {cstr(en)} }},")
        w("};")
        w("")
    w("const struct tm_packet tm_packets[] = {")
    for main_t, sub_t, code, fields in packets:
        arr = code.lower().replace("-", "_")
        w(f"\t{{ 0x{main_t:02X}, 0x{sub_t:02X}, {cstr(code)}, "
          f"{arr}, NELEM({arr}) }},")
    w("};")
    w("")
    w("const unsigned int tm_npackets = NELEM(tm_packets);")
    w("")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
