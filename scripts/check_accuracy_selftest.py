#!/usr/bin/env python3
"""check_accuracy.py 자체 검증 — 합성 회귀를 주입해 실제로 잡는지 확인한다.

안 잡히는 하네스는 없는 것만 못하다. 기준 CSV 를 변조해 알려진 회귀를 만들고,
FAIL 이 나야 할 것에서 FAIL 이, PASS 여야 할 것에서 PASS 가 나는지 본다.

사용: python3 scripts/check_accuracy_selftest.py <기준CSV디렉토리>
"""
import csv
import glob
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "check_accuracy.py")


def read(path):
    with open(path) as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write(rows, fields, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "candidate.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    return out_dir


# ── 합성 회귀 주입기 ───────────────────────────────────────

def drop_label(rows, label="smartphone"):
    """소물체 완전 소실 — 2026-08-10 의 560px 사건과 같은 형태."""
    return [r for r in rows if r["label"] != label]


def halve_label(rows, label="smartphone"):
    """소물체 커버리지 반토막 — 깜빡이며 놓치는 형태."""
    out, seen = [], 0
    for r in rows:
        if r["label"] == label:
            seen += 1
            if seen % 2:
                continue
        out.append(r)
    return out


def shift_center(rows, label, mm):
    out = []
    for r in rows:
        r = dict(r)
        if r["label"] == label:
            r["x"] = f"{float(r['x']) + mm / 1000:.4f}"
        out.append(r)
    return out


def shrink_extent(rows, label, mm):
    out = []
    for r in rows:
        r = dict(r)
        if r["label"] == label:
            r["w"] = f"{float(r['w']) - mm / 1000:.4f}"
        out.append(r)
    return out


def add_jitter(rows, label, mm):
    """지터만 키운다 — 중앙값은 그대로 두고 프레임마다 번갈아 흔든다."""
    out, i = [], 0
    for r in rows:
        r = dict(r)
        if r["label"] == label:
            r["x"] = f"{float(r['x']) + (mm if i % 2 else -mm) / 1000:.4f}"
            i += 1
        out.append(r)
    return out


CASES = [
    ("무변화 (동일 CSV)",                 lambda r: r,                              "PASS"),
    ("스마트폰 완전 소실",                lambda r: drop_label(r),                  "FAIL"),
    ("스마트폰 커버리지 50%",             lambda r: halve_label(r),                 "FAIL"),
    ("중심 1mm 이동 (하한 아래)",         lambda r: shift_center(r, "book", 1.0),   "PASS"),
    ("중심 5mm 이동",                     lambda r: shift_center(r, "book", 5.0),   "FAIL"),
    ("크기 10mm 축소",                    lambda r: shrink_extent(r, "laptop", 10), "FAIL"),
    ("지터 5mm 증가 (중앙값 불변)",       lambda r: add_jitter(r, "thermos", 5.0),  "FAIL"),
]


def main():
    ref_dir = sys.argv[1]
    src = [p for p in glob.glob(os.path.join(ref_dir, "*.csv"))
           if not p.endswith("_raw.csv")][0]
    fields, rows = read(src)

    print(f"기준: {src}\n")
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, fn, expect) in enumerate(CASES):
            cand = write(fn(rows), fields, os.path.join(tmp, f"case{i}"))
            p = subprocess.run(
                [sys.executable, HARNESS, "--ref", ref_dir, "--cand", cand],
                capture_output=True, text=True)
            got = "FAIL" if p.returncode else "PASS"
            hit = got == expect
            ok &= hit
            print(f"  {'OK ' if hit else '틀림'}  {name:28s} 기대={expect} 실제={got}")
            if not hit:
                print("\n".join("        " + ln for ln in p.stdout.splitlines()))
    print("\n자체 검증 " + ("통과 — 하네스가 회귀를 실제로 잡는다"
                            if ok else "실패 — 하네스를 고쳐야 한다"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
