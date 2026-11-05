"""Summarises dotnet-short.log: per input and setting, the median and the fastest run after warm-up, per process."""
import re
import statistics as st
import sys
from collections import defaultdict

rows = defaultdict(list)
key = None
for line in open(sys.argv[1]):
    m = re.match(r"(\S+) (arena|noarena) threads (\d+) session arena (\w+)", line)
    if m:
        key = (m.group(1), m.group(2))
        cur = []
        continue
    m = re.search(r"(?:tokens (\d+) )?rows (\d+) run (\d+) .* time ([\d.]+)s peak ([\d.]+)G rss ([\d.]+)G", line)
    if m:
        cur.append((m.group(1) or "-", m.group(2), float(m.group(4)), float(m.group(5)), float(m.group(6))))
    if line.startswith("# exit") and key:
        k = (key[0], cur[0][0], cur[0][1], key[1])
        rows[k].append(cur[1:])  # run 1 is warm-up
        key = None

print("| Profile | Tokens | Rows | Arena | Median of process medians (ms) | Fastest run (ms) | Peak RSS (GB) | RSS after (GB) |")
for k in sorted(rows):
    med = [st.median(r[2] for r in p) * 1000 for p in rows[k]]
    best = min(r[2] for p in rows[k] for r in p) * 1000
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {st.median(med):.1f} ({' '.join(f'{x:.1f}' for x in med)}) | {best:.1f} | {max(p[-1][3] for p in rows[k]):.2f} | {max(p[-1][4] for p in rows[k]):.2f} |")
