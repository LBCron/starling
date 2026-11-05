"""Summarises dotnet-paired.log: per input and process, the setting, the load at start, and the median and fastest run after run 1."""
import re
import statistics as st
import sys
from collections import defaultdict

groups = defaultdict(list)
load = runs = tokens = None
for line in open(sys.argv[1]):
    m = re.match(r"# \S+ unit=(\S+) .* load: ([\d.]+)", line)
    if m:
        unit, load = m.group(1), float(m.group(2))
        continue
    m = re.match(r"(\S+) (arena|noarena) threads (\d+) session arena (\w+)", line)
    if m:
        name, setting = m.group(1), m.group(2)
        assert (setting == "arena") == (m.group(4) == "True")
        runs, tokens = [], None
        continue
    m = re.search(r"(?:tokens (\d+) )?rows (\d+) run (\d+) .* time ([\d.]+)s", line)
    if m:
        tokens = m.group(1) or "60"
        runs.append(float(m.group(4)))
    if line.startswith("# exit"):
        assert line.startswith("# exit 0"), line
        groups[(name, tokens)].append((unit, setting, load, runs[1:]))

for key, rows in groups.items():
    print(key)
    for unit, setting, started, times in sorted(rows, key=lambda r: r[1]):
        print(f"  {unit} {setting:8} load {started:5.1f} median {st.median(times):7.3f} fastest {min(times):7.3f}")
    on = [r[3] for r in rows if r[1] == "arena"]
    off = [r[3] for r in rows if r[1] == "noarena"]
    median_on, median_off = st.median(map(st.median, on)), st.median(map(st.median, off))
    fastest_on, fastest_off = min(map(min, on)), min(map(min, off))
    print(f"  median of medians: on {median_on:.3f}, off {median_off:.3f}, {median_off / median_on - 1:+.1%}")
    print(f"  fastest: on {fastest_on:.3f}, off {fastest_off:.3f}, {fastest_off / fastest_on - 1:+.1%}")
