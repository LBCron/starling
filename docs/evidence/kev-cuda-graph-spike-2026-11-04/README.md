# Kev-0.8B CUDA graph spike: builder graph, placement and S1-shape latency, 2026-11-04

Issue #140, ask 1, step 2 of the triage plan. Development evidence on an RTX 4060 Ti, in Python ONNX Runtime. It adds no profile, changes no pin and changes no capability status.

## Result

- **Negative as asked.** The builder's `-e cuda -p fp32` export does not load under ONNX Runtime 1.30.0, and after the load repair it cannot continue from a cached state. It takes three hand repairs to run Kev's path.
- **This graph has no `GroupQueryAttention`.** The builder uses `MultiHeadAttention` here, because it has no fp32 `GroupQueryAttention` configuration for CUDA, and ONNX Runtime 1.30.0 keeps the fp32 node on the CPU in the current bundle. After the repairs all attention runs on the device. `GroupQueryAttention` on the device exists only in the fp16 and bf16 exports.
- **The state cache leaves the device on every graph with a plain `Run`.** It stays there only when the caller binds `present.*` to the device (I/O binding). `KevDecisionModel` uses a plain `Run`, so a new graph alone does not keep the cache on the device.
- **Latency improves.** With 4 questions, against the current profile's 270 ms: repaired fp32 184 ms, or 146 ms with the cache on the device. fp16 is 83 ms and 67 ms. fp16 is not parity-equal to the fp32 profile.
- **Load sensitivity needs both changes.** The graph alone more than halves the time at load but it still tracks load. Repaired fp32 with the cache on the device held 269 ms at load 40 to 49, while the host path took 530 ms at load 22 to 26.
- **Recommended next step:** a reproducible repaired fp32 graph plus device-bound caching in `KevDecisionModel`, run first from .NET on the RTX 3090. See [Recommendation](#recommendation).

## Setup

- Intel i7-5820K (12 threads), Linux 7.2.8, NVIDIA GeForce RTX 4060 Ti 16 GB (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`), driver 615.78.08. The RTX 3090 was not used.
- Python 3.12.14, `onnxruntime-gpu` 1.30.0, `onnxruntime-genai` 0.17.1, onnx 1.23.1. CUDA EP with `use_tf32=0`, read back from the session. Intra-op threads left at the ONNX Runtime default.
- The machine was shared with CI runners and other jobs throughout. Every cell records the 1-minute load average before and after. Load changed during the run, so cells are not all comparable; the tables say so where it matters.

Commands are in [run_spike.sh](run_spike.sh), the measurement in [spike_s1.py](spike_s1.py), and the repairs in [patches/](patches/). Each step ran inside `systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`:

```sh
python -m onnxruntime_genai.models.builder -i kev-merged-fp32 -o export-cuda-fp32 -p fp32 -e cuda -c cache \
  --extra_options exclude_lm_head=true exclude_mtp=true
run_spike.sh patch; run_spike.sh load; run_spike.sh placement; run_spike.sh parity
run_spike.sh bench; run_spike.sh hog 12; run_spike.sh bench rep2
```

### Graphs

| Name here | Source | `model.onnx` SHA-256 | Weights |
| --- | --- | --- | --- |
| current | The locked `kev-0.8b/onnx-fp32-cuda` bundle (`-e cpu -p fp32`) | `046aa7b4367dee6c29bd494f330e60c5d2f665540558e26ea02148ecfd7f6278` | `dd78b21c...` |
| builder fp32 | `-e cuda -p fp32`, exported again today | `b87bde0525c53d5ac3e6d9914bce9b5317cd89eefadfd6c3251579ec8660a763` | `dd78b21c...` |
| repaired fp32 | builder fp32 plus the three repairs | `a6478f4534ac898558ff8f5b3207af546c52cd6249848ff9d8a4f74eef858ab0` | `dd78b21c...` |
| fp16 | `-e cuda -p fp16` from the 2026-10-05 export, not rebuilt | `b5a8c359cb407bd3e27f30ad9a084ad7f67f591dff06c7e800781bf399b2b548` | its own file |

Today's export is byte-identical to the one in `admissions/kev.md`, so the builder is reproducible for this graph. The repaired graph also has the same digest as the earlier one. `model.onnx.data` of the export is byte-identical to the locked bundle's (`dd78b21c101ef945dc4b7ca31367647da70e4d7fbd3f3d2e05a3245c573dc51a`): an fp32 CUDA profile differs from the bundle by one 292 KB graph file.

## Does it load?

Each stage on the CUDA EP, with a 600-token state and then a 40-token row from its cache ([results/load-*.json](results/)).

| Graph | Session | State pass | Row from cache | Error |
| --- | --- | --- | --- | --- |
| builder fp32, as built | fails | | | `Node input 'past_key_values.0.key' is not a graph input, initializer, or output of a previous node.` |
| after repair 1 (past length read from layer 3) | ok | ok | fails | `Expand`: `left operand cannot broadcast on dim 3 LeftShape: {1,1,40,40}, RightShape: {1,1,40,640}` |
| after repair 2 (causal triangle [S, past + S]) | ok | ok | fails | `repeat_kv/Transpose_2`: buffer reuse shape mismatch |
| after repair 3 (1,480 `value_info` entries removed) | ok | ok | ok | |
| fp16, unmodified | ok | ok | ok | |
| current bundle | ok | ok | ok | |

These are the two failures of `admissions/kev.md`, "GPU results", unchanged. The repairs are hand edits of builder output and should not be adopted as they are.

Both causes are in the builder source and are still there in 0.19.0, the latest on PyPI. `make_past_key_subgraph` hardcodes `past_key_values.0.key`, which does not exist when layer 0 is a DeltaNet layer, and `is_gqa_supported` lists no `("cuda", FLOAT)` pair. This comes from reading the 0.19.0 wheel's Python files. That builder was not run, so a newer builder is not expected to fix this but it is not proven either.

## Placement

Session profile of one 600-token state pass and two 82-token rows from its cache ([results/placement-*.json](results/)). Log excerpts: [logs/](logs/).

| Graph | Attention layers | Nodes on CUDA | Nodes on CPU | Memcpy nodes | `present.*` after plain `Run` | `present.*` when bound |
| --- | --- | ---: | --- | ---: | --- | --- |
| current | 6 `GroupQueryAttention` **on CPU** | 649 | 8: 6 `GroupQueryAttention`, Gather, Cast | 25 | host | device |
| repaired fp32 | 6 `MultiHeadAttention` on CUDA | 761 | 188, all shape arithmetic for the mask and `repeat_kv` (Gather, Unsqueeze, Concat, Equal, Where, Reshape, Mul, Add, Slice, Squeeze) | 1 | host | device |
| fp16 | 6 `GroupQueryAttention` on CUDA | 654 | 2: Gather, Cast | 0 | host | device |

- All 18 `LinearAttention` layers are on CUDA in all three graphs.
- `GroupQueryAttention` took 115.6 ms of CPU kernel time in the current graph across those three passes. The repaired graph's CPU nodes took 4.8 ms in total.
- CUDA kernel times are not listed per op because CUDA work is queued, and a node's time can include waiting for earlier nodes.
- The cache of a 600-token state is 34.9 MB (17.5 MB on fp16), most of it not attention keys and values. With a host cache it crosses PCIe once per row.

## Latency in the S1 shape

The S1 shape: a state pass of 600 tokens, then one row per question from the state's cache, then the pointer head. Rows run one at a time, as `KevDecisionModel` does.

- **Tokens:** the first 600 tokens of a 2k document state, with real question branches from other records (17, 33, 62, 82 tokens for 4 questions, plus 36, 52, 99, 113 for 8). This is the branch length range Corvid's S1 used. The answers mean nothing and are read only for agreement between graphs.
- **Modes:** `host` keeps the cache in host memory with a plain `Run`, as `KevDecisionModel` does. `device` binds `present.*` to the device with I/O binding and feeds those values to every row.
- **Timing:** one process per cell, 10 warm-up calls excluded, then 100 calls, nearest-rank percentiles.

### First pass

Load 5 to 7 for the 4-question cells. The 8-question cells ran as other jobs started, so read those by the `min` column.

| Questions | Graph | Cache | p50 ms | p90 ms | min ms | State pass p50 | Rows p50 | GPU busy | Load |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | current | host | **270** | 281 | 261 | 151 | 119 | 60% | 6 |
| 4 | current | device | 265 | 279 | 255 | 150 | 114 | 59% | 5–7 |
| 4 | repaired fp32 | host | **184** | 184 | 183 | 94 | 90 | 92% | 6 |
| 4 | repaired fp32 | device | **146** | 147 | 145 | 86 | 60 | 97% | 6–7 |
| 4 | fp16 | host | 83 | 83 | 82 | 39 | 44 | 94% | 5 |
| 4 | fp16 | device | 67 | 68 | 66 | 35 | 32 | 94% | 6 |
| 8 | current | host | **508** | 586 | 455 | 190 | 320 | 46% | 6–10 |
| 8 | current | device | 748 | 1,153 | 456 | 201 | 503 | 29% | 15–17 |
| 8 | repaired fp32 | host | **283** | 290 | 279 | 94 | 189 | 87% | 10–11 |
| 8 | repaired fp32 | device | **216** | 218 | 214 | 87 | 130 | 95% | 13–16 |
| 8 | fp16 | host | 146 | 335 | 132 | 40 | 106 | 57% | 11–14 |
| 8 | fp16 | device | 106 | 112 | 103 | 35 | 71 | 91% | 12–13 |

Relative to the current profile with a host cache, using p50 at 4 questions and `min` at 8 questions:

| Graph, cache | 4 questions | 8 questions |
| --- | ---: | ---: |
| current, device | 0.98 | 1.00 |
| repaired fp32, host | 0.68 | 0.61 |
| repaired fp32, device | 0.54 | 0.47 |
| fp16, host | 0.31 | 0.29 |
| fp16, device | 0.25 | 0.23 |

- Binding the cache does nothing for the current graph, because `GroupQueryAttention` on the CPU forces the copies anyway.
- On the repaired graph, binding saves about 7 ms per row and 7 ms per state pass.
- The GPU is now the limit: 87 to 97% busy on the repaired graph with a quiet CPU, against 46 to 60% today. S1 saw 31 to 51% on the RTX 3090.
- An earlier host-only pass gave the same picture: p50 286, 184, 83 ms at 4 questions and 560, 281, 132 ms at 8. Its result files were overwritten by the run above, so these six numbers have no file here.

### Under CPU load (8 questions)

`hog` adds 12 busy loops (30 calls, 5 warm-up). `rep2` is a full repeat that overlapped CI runners, which pushed load well past anything planned. Cells ran one after another, so their loads differ and are listed.

| Graph | Cache | First pass, p50 (p90) ms, load | `hog` p50 (p90) ms, load | `rep2` p50 (p90) ms, load |
| --- | --- | ---: | ---: | ---: |
| current | host | 508 (586), 6–10 | 965 (1,462), 12–16 | 1,347 (2,118), 20–27 |
| current | device | 748 (1,153), 15–17 | 908 (1,126), 17–18 | 1,658 (2,093), 21–48 |
| repaired fp32 | host | 283 (290), 10–11 | 325 (456), 17 | 530 (736), 22–26 |
| repaired fp32 | device | 216 (218), 13–16 | 253 (414), 19 | **269 (360), 40–49** |
| fp16 | host | 146 (335), 11–14 | 203 (292), 17 | 355 (602), 21 |
| fp16 | device | 106 (112), 12–13 | 149 (235), 18 | **145 (185), 33–38** |

The current profile tracks load, as S1 found. A GPU-resident graph with a host cache still does, by less. With the cache on the device, the p50 moved by 24 to 36% while load went up by a factor of three. Why the host path is so much more sensitive was not profiled; the extra CPU work per row is the likely cause.

Device mode's rows are also the cost of a reused state. Without a state pass, four questions take 60 ms on repaired fp32 and 32 ms on fp16, and eight take 130 ms and 71 ms. That is arithmetic from these cells, not a measurement of `KevState` or of reuse across requests.

## Parity

| Check | Repaired fp32 | fp16 | Bound |
| --- | ---: | ---: | --- |
| S1-shape probabilities against the current graph, max abs ([results/](results/)) | 2.06e-6 | 9.69e-4 | not a gate |
| 64 records, 79 questions, against PyTorch CPU fixtures, single rows ([results/parity-cuda-fp32-repaired-fixtures.json](results/parity-cuda-fp32-repaired-fixtures.json)) | 4.03e-5, 0 flips | not run today | profile 2e-4 |
| Same, cached rows against PyTorch `probs` layout | 4.07e-5 | not run today | profile 2e-4 |
| Same, cached rows against single rows | 1.73e-6 | not run today | profile 1e-5 |
| decision-v7 development partition, max abs, recorded in `admissions/kev.md` | 8.05e-5, 0 flips | 6.39e-3, 0 flips | |

Host and device modes gave bit-identical probabilities in all six graph and question-count pairs of the first pass (last call of each cell). The repaired graph keeps the fp32 profile's bounds on these checks. fp16 does not: its drift is over the 2e-4 bound, so it would need its own proposed bound and a maintainer decision. These are development checks and no gate ran.

## Not covered

- **This is not the RTX 3090.** S1's 220 to 361 ms are from the 3090 and a whole Corvid call. Compare ratios, not milliseconds. Whether the ratios hold on the 3090 is unknown.
- **This is Python, not .NET.** Neither `KevDecisionModel` nor the Corvid path ran. The graph signature, the session digest lock and the cached path of the repaired graph were not exercised from C#. `OnnxModelSession` has `CreateIoBinding`, `CreateDeviceMemoryInfo` and `RunWithBinding`, but `KevDecisionModel` does not use them, and device binding of the Kev cache from C# was not tried.
- **The device cache was only measured.** Its lifetime, cancellation, concurrent callers and growth over time were not tested, and `KevState` keeps host values today.
- No soak, no gate, no long-state run. The earlier ceiling of the fp32 mask (8,192 tokens passed, 12,288 failed) comes from `admissions/kev.md`, and 8,192 is also `MaxStateTokens`, so the limit itself needs a check.
- `use_tf32=1`, bf16 and Kev-4B were not measured. Newer ONNX Runtime and GenAI versions were not run.
- No quiet-machine repeat of the 8-question cells: load stayed high at the end of the time box.

## Recommendation

1. **Build the fp32 route.** Make the graph repair a reviewed, reproducible step in `model-pipeline/reference/kev/` that starts from the pinned builder output, and pin the resulting digest. The weights are already hosted and locked, so only one small graph file needs hosting. This keeps the current bounds.
2. **Add device-bound caching to `KevDecisionModel` in the same work.** The graph alone gives 0.6 to 0.7 of today's latency and stays load-sensitive. With binding it is about 0.5, and it stops following CPU load, which is what Corvid asked for.
3. **Check it from .NET on the RTX 3090 before any profile.** This is a short throwaway measurement: the repaired graph, device binding, the S1 shape, and a test at 8,192 tokens. If the ratios hold, proceed to `kev-0.8b/onnx-fp32-cuda-native` with its own lock, parity evidence and admission, at Experimental at most.
4. **Ask the maintainer about fp16 as a separate decision.** It loads unmodified and is about twice as fast again, with half the device memory. Its drift is over today's bound and would need a new proposed bound. Consider it if 6 to 8 questions under load still miss 300 ms on the 3090.
5. **Report the builder failures upstream.** Both are small and have a reproducer in `admissions/kev.md`. Do not wait for a fix, and do not plan on a newer builder fixing them.

Cross-request prefix reuse stays a separate design. The rows-only numbers above show it would be worth about the whole state pass.
