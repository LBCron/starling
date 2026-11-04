# gte-reranker-modernbert-base on ONNX Runtime, CPU fp32: onboarding evidence (2026-11-04)

Development evidence for `Alibaba-NLP/gte-reranker-modernbert-base` on the `bert-onnx-cpu-v1` route (issues #101 and #102, PR #291), from the onboarding recipe run on 2026-11-04 on Linux x64 (Intel Core i7-5820K, 12 logical CPUs, 94 GB RAM shared with other work), CPU only, inside a 20 GB memory limit (`systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`). It is not release qualification: no gate ran, no fixture is protected, and nothing here is binding.

## Inputs

- Upstream: `Alibaba-NLP/gte-reranker-modernbert-base` at `f7481e6055501a30fb19d090657df9ec1f79ab2c`, including its `onnx/model.onnx`. Every file the candidate pins was checked by size and SHA-256 (`source-lock.json`).
- Candidate: `model-pipeline/candidates/gte-reranker-modernbert-base/profile.json`, SHA-256 `4db17cb8b027457c02715b3828175b3f437a1c6345558465477688b732c2aa0a`.
- Corpus: `model-pipeline/corpora/gte-reranker-modernbert-base/reranking.jsonl`, 28 cases of 3 to 1,468 tokens.
- Environment: `environment.json`; `model-pipeline/reference/encoder/requirements.txt` (torch 2.14.1+cpu, transformers 5.18.0, sentence-transformers 6.1.0, tokenizers 0.23.2, onnxruntime 1.30.0, numpy 2.4.6).

## Runs

The recipe ran in full twice, seven stages each time, at two threads. The second run, 10:53 to 11:02 UTC, is the one `report.json` comes from; the first gave a byte-identical report. After it only a comment and a message string in `onboarding_recipe.py` changed, which is not a route input. A final `--resume` on commit `2fa9869` at 11:08 UTC reused six stages and reran `review`, so `run.json` shows six stages `cached`, and `review.json` names that commit.

## Results

Status **passed**: 280 checks, 0 failed, 28 cases (`report.json`).

| Comparison | Checks | Largest difference | Bound |
| --- | ---: | ---: | ---: |
| .NET from text vs Python ONNX Runtime | 28 | 1.187e-07 | 1e-05 |
| .NET from text vs PyTorch | 28 | 2.872e-06 | 1e-04 |
| .NET, padded batch vs single, from ids | 28 | 7.000e-07 | 1e-05 |
| .NET, padded batch vs single, from text | 28 | 7.000e-07 | 1e-05 |
| .NET from ids vs Python ONNX Runtime | 28 | 1.187e-07 | 1e-05 |
| .NET from ids vs PyTorch | 28 | 2.872e-06 | 1e-04 |
| Python ONNX Runtime, padded batch vs single | 28 | 7.153e-07 | 1e-05 |
| Python ONNX Runtime vs PyTorch | 28 | 2.861e-06 | 1e-04 |
| PyTorch, padded batch vs single | 28 | 6.557e-06 | 1e-05 |
| Reference vs the public `CrossEncoder` API | 28 | 0 | 1e-04 |

The padded-batch bound for PyTorch is fixed policy; its value here is 65% of the bound. A longer corpus could exceed it, which would be a finding to report, not a value to change.

## Exploratory probes

Outside the recipe and not gate evidence. These are Python, one row, four threads, on a shared machine, so the timings are development timings. The .NET measurements follow under [.NET: CPU arena](#net-cpu-arena).

`probe.py` (`probe.log`) compared PyTorch and ONNX Runtime at 8 to 8,000 tokens, within 4.3e-06 at each length, and showed that padding with id 0 or 50283 gives the same result. Its peak RSS of 2.1 to 11.4 GB is for a process that also held two PyTorch models, so it is not an ONNX Runtime figure.

`probe8192.py` (`probe8192.log`, rerun to retain its output) tokenized one pair to 8,192 tokens, the same ids as transformers with `truncation=True`, and PyTorch and ONNX Runtime differed by 2.9e-06. Its 10.5 GB peak also includes PyTorch.

`ort_peak.py` (`ort-peak.log`) has ONNX Runtime alone in the process, on one 8,192-token pair run repeatedly on one session:

| CPU arena | Runs | Time per run | Peak RSS after load | After run 1 | After run 2 | After runs 3 and 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| on (default, two processes) | 2, 4 | 45 to 53 s | 0.9 GB | 10.1 GB | 19.2 GB | 19.2 GB |
| off | 2 | 54 to 56 s | 0.9 GB | 9.3 GB | 10.0 GB | not run |

The first three lines of `ort-peak.log` come from an earlier version of the script, which always ran twice with the arena on.

## .NET: CPU arena

The maintainer's decision on PR #291 was to measure on .NET first, and to disable the arena for the route if .NET showed the growth. It did, so `BertSession` now runs all nine profiles with the arena off.

`arena-probe/` is an out-of-tree console project: it calls the public `BertReranker` and `BertTextEmbedder` APIs, one setting per process, because `VmHWM` never falls. It reads `VmHWM` and `VmRSS` from `/proc/self/status` after each run. Each process ran in a transient systemd unit with `MemorySwapMax=0`. The arena-on run of `dotnet-long.log` had `MemoryMax=26G`, used because `agent-budget level` printed `ok` at the start, and every other process had 20G. No cap was hit. The switch that selects the setting is the `AppContext` switch `Starlings.Models.Bert.CpuMemoryArena`.

One pair of 8,192 tokens, four threads, repeated on one session (`dotnet-long.log`):

| CPU arena | Runs | Peak RSS after load | After run 1 | After run 2 | After runs 3, 4 | RSS after the last run | Unit memory peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| on | 4 | 0.94 GB | 10.12 GB | 19.27 GB | 19.28 GB | 19.28 GB | 18.3 GB |
| off | 4 | 0.94 GB | 9.43 GB | 10.11 GB | 10.11 GB | 0.98 GB | 9.6 GB |

The growth on .NET matches the Python probe. The unit's memory peak is about 1 GB below `VmHWM` in the arena-on run and 0.5 GB below in the arena-off run; the reason was not investigated.

Scores: the logit was 1.492696 on every run of both settings, the value Python ONNX Runtime gives (`ort-peak.log`). Through `BertReranker` at `maxTokens` 64, 512 and 2,048 the logits were 0.807708, 0.985387 and 1.230980, each equal to `ort_peak.py` at that length to the six printed decimals. In `dotnet-short.log` both settings gave the same printed value on each input, including the first vector element of `paraphrase-multilingual` (0.294363) and `bge-base` (-0.026069). This is one repetitive pair per length, not parity evidence over a corpus, so pairs past 1,468 tokens stay untested on .NET.

Latency is the weaker part of this evidence: the machine was shared, and a quiet window for a paired rerun did not come in about two hours of waiting, so no clean paired .NET latency exists. At 8,192 tokens, arena on took 56.5, 52.5, 47.1 and 43.6 s with the machine quiet at the start. Arena off took 52.2 s for the first run, then 101.0, 125.3 and 85.8 s while other jobs started on the machine (the level was `tight` when it ended), so only the 52.2 s is comparable. Together with the Python probe's 45 to 53 s against 54 to 56 s, the estimate is about 10 to 20 percent slower without the arena at this length. Process CPU time over the four runs was 12 min 9 s with the arena and 14 min 44 s without, which is contaminated in the same way.

Short and medium inputs, alternating settings, three processes each, two threads (`dotnet-short.log`, summarized by `summarize_short.py`). These ran under heavy load, with the load average reaching 101 on 12 cores. Load only slows a run, so the table gives the fastest run after the first in each process, and it is only good for spotting a large cost:

| Profile | Input | Fastest, arena on | Fastest, arena off | Difference | Peak RSS on, off | RSS after on, off |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gte-reranker-modernbert-base | 64 tokens, 8 rows | 1.477 s | 1.477 s | 0% | 1.08, 1.04 GB | 1.08, 0.99 GB |
| gte-reranker-modernbert-base | 512 tokens, 1 row | 1.615 s | 1.821 s | +13% | 1.13, 1.06 GB | 1.12, 0.99 GB |
| gte-reranker-modernbert-base | 2,048 tokens, 1 row | 9.814 s | 10.942 s | +11% | 2.35, 1.68 GB | 2.35, 1.01 GB |
| paraphrase-multilingual | about 60 tokens, 32 rows | 1.520 s | 1.438 s | -5% | 0.96, 0.96 GB | 0.77, 0.89 GB |
| bge-base | about 60 tokens, 8 rows | 0.594 s | 0.802 s | +35% | 0.77, 0.73 GB | 0.77, 0.73 GB |

The medians overlap heavily between settings, and the bge-base difference rests on single fastest runs, so it should be remeasured on a quiet machine before anyone relies on it.

Not measured: `bge-m3` and `bge-reranker-v2-m3`, the other 8,192-token profiles, which this change also affects. `DedicatedRunThreads` stays off on this route, unlike EmbeddingGemma 2, so run buffers are freed into the calling thread's glibc heap; a host calling from a rotating thread pool was not measured. No soak ran.

`dotnet-long-aborted-parser-bug.log` is the first attempt, which stopped at a parsing bug in the probe's `/proc` reader once RSS reached 10 GB. It is kept for completeness; no figure comes from it.

## Not done

- The .NET path past 1,468 tokens over a corpus: only the one repetitive pair above ran. No corpus row reaches truncation.
- Peak memory for a batch of long rows.
- `--distribution`, any gate from G0 to G7, Windows, CUDA.
- References and bounds are proposed; none is adopted into `protected/`.

## Files

| File | Contents |
| --- | --- |
| `report.json`, `report.md` | Every check of the evaluator with its value and bound |
| `review.json`, `review.md` | The review stage's record (development parity passed; qualification unapproved) |
| `source-lock.json` | The upstream files verified by the `assets` stage |
| `environment.json` | Host, SDK and package versions (local paths replaced by `<repo>`, `<models>`, `<dotnet>`) |
| `run.json` | The pipeline run record of the `--resume` (stage inputs, digests, timings; local paths replaced the same way) |
| `probe.py`, `probe.log`, `probe8192.py`, `probe8192.log`, `ort_peak.py`, `ort-peak.log` | The exploratory probes and their output |
| `arena-probe/` | The .NET probe (`Program.cs`, `ArenaProbe.csproj`) |
| `dotnet-long.log`, `dotnet-short.log`, `dotnet-long-aborted-parser-bug.log`, `summarize_short.py` | Its output (local paths replaced) and the summary of `dotnet-short.log` |

Commands (from the repository root; paths shortened):

```sh
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
  .venv/bin/model-pipeline run model-pipeline/recipes/gte-reranker-modernbert-base-cpu.yaml \
    --snapshot <snapshot at f7481e60> --reference-python <encoder environment>/bin/python \
    --out artifacts/onboarding/gte-reranker-modernbert-base --threads 2 --timeout 1800
python -I docs/evidence/gte-reranker-modernbert-onboarding-2026-11-04/ort_peak.py <snapshot> 8192 4 arena
python -I docs/evidence/gte-reranker-modernbert-onboarding-2026-11-04/ort_peak.py <snapshot> 8192 2 noarena
dotnet build -c Release docs/evidence/gte-reranker-modernbert-onboarding-2026-11-04/arena-probe -p:Repo=<repo>   # copy it out of the tree first
systemd-run --user --pipe --wait -p MemoryMax=26G -p MemorySwapMax=0 <probe> <snapshot> gte-reranker-modernbert-base 4 4 arena 8192
systemd-run --user --pipe --wait -p MemoryMax=20G -p MemorySwapMax=0 <probe> <snapshot> gte-reranker-modernbert-base 4 4 noarena 8192
```
