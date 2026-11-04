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

Outside the recipe and not gate evidence. All are Python, one row, four threads, on a shared machine, so the timings are development timings. No probe ran on .NET.

`probe.py` (`probe.log`) compared PyTorch and ONNX Runtime at 8 to 8,000 tokens, within 4.3e-06 at each length, and showed that padding with id 0 or 50283 gives the same result. Its peak RSS of 2.1 to 11.4 GB is for a process that also held two PyTorch models, so it is not an ONNX Runtime figure.

`probe8192.py` (`probe8192.log`, rerun to retain its output) tokenized one pair to 8,192 tokens, the same ids as transformers with `truncation=True`, and PyTorch and ONNX Runtime differed by 2.9e-06. Its 10.5 GB peak also includes PyTorch.

`ort_peak.py` (`ort-peak.log`) has ONNX Runtime alone in the process, on one 8,192-token pair run repeatedly on one session:

| CPU arena | Runs | Time per run | Peak RSS after load | After run 1 | After run 2 | After runs 3 and 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| on (default, two processes) | 2, 4 | 45 to 53 s | 0.9 GB | 10.1 GB | 19.2 GB | 19.2 GB |
| off | 2 | 54 to 56 s | 0.9 GB | 9.3 GB | 10.0 GB | not run |

The first three lines of `ort-peak.log` come from an earlier version of the script, which always ran twice with the arena on. `BertSession` leaves the arena at its default, so a .NET caller is likely to see the second figure, but this was not measured.

## Not done

- The .NET path past 1,468 tokens. No corpus row reaches truncation.
- Peak memory for a batch of long rows, or for any length other than 8,192 tokens with ONNX Runtime alone.
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

Commands (from the repository root; paths shortened):

```sh
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
  .venv/bin/model-pipeline run model-pipeline/recipes/gte-reranker-modernbert-base-cpu.yaml \
    --snapshot <snapshot at f7481e60> --reference-python <encoder environment>/bin/python \
    --out artifacts/onboarding/gte-reranker-modernbert-base --threads 2 --timeout 1800
python -I docs/evidence/gte-reranker-modernbert-onboarding-2026-11-04/ort_peak.py <snapshot> 8192 4 arena
python -I docs/evidence/gte-reranker-modernbert-onboarding-2026-11-04/ort_peak.py <snapshot> 8192 2 noarena
```
