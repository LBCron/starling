# Phi-4-mini-instruct on ONNX Runtime GenAI, CPU fp32: onboarding evidence (2026-11-04)

Development evidence for `phi4-mini-instruct/ortgenai-fp32-cpu` (model request #109), from the `genai-chat-cpu-v1` onboarding route run for real on 2026-11-04 on Linux x64 (Intel Core i7-5820K, 12 logical CPUs, 94 GB RAM shared with other work), CPU only, two threads, inside a 20 GiB memory limit. It is not release qualification: no gate ran, no fixture is protected, and nothing here is binding. The design note `docs/design/generic-genai-chat-route.md` is accepted for the route shape and the generic loader; the admission level and hosting a bundle stay open.

This directory holds one fresh run, through the generic loader, of the commit named below. It replaces the earlier evidence for this candidate, which came from two invocations through the removed thin loader.

| Candidate | Recipe | Result | Rows |
| --- | --- | --- | --- |
| `phi4-mini-instruct/ortgenai-fp32-cpu` | `recipes/phi4-mini-cpu.yaml` | **passed** (development parity; `report.json`, `status: passed`, no failures) | 14 of 14 |

## Inputs

- Commit: `2356cdb962ee93ee6af3a7af16222750145a622e`, recorded in `review.json` and in `run.json`. The run started from a clean tree.
- Upstream: `microsoft/Phi-4-mini-instruct` at `cfbefacb99257ffa30c83adab238a50856ac3083`. The `assets` stage checked every pinned file by size and SHA-256 (`source-lock.json`).
- Bundle: built by the `bundle` stage with the ONNX Runtime GenAI 0.17.1 builder from the pinned safetensors (`bundle-lock.json`). All six files passed `GenAIChatModel.Load`'s check against `Phi4Checkpoint.BundleFiles`, including `model.onnx` and the 15.3 GB `model.onnx.data`, and `bundle-lock.json` is the same as the earlier build's, so the builder reproduced its digests.
- Reference environment: `.models/venv-generation` in the main checkout (`reference/generation/requirements.txt`, recorded in `environment.json`), transformers 5.18.0 with `trust_remote_code` off.
- Corpus: `model-pipeline/corpora/five-models/chat.jsonl`, 14 conversations.
- .NET: `Starlings.Generation.Abstractions`, `Starlings.Generation.OnnxGenAI` (`GenAIChatModel`), `Starlings.Models.Phi4` and `ai-dotnet/tools/Starlings.GenAIChat.Fidelity`, `Starlings.Tokenizers` 0.3.5 (the pin in `ai-dotnet/Directory.Packages.props`). `dotnet-environment.json` records the identities and the GenAI package the tool loaded.

## Result

| Comparison, against transformers | Largest difference | Row | Bound |
| --- | ---: | --- | ---: |
| Python GenAI, initial logits, max absolute | 2.14e-4 | `reasoning-content` | 5e-3 |
| Python GenAI, initial logits, mean absolute | 4.27e-5 | `reasoning-content` | 5e-4 |
| .NET, initial logits, max absolute | 2.14e-4 | `reasoning-content` | 5e-3 |
| .NET, initial logits, mean absolute | 4.27e-5 | `reasoning-content` | 5e-4 |

Prompt ids and greedy output ids were exact on every row for both backends. On all 14 rows the .NET run's seven checks were true: cache, isolation, stream, usage, cancellation, token stop and string stop (`dotnet-results.jsonl`). That is 98 checks and 28 comparisons, each row on two backends. `summarize_report.py` cannot read this route's report, which has no `checks` list, so these figures come from `report.json` directly.

The rows are short. The longest is `long-prefill`, 361 prompt tokens plus 5 completion tokens, and most completions stop at the corpus limit of 16 tokens or sooner (`dotnet-rows.log`). So this run does not exercise the route's 4,096-token bound, only stays below it. Long context stays `untested`.

The Phi-4 test project passes, 76 of 76 with none skipped (`phi4-tests.log`), with the pinned `tokenizer.json` available, so the template and tokenizer tests ran rather than skipped. It ran on a build of the same C# as the commit above.

## Run, times and memory

One invocation, in a transient user unit (`systemd-run --user --unit=phi4gen-onboarding -p MemoryMax=20G -p MemorySwapMax=0`), with no `--resume`. Every stage in `run.json` has `cache_status: executed`. The command exited 0: `onboarding-unit.log` ends with `ONBOARD-EXIT 0`.

| Stage | Seconds | Anonymous memory, peak |
| --- | ---: | ---: |
| `assets` | 156 | not sampled |
| `environment` | 0 | not sampled |
| `reference` | 1,115 | 15.3 GiB |
| `bundle` | 413 | 17.0 GiB |
| `backend` | 1,311 | 17.7 GiB |
| `dotnet` | 2,156 | 17.4 GiB |
| `evaluate` | 2 | not sampled |
| `review` | 0 | not sampled |

Total 5,153 s, about 86 minutes. The machine had other work running: three runner jobs and a load of about 21 on 12 cores at the start. These are not benchmark numbers.

`memory-samples.log` holds a reading of the unit's control group every 15 s, from 19:54:57 UTC, about 10 minutes after the start and during `reference`, to the end. Anonymous memory is the figure closest to process memory, and its highest sample is 17.7 GiB, in `backend`. Readings 15 s apart can miss a shorter peak. The control group's own peak was 15.3 GiB at the first reading, so nothing before that reading went higher.

The group's total, which counts reclaimable file cache, reached the 20 GiB limit in `bundle`, `backend` and `dotnet`. File cache reached 19.9 GiB. The `max` event count rose to 243,489, while `oom` and `oom_kill` stayed 0 and nothing was killed. A limit below 20 GiB was not tried. Anonymous memory left about 2.3 GiB of room under this one.

## Not run

- Protected G1/G2 and release qualification; `review.json` records both as unapproved.
- Any context of 4,096 tokens or more, and any row near that bound.
- A clean-consumer drill or distribution stage: the family package is source only.
- Windows, ARM64 and GPU execution.
