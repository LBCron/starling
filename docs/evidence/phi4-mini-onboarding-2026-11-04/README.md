# Phi-4-mini-instruct on ONNX Runtime GenAI, CPU fp32: onboarding evidence (2026-11-04)

Development evidence for `phi4-mini-instruct/ortgenai-fp32-cpu` (model request #109), from the `genai-chat-cpu-v1` onboarding route run for real on 2026-11-04 on Linux x64 (Intel Core i7-5820K, 12 logical CPUs, 94 GB RAM shared with other work), CPU only, two threads, inside a 20 GiB memory limit. It is not release qualification: no gate ran, no fixture is protected, and nothing here is binding. The route's design note, `docs/design/generic-genai-chat-route.md`, is still proposed.

| Candidate | Recipe | Result | Rows |
| --- | --- | --- | --- |
| `phi4-mini-instruct/ortgenai-fp32-cpu` | `recipes/phi4-mini-cpu.yaml` | **passed** (development parity; `report.json`, `status: passed`, no failures) | 14 of 14 |

## Inputs

- Upstream: `microsoft/Phi-4-mini-instruct` at `cfbefacb99257ffa30c83adab238a50856ac3083`. The `assets` stage checked every pinned file by size and SHA-256 (`source-lock.json`).
- Bundle: built by the `bundle` stage with the ONNX Runtime GenAI 0.17.1 builder from the pinned safetensors (`bundle-lock.json`). All six files passed `Phi4Model.Load`'s check against the lock in `Phi4Profile`, including `model.onnx` and the 15.3 GB `model.onnx.data`, so this build reproduced the digests pinned from an earlier build in another checkout.
- Reference environment: `.models/venv-generation` (`reference/generation/requirements.txt`, recorded in `environment.json`), transformers 5.18.0 with `trust_remote_code` off.
- Corpus: `model-pipeline/corpora/five-models/chat.jsonl`, 14 conversations.
- .NET: `Starlings.Models.Phi4`, `Starlings.Models.Phi4.OnnxGenAI` and `ai-dotnet/tools/Starlings.GenAIChat.Fidelity`, `Starlings.Tokenizers` 0.3.5 (the pin in `ai-dotnet/Directory.Packages.props`). `dotnet-environment.json` records the identities and the GenAI package the tool loaded.

## Result

| Comparison, against transformers | Largest difference | Row | Bound |
| --- | ---: | --- | ---: |
| Python GenAI, initial logits, max absolute | 2.14e-4 | `reasoning-content` | 5e-3 |
| Python GenAI, initial logits, mean absolute | 4.27e-5 | `reasoning-content` | 5e-4 |
| .NET, initial logits, max absolute | 2.14e-4 | `reasoning-content` | 5e-3 |
| .NET, initial logits, mean absolute | 4.27e-5 | `reasoning-content` | 5e-4 |

Prompt ids and greedy output ids were exact on every row for both backends. On all 14 rows the .NET run's seven checks were true: cache, isolation, stream, usage, cancellation, token stop and string stop (`dotnet-results.jsonl`).

The rows are short. The longest is `long-prefill`, 361 prompt tokens plus 5 completion tokens, and most completions stop at the corpus limit of 16 tokens or sooner (`dotnet-rows.log`). So this run does not exercise the route's 4,096-token bound, only stays below it. Long context stays `untested`.

The Phi-4 test project passes, 75 of 75 with none skipped (`phi4-tests.log`). That run had the pinned `tokenizer.json` available, so the template and tokenizer tests ran rather than skipped.

## Runs, times and memory

Two invocations of one run directory, each a transient user unit (`systemd-run --user --unit=phi4-onboarding -p MemoryMax=20G -p MemorySwapMax=0`). `onboarding-unit.log` has the output of both, and `run.json` the record of the second.

1. The first run completed `assets` through `backend` and failed at `dotnet`: the fidelity tool did not compile (`first-attempt-build-error.log`), because it read a `Phi4Model.Info` member that does not exist. CI failed on the same line.
2. The tool now writes its `environment.json` from the checkpoint and runtime constants. That file plays no part in the parity decision. The second run used `--resume`: `assets`, `environment`, `reference`, `bundle` and `backend` were verified and reused, and `dotnet`, `evaluate` and `review` ran.

So `reference`, `bundle` and `backend` ran at commit `68ad57d`, and `dotnet`, `evaluate` and `review` at `7cc7d2c`, which differs only in that tool file. `review.json` records `7cc7d2c`. The later commit that adds this directory changes documentation only.

A review follow-up after the run changes three things `run.json` hashed: `onboarding_genai_chat.py`, which declares more `dotnet` stage inputs, and the two Phi-4 library READMEs. No stage's behavior changed, and nothing was rerun. `run.json` keeps the hashes those files had at `7cc7d2c`, so they no longer match the branch tip, and the wider input list means a `--resume` of this run reruns its stages.

| Stage | Seconds |
| --- | ---: |
| `assets` | 70 |
| `environment` | 0 |
| `reference` | 798 |
| `bundle` | 274 |
| `backend` | 989 |
| `dotnet` | 1,389 |
| `evaluate` | 2 |
| `review` | 0 |

Total 3,522 s, about 59 minutes, on a machine with other work running; these are not benchmark numbers.

Memory was read from the unit's control group, which counts reclaimable file cache as well as process memory, so its peak is an upper bound and not the model's footprint. The second invocation peaked at 17.7 GiB. The first reached the 20 GiB limit while the bundle's files were in the page cache: the group's `max` event count rose, but `oom` and `oom_kill` were 0 at the last reading, during `backend`, and neither invocation was killed. During `reference` the group held 15.0 GiB. Process memory alone was not sampled.

## Not run

- Protected G1/G2 and release qualification; `review.json` records both as unapproved.
- Any context of 4,096 tokens or more, and any row near that bound.
- A clean-consumer drill or distribution stage: the .NET projects are source only.
- Windows, ARM64 and GPU execution.
