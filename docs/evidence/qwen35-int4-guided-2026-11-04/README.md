# Qwen3.5-0.8B int4 on CPU: guided decoding (issue #163)

Run on 2026-11-04. Until now `structured_output` on the Experimental profile `qwen3.5-0.8b/ortgenai-int4-cpu` was `untested`: hosting revision r2 put the guidance files in `ortgenai-int4/`, but no guided run had used the int4 graph (`docs/evidence/qwen35-hosting-r2-2026-11-02/README.md`, section 5). This run uses it. All 10 int4 checks passed, so the profile's `structured_output` moves to **experimental**.

Nothing here sets a tolerance, a gate, a golden or a task-quality floor; those are the maintainer's step 2 of #163. No `model-pipeline/protected/` file changed.

## 1. Environment

- Host: Intel Core i7-5820K (6 cores, AVX2, no AVX-512), 94 GB RAM, Linux x86-64, shared with CI runners (load average about 11 on 12 threads), so the decode times in the logs are not benchmarks.
- .NET runtime 10.0.12, Microsoft.ML.OnnxRuntimeGenAI 0.17.1 (CPU package, NuGet). No GPU: `CUDA_VISIBLE_DEVICES` was empty and the int4 graph is CPU only.
- Model files: `Starlinghq/qwen3.5-0.8b-ortgenai@22477cdb8a2dc71f4c98d2e6af0a046ec0d74f45` (r2), the anonymous download of the r2 evidence, read from its cache through `STARLING_MODELS`. Before the run, `sha256sum` of the five files in `ortgenai-int4/` matched the lock.
- Each test run was in a `systemd-run --user --scope` with `MemoryMax=20G` and `MemorySwapMax=0`.
- Code: `origin/main` at `b87c243bd4f32bf514a51472b1cfd4b8a570c4c1` plus this branch's test changes. Each log's first line records the run.

## 2. What ran

The checks are the ones fp32x and fp16x already pass. The fp32x classes were split into shared checks (`GuidanceCpuChecks`, `GuidedRealModelChecks`) plus one small class per variant, so int4 gets the same assertions, not relaxed copies. Greedy; outputs are checked against the constraint, not a reference. The one exception is the unguided check, which compares ids with the proposed int4 fixture.

| Log | Runs | Result |
| --- | --- | --- |
| `int4-guidance-cpu-tests.log` | `Int4GuidanceCpuModelTests`, typed API on int4: guidance files present, unguided greedy ids unchanged (`en-short`), regex, JSON schema, required tool grammar with and without fast-forward tokens, fast-forward text equality | 7 passed, 0 skipped |
| `int4-guided-chat-client-tests.log` | `Int4GuidedRealModelTests`, IChatClient on int4: `ToolMode` RequireAny, RequireSpecific, JSON `ResponseFormat` | 3 passed, 0 skipped |
| `fp32x-regression-tests.log` | `GuidanceCpuModelTests`, after the split: fp32x still passes | 7 passed, 0 skipped |
| `fp32x-chat-client-regression-tests.log` | `GuidedRealModelTests`, after the split: fp32x still passes | 3 passed, 0 skipped |
| `capability-tests.log` | `AssetAndCapabilityTests`, `StableContractTests`, after the status change: every generated capability record equals its manifest | 19 passed, 0 skipped |

## 3. Constrained outputs on int4

| Request | Constraint | Finish | Output |
| --- | --- | --- | --- |
| "Write a product code." | regex `FORCED-[0-9]{3}` | stop, 6 tokens | `FORCED-001` |
| "Name the capital of France and its country as JSON." | JSON schema (two required strings, no others) | stop, 12 tokens | `{"city": "Paris", "country": "France"}` |
| "What's the weather in Porto?" | required tool grammar, both fast-forward settings | stop, 40 tokens | `get_weather` with `city` Porto, `units` celsius |
| "Set a timer for 25 minutes." | required tool grammar, both fast-forward settings | stop, 27 tokens | `set_timer` with `minutes` 25; the same text in both runs |
| "What's the weather in Porto?" | IChatClient `RequireAny` over `get_weather`, `get_time` | `tool_calls` | `get_weather({"city":"Porto"})` |
| "What time is it in Porto?" | IChatClient `RequireSpecific("get_weather")` | `tool_calls` | `get_weather({"city":"Porto"})` |
| "What is 6 times 7? Reply as JSON." | IChatClient JSON schema `{"answer": integer}` | stop | `{"answer": 42}` |

The regex and JSON outputs differ from fp32x's (`FORCED-101`, and JSON with different whitespace), which is expected: int4 is a different graph, and it matches the transformers fp32 reference on 1 of 22 greedy sequences (M0). Correct values here are a handful of prompts, not a measure of int4's task quality.

## 4. What changed in status

- `qwen3.5-0.8b/ortgenai-int4-cpu`: `structured_output` untested to **experimental**, in the manifest and the generated `ProfileCapabilities.g.cs`, catalog and compatibility data. Not Supported: a handful of prompts per mode on one machine, no latency qualification, no gate, and no tolerance or task-quality floor for the profile.
- Package-level `structured_output` in `Qwen35Capabilities.All` was already Experimental; only its comment changed.
- Left as they were: `tool_calling` and `extensions_ai` stay **untested** on int4. The guided cases above touch both paths, but the rest of each (tool messages and tool-result turns, the function-invocation loop, streaming parsing, plain chat through the adapter) did not run on int4.

## 5. Not done here

- No tolerances or task-quality floor (#163 step 2, maintainer's protected change).
- No constraint check against a Python reference: the outputs were checked against the constraint only.
- No guided streaming run on int4, no Lark grammar other than the tool grammar, no long prompt, no adversarial schema.
- No latency measurement, Native AOT run, Windows run or gate.
