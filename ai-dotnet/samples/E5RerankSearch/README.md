# E5RerankSearch sample

Two-stage local search with the published Starling packages: multilingual-e5-small retrieves the top 5 from an in-memory `Microsoft.Extensions.VectorData` collection, then `bge-reranker-v2-m3` reorders those candidates with `Reranking.RerankAsync`. Requires the .NET 10 SDK. The in-memory connector is `CommunityToolkit.VectorData.InMemory`; it belongs only to this sample, not the Starling libraries.

From this directory:

```sh
dotnet build -c Release
dotnet run -c Release --no-build -- download
dotnet run -c Release --no-build -- search "Which river flows through Cologne?"
```

`download` explicitly acquires and verifies both pinned models: e5 (about 487 MB) and the reranker's `onnx/model.onnx`, `onnx/model.onnx_data` and `tokenizer.json` (about 2.3 GB, from Starling's ONNX export `Starlinghq/bge-reranker-v2-m3-onnx`, no remote code). `search` verifies the local caches and runs offline. Both accept `--cache <dir>`; otherwise they use `$STARLING_MODELS` or the user cache (`~/.cache/starlings/models` on Linux). A missing or corrupt cache fails rather than downloading during search. `StarlingsVersion` pins the published package version `0.5.0`, the first with the reranking helpers. The sample uses `PackageReference`s only, no project references, and needs `Microsoft.ML.OnnxRuntime` 1.30.0 (CPU). `packages.lock.json` records the dependency graph. To verify it, use `dotnet restore --locked-mode`.

## Maturity

Experimental, both the helpers and the reranker. `Reranking` in `Starlings.Models.Abstractions` is Experimental API, and `bge-reranker-v2-m3/onnx-fp32-cpu` is an Experimental profile with gates G0 to G7 not run, no G3 reranking dataset and no protected tolerances. The output below shows that this sample runs; it is not a measure of retrieval quality. multilingual-e5-small is Stable on CPU; see the [catalog](../../../docs/catalog/README.md).

## How it works

The record's string-valued `[VectorStoreVector(384)]` property provides `passage: ` text, and `SearchAsync` receives a `query: ` prefixed question, as in [E5VectorStore](../E5VectorStore/README.md). `reranker.RerankAsync` reads that search to the end, scores every candidate in one `Score` call and returns them by descending score. It takes the plain question, and the selector `hit => hit.Record.Text` gives it the plain passage: the e5 prefixes are for the embedder only.

The sample leaves out the `top` argument of `RerankAsync` so that it can print the first-stage order as well; a pipeline passes `top: 3` and uses the result. The first-stage cosine stays on `r.Item.Score`, and `r.Index` is the candidate's first-stage position. The reranker's scores are raw logits, comparable only within one call and not probabilities. Reranking cannot recover a passage that the embedder left out of the top-k.

Scoring is synchronous CPU work: with an asynchronous vector store it runs on the thread that resumed `RerankAsync`. See "Reranking retrieval results" in the [`Starlings.Models.Abstractions` README](../../src/Starlings.Models.Abstractions/README.md). Eight short passages and five candidates are a demonstration, so the sample reads no files and the collection is recreated each run. Scoring cost grows with the number of candidates and their length: the graph is much larger than e5's, and peak resident memory of one `search` here was about 1.8 GB.

## Recorded output

Actual stdout from Linux x64 CPU, .NET SDK 10.0.100, runtime 10.0.12, Starlings 0.5.0 and connector 1.0.1, on 2026-11-04. `download` ran first with e5 already cached and fetched the reranker from the Hub.

```text
Indexed 8 multilingual passages in memory.
Query: Which river flows through Cologne?
e5 top 5 (cosine):
  #1 0.8186  rhine: Der Rhein fließt durch Köln.
  #2 0.7749  cologne-cathedral: Der Kölner Dom ist das Wahrzeichen von Köln.
  #3 0.7542  seine: The Seine flows through Paris on its way to the English Channel.
  #4 0.7464  danube: Le Danube traverse Vienne et Budapest avant de rejoindre la mer Noire.
  #5 0.7312  sumida: 隅田川は東京を流れて東京湾に注ぐ。
bge-reranker-v2-m3 top 3 (logit, was e5 #):
  6.1152  (was #1)  rhine: Der Rhein fließt durch Köln.
  -3.2621  (was #4)  danube: Le Danube traverse Vienne et Budapest avant de rejoindre la mer Noire.
  -4.1970  (was #3)  seine: The Seine flows through Paris on its way to the English Channel.
```

In this run e5 already has the expected passage first. The reranker keeps it there with a much wider gap to the other candidates, and the Cologne Cathedral passage drops out of the top 3.

The same command with `"東京を流れる川はどれですか？"`:

```text
Indexed 8 multilingual passages in memory.
Query: 東京を流れる川はどれですか？
e5 top 5 (cosine):
  #1 0.8781  sumida: 隅田川は東京を流れて東京湾に注ぐ。
  #2 0.8194  tokyo: 東京は日本の首都です。
  #3 0.7424  rhine: Der Rhein fließt durch Köln.
  #4 0.7328  danube: Le Danube traverse Vienne et Budapest avant de rejoindre la mer Noire.
  #5 0.7170  seine: The Seine flows through Paris on its way to the English Channel.
bge-reranker-v2-m3 top 3 (logit, was e5 #):
  3.6001  (was #1)  sumida: 隅田川は東京を流れて東京湾に注ぐ。
  -5.6437  (was #2)  tokyo: 東京は日本の首都です。
  -6.5987  (was #5)  seine: The Seine flows through Paris on its way to the English Channel.
```
