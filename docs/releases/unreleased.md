# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Starlings.Models.Abstractions: reranking helpers for vector-search pipelines (Experimental; #168)

New `Reranking` and `RerankedItem<T>(Item, Index, Score)` put an `IReranker` (today `BertReranker`, for example `bge-reranker-v2-m3`) after a first-stage retriever. Microsoft.Extensions.AI has no reranker abstraction, so the helpers take any sequence plus a text selector and add no package reference: `RerankAsync` reads an `IAsyncEnumerable<T>` such as a `Microsoft.Extensions.VectorData` `SearchAsync` top-k, scores it with one `Score` call and orders by descending score (ties in input order, optional `top`); `Rerank` is the synchronous form; `AsScoringFunction` wraps an `IReranker` as a `Func<string, IReadOnlyList<string>, Task<float[]>>`, and a `RerankAsync` overload takes any such scorer. `AsScoringFunction` scores synchronously and returns a completed task; `RerankAsync` validates its arguments synchronously and scores after the last candidate arrives, on the thread that resumed it. API only and Experimental; every reranker profile stays Experimental (the Stable `bge-reranker-v2-m3` profile is still open in #168).
