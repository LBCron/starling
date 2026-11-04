# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Tokenizer

Every package that tokenizes now pins `Starlings.Tokenizers [0.3.5]` (0.3.4 in `0.5.0`), and every manifest's `processing.tokenizer.version` moved with the pin. `0.3.5` adds the `WhitespaceSplit` pre-tokenizer, the o200k `Split` regex and three experimental profiles (`phi4-bpe-chat`, `xlmr-unigram-paraphrase-multilingual`, `modernbert-bpe-gte-reranker`), none used by a package yet. Outputs of existing profiles are unchanged: the conformance fixtures they have pass unchanged, the profiles without fixtures reach no code that changes their output, and no model was rerun (`docs/support-policy.md`, "Tokenizer bumps", option B). A tokenizer with an `rstrip` added token and, in the same matching pass, an added token that starts with whitespace now fails at load instead of diverging from upstream; no pinned asset has that combination.

## Reranking sample

New sample `ai-dotnet/samples/E5RerankSearch`: multilingual-e5-small retrieves the top 5 from an in-memory `Microsoft.Extensions.VectorData` collection and `bge-reranker-v2-m3` reorders them with `Reranking.RerankAsync`, on the published 0.5.0 packages, with its lock file and recorded output. The helpers and the reranker profile stay Experimental: gates G0 to G7 have not run for `bge-reranker-v2-m3/onnx-fp32-cpu`. No package changes.
