# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Tokenizer

Every package that tokenizes now pins `Starlings.Tokenizers [0.3.5]` (0.3.4 in `0.5.0`), and every manifest's `processing.tokenizer.version` moved with the pin. `0.3.5` adds the `WhitespaceSplit` pre-tokenizer, the o200k `Split` regex and three experimental profiles (`phi4-bpe-chat`, `xlmr-unigram-paraphrase-multilingual`, `modernbert-bpe-gte-reranker`), none used by a package yet. Outputs of existing profiles are unchanged: the conformance fixtures they have pass unchanged, the profiles without fixtures reach no code that changes their output, and no model was rerun (`docs/support-policy.md`, "Tokenizer bumps", option B). A tokenizer with an `rstrip` added token and, in the same matching pass, an added token that starts with whitespace now fails at load instead of diverging from upstream; no pinned asset has that combination.

## BERT route: paraphrase-multilingual-MiniLM-L12-v2 draft contract (#182, unsupported)

- `BertModelProfile.Load` and `Parse` accept an eighth repository, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (XLM-R Unigram tokenizer with BERT segment ids, mean pooling, no L2, 128 tokens, 384 wide; candidate `model-pipeline/candidates/paraphrase-multilingual`). Text embedding through it is **unsupported**: the pinned `Starlings.Tokenizers` 0.3.4 refuses the model's `WhitespaceSplit` pre-tokenizer, so `BertTextEmbedder.Create` throws `UnsupportedTokenizerComponentException`. It is not a built-in profile (`BuiltInIds` still lists seven), has no registry manifest and claims no maturity. A test pins the refusal, so the Tokenizers bump that fixes it is noticed.
- `model-pipeline scaffold` takes `--token-type-ids` / `--no-token-type-ids` to state whether the graph takes `token_type_ids`; without either flag an existing entry keeps its value, and a value that contradicts the entry is refused.
