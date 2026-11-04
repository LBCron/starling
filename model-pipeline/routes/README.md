# Route contract tables

A route contract table is the allow-list of an onboarding route as data: one entry per upstream repository the route
implements, with the processing contract its code must reproduce. Both implementations of the route read the same
file, so they cannot drift apart.

| File | Route | Read by |
| --- | --- | --- |
| [`bert-onnx-cpu-v1/contracts.json`](bert-onnx-cpu-v1/contracts.json) | `bert-onnx-cpu-v1` | `src/model_pipeline/onboarding_bert.py` (`load_contracts`, `validate_contracts`) and `ai-dotnet/src/Starlings.Models.Bert` (embedded resource `Starlings.Models.Bert.Contracts.json`; `BertModelProfile.SupportedRepositories`) |
| [`genai-chat-cpu-v1/contracts.json`](genai-chat-cpu-v1/contracts.json) | `genai-chat-cpu-v1` | `src/model_pipeline/onboarding_genai_chat.py` (`load_contracts`, `validate_contracts`); `Starlings.Models.Phi4.Tests` compares `Phi4Checkpoint` with it |

The other routes (e5, Qwen, Kev, Whisper) still keep their allow-lists in code.

## `bert-onnx-cpu-v1/contracts.json`

`format` is `starlings.bert-route-contracts`, `format_version` is `1` and `route` is `bert-onnx-cpu-v1`. `contracts` maps
each Hugging Face repository (`<owner>/<name>`) to an object with exactly these fields:

| Field | Meaning | Allowed values |
| --- | --- | --- |
| `task` | Typed API the profile serves | `embeddings` or `reranking` |
| `max_tokens` | Token limit, special tokens and prompt included | positive integer |
| `pooling` | Pooling of the graph output | `mean` or `cls` (embeddings); `none` (reranking) |
| `hidden_size` | Width of the pooled output | positive integer |
| `normalize` | Vector normalization | `l2` or `none` (embeddings); `none` (reranking) |
| `vocabulary_size` | Exclusive upper bound of token ids | positive integer |
| `cls_token_id`, `sep_token_id`, `pad_token_id` | First, separator and right-padding ids | inside the vocabulary |
| `output_name` | Graph output to pool | non-empty string |
| `tokenizer_profile` | Owned tokenizer pipeline | `bert-wordpiece-uncased` or `xlmr-unigram` |
| `token_type_ids` | Whether the graph takes segment ids | boolean |
| `prompts` | Named prefixes | object of strings that includes `"none": ""` |
| `score_activation_source` (optional, reranking only) | Where the reference reads the score activation | `config` (default) or `explicit` |

Both readers refuse a table with another format, a duplicate key, a missing or unknown field, a value of the wrong
type or any value outside the column above, and the .NET package fails on first use rather than serve a profile from
a malformed table. A candidate's `behavior` must then equal its entry field for field (`validate_candidate_behavior`
and `BertModelProfile.Parse`).

Add an entry with `model-pipeline scaffold <candidate>` ([ONBOARDING.md](../ONBOARDING.md#scaffold-a-bert-route-candidate)),
not by hand. An entry is not a support claim: it lets the route run the candidate; gates, maturity and protected
evidence are unchanged.

## `genai-chat-cpu-v1/contracts.json`

`format` is `starlings.genai-chat-route-contracts`, `format_version` is `1` and `route` is `genai-chat-cpu-v1`. `contracts` maps each Hugging Face repository to an object with exactly `id` (the profile id, ending in `/ortgenai-fp32-cpu`), `revision` (full commit SHA) and `behavior`. A candidate's `behavior` must equal the entry's field for field.

| `behavior` field | Meaning | Allowed values |
| --- | --- | --- |
| `architecture`, `model_type` | The built-in transformers architecture and config type | strings; the reference refuses a class outside transformers |
| `config` | Exactly `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `intermediate_size`, `max_position_embeddings`, `original_max_position_embeddings`, `partial_rotary_factor`, `rope_type`, `tie_word_embeddings` | checked against the checkpoint's `config.json` by the bundle stage |
| `vocab_size` | Rows of the embedding and the logits | positive integer |
| `eos_token_ids` | The ids that end a turn, from `generation_config.json` | non-empty, inside the vocabulary |
| `add_bos_token` | Whether the tokenizer adds a BOS | boolean |
| `pad_token_id_overlay`, `pad_overlay_reason` | The pad id overlaid on the GenAI configuration, and why | an embedding row outside the EOS ids; the backend stage refuses a row the tokenizer uses |
| `tokenizer_profile` | Owned tokenizer pipeline | `phi4-bpe-chat` |
| `template`, `template_sha256` | Template id and SHA-256 of the upstream `chat_template` string | lowercase hex digest; the bundle stage checks it |
| `max_new_tokens` | Default completion limit of a corpus row | positive integer |
| `max_context_tokens` | Prompt plus completion must stay below this | at most `original_max_position_embeddings` |
| `cache` | Cache statement of the evidence | string |
| `tools`, `thinking`, `multimodal`, `long_context` | Explicit capability status | `supported`, `unsupported`, `experimental` or `untested` |

An entry is not a support claim: it lets the route run the candidate; gates, maturity and protected evidence are unchanged. Unlike the BERT table, no .NET package reads this one at run time: whether one generic loader should is an open maintainer question (`docs/design/generic-genai-chat-route.md`).
