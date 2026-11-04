# Starlings.Models.Phi4

The model definition of [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) at revision `cfbefacb99257ffa30c83adab238a50856ac3083`, without a runtime. Generation is in `Starlings.Models.Phi4.OnnxGenAI`.

This library is **experimental**, source only, and its admission is proposed, not decided (`model-pipeline/admissions/phi4-mini.md`).

- `Phi4ChatTemplate`: the checkpoint's `chat_template` rendered in C# for text turns (system, user, assistant).
- `Phi4Checkpoint`: repository and revision; the special token ids; the decoder shape; `Profile` (a `ModelProfile`); `Lock` (the asset lock of the Hub files a bundle is built from, about 7.7 GB of bf16 weights); `OpenTokenizer`, which verifies the pinned `tokenizer.json` and loads it with the owned tokenizer (`phi4-bpe-chat`, `Starlings.Tokenizers` 0.3.5).
- `Phi4Capabilities.All`: every capability with an explicit status.

```csharp
using Starlings.Generation;
using Starlings.Models.Phi4;

string text = Phi4ChatTemplate.Render([ChatTurn.System("Be brief."), ChatTurn.User("What is a river?")]);
var tokenizer = Phi4Checkpoint.OpenTokenizer(tokenizerPath);
int[] promptIds = [.. tokenizer.Encode(text, addSpecialTokens: false).Ids];
```

## Template notes

Text is inserted as given: a special token's text in a turn becomes that special token. The role tags have `rstrip` in `tokenizer.json`, so the whitespace after a tag does not reach the model. An assistant turn's reasoning is ignored, as the upstream template ignores it.

## Capabilities

- **Experimental:** the text template, the checkpoint lock, greedy CPU fp32 generation through `Starlings.Models.Phi4.OnnxGenAI`, streaming, cancellation, and stop tokens and sequences.
- **Untested:** prompt plus completion at or past 4,096 tokens, where the long rope factors start; sampling, logprobs and guided decoding on this checkpoint; AOT.
- **Unsupported (rejected or absent):** tools (the template's tool section, tool calls and tool turns), thinking, vision, CUDA, fp16, `Microsoft.Extensions.AI`.

Evidence: `docs/evidence/phi4-mini-onboarding-2026-11-04/README.md`.

## License

Phi-4-mini-instruct, Copyright (c) Microsoft Corporation, MIT (`LICENSE` in the Hub repository at the pinned commit). `Phi4ChatTemplate` is a C# port of its `chat_template` (`ai-dotnet/THIRD-PARTY-NOTICES.md`).
