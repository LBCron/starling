# Starlings.Models.Phi4.OnnxGenAI

Text chat with [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) at revision `cfbefacb99257ffa30c83adab238a50856ac3083`, in-process on ONNX Runtime GenAI 0.17.1, CPU fp32.

This library is **experimental** and source only. Its admission is proposed, not decided (`model-pipeline/admissions/phi4-mini.md`). Whether this thin loader stays or gives way to one generic GenAI chat loader is an open maintainer question (`docs/design/generic-genai-chat-route.md`).

No hosted bundle exists, and nothing downloads one. Build it from the pinned snapshot with `model-pipeline/reference/genai-chat/build_bundle.py` (the `bundle` stage of `model-pipeline/recipes/phi4-mini-cpu.yaml`). `Phi4Model.Load` verifies all six files against `Phi4Profile.Mini.BundleFiles`, then loads with the pad overlay.

```csharp
using Starlings.Generation;
using Starlings.Models.Phi4;

using Phi4Model model = Phi4Model.Load(Phi4Profile.Mini, bundleDirectory);
GenerationResult result = model.Generate([ChatTurn.User("What is a river?")], new GenerationOptions { MaxNewTokens = 64 });
Console.WriteLine(result.Text);
```

Sizes: `model.onnx.data` is 15.3 GB, so a loaded model takes more than 15 GB of memory. One or two native threads.

Prompt plus completion should stay below 4,096 tokens: past that the long rope factors start, and no evidence covers them. The loader does not refuse a longer request; the status is untested, not unsupported. See `Phi4Capabilities` for every status.

Evidence: `docs/evidence/phi4-mini-onboarding-2026-11-04/README.md`.
