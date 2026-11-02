namespace Starlings.Models.Qwen3Embedding;

/// <summary>
/// Explicit support status of each capability of this package (plan section 9). A capability not listed here is not
/// supported. Nothing in this package is qualified: the onboarding evidence is development parity on CPU, and the
/// profiles are Experimental (starlinghq/starling#144).
/// </summary>
public static class Qwen3EmbeddingCapabilities
{
    /// <summary>Text embeddings (last-token pooling, L2, 1024 floats) on ONNX Runtime CPU, fp32.</summary>
    public const CapabilityStatus Embeddings = CapabilityStatus.Experimental;

    /// <summary>The named prompts <c>none</c>, <c>document</c> and <c>query</c>, and custom instructions (<see cref="Qwen3TextEmbedder.FormatQuery"/>).</summary>
    public const CapabilityStatus Prompts = CapabilityStatus.Experimental;

    /// <summary>Query-document reranking (raw log-odds <c>logit[yes] - logit[no]</c>) on ONNX Runtime CPU, fp32, with custom instructions.</summary>
    public const CapabilityStatus Reranking = CapabilityStatus.Experimental;

    /// <summary>Right-padded batches of several rows per ONNX Runtime call, with a row and an optional padded-cell budget.</summary>
    public const CapabilityStatus Batching = CapabilityStatus.Experimental;

    /// <summary>Inputs up to 8192 tokens; longer input is truncated from the right (the reranker keeps the template prefix and suffix and cuts the end of the instruction, query and document content, the document first).</summary>
    public const CapabilityStatus LongInput = CapabilityStatus.Experimental;

    /// <summary>Microsoft.Extensions.AI <c>IEmbeddingGenerator</c> (Starlings.Models.Qwen3Embedding.Extensions.AI).</summary>
    public const CapabilityStatus ExtensionsAI = CapabilityStatus.Experimental;

    /// <summary>
    /// Matryoshka (MRL) output dimensions below 1024. Not offered: a request for another dimension is rejected with
    /// <see cref="NotSupportedException"/>. Becomes Untested or Experimental once MRL slicing exists.
    /// </summary>
    public const CapabilityStatus MatryoshkaDimensions = CapabilityStatus.Unsupported;

    /// <summary>ONNX Runtime CUDA. Not offered: the sessions are CPU only. Becomes Untested or Experimental once a CUDA session path exists.</summary>
    public const CapabilityStatus Cuda = CapabilityStatus.Unsupported;

    /// <summary>
    /// Downloading the released directory from the public Starlinghq hosting repositories, pinned by commit, size and
    /// SHA-256 (<see cref="Qwen3EmbeddingModelAssets.DownloadAsync"/>).
    /// </summary>
    public const CapabilityStatus Download = CapabilityStatus.Experimental;

    /// <summary>fp16 or bf16 graphs.</summary>
    public const CapabilityStatus ReducedPrecision = CapabilityStatus.Unsupported;

    /// <summary>The 4B and 8B checkpoints.</summary>
    public const CapabilityStatus LargerCheckpoints = CapabilityStatus.Unsupported;

    /// <summary>Inputs longer than 8192 tokens without truncation.</summary>
    public const CapabilityStatus BeyondMaxTokens = CapabilityStatus.Unsupported;
}
