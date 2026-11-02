using Starlings.Acquisition;

namespace Starlings.Models.Qwen3Embedding;

/// <summary>Session and batching options of <see cref="Qwen3TextEmbedder"/> and <see cref="Qwen3Reranker"/>.</summary>
public sealed record Qwen3EmbeddingOptions
{
    /// <summary>ONNX Runtime intra-op threads (1 to 64). Default 4.</summary>
    public int Threads { get; init; } = 4;

    /// <summary>Rows per ONNX Runtime call at most (1 to 128). Default 8.</summary>
    public int MaxBatchSize { get; init; } = 8;

    /// <summary>
    /// Padded cells (rows x longest row) per call at most, or null for no budget. A row longer than the budget runs
    /// alone. Default 8192: the graph uses eager attention, whose scores grow with the square of the row length (about
    /// 16 heads x 8192^2 x 4 bytes, roughly 4.3 GB per layer, for one 8192-token row), so a maximum-length row runs alone.
    /// A budget of 16,384 lets two 8192-token rows share a call and needs about twice that transient memory.
    /// </summary>
    public int? MaxBatchTokens { get; init; } = 8192;

    /// <summary>How the released directory is checked before loading. Default SHA-256 of every file (about 2.4 GB read).</summary>
    public CacheVerification Verification { get; init; } = CacheVerification.Sha256;

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(Threads, 1, nameof(Threads));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(Threads, 64, nameof(Threads));
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxBatchSize, 1, nameof(MaxBatchSize));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(MaxBatchSize, 128, nameof(MaxBatchSize));
        if (MaxBatchTokens is { } tokens)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(tokens, 1, nameof(MaxBatchTokens));
        }
    }
}

/// <summary>One tokenized input: token ids and attention mask of equal length (all mask 1 for an unpadded row).</summary>
/// <param name="InputIds">Token ids.</param>
/// <param name="AttentionMask">1 for each token.</param>
public sealed record Qwen3Input(ReadOnlyMemory<long> InputIds, ReadOnlyMemory<long> AttentionMask)
{
    /// <summary>Tokens in the row.</summary>
    public int Length => InputIds.Length;
}
