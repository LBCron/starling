using Starlings.Acquisition;

namespace Starlings.Models.Qwen3Embedding;

/// <summary>
/// Explicit acquisition of a built-in profile's released directory from its public hosting repository
/// (<see cref="Qwen3EmbeddingProfile.HostingRepository"/> at <see cref="Qwen3EmbeddingProfile.HostingRevision"/>). The
/// embedder and the reranker never download: call <see cref="DownloadAsync"/> once (setup step), then
/// <see cref="OpenLocal"/> at startup and pass <see cref="ModelSnapshot.Directory"/> to
/// <see cref="Qwen3TextEmbedder.Create"/> or <see cref="Qwen3Reranker.Create"/>. About 2.4 GB per profile.
/// </summary>
/// <remarks>
/// Every file is pinned by size and SHA-256 in <see cref="Qwen3EmbeddingProfile.RuntimeLock"/>: the digest, not the
/// address, pins the bytes. The hosting repositories are public, so no Hugging Face token is needed.
/// </remarks>
public static class Qwen3EmbeddingModelAssets
{
    /// <summary>The embedding profile's files as an <see cref="IModelAssets{TFiles}"/>.</summary>
    public static IModelAssets<ModelSnapshot> Embedding06B { get; } = new AssetsInstance(Qwen3EmbeddingProfile.Embedding06B);

    /// <summary>The reranker profile's files as an <see cref="IModelAssets{TFiles}"/>.</summary>
    public static IModelAssets<ModelSnapshot> Reranker06B { get; } = new AssetsInstance(Qwen3EmbeddingProfile.Reranker06B);

    /// <summary>
    /// Explicitly downloads (or verifies already cached) the released directory of <paramref name="profile"/>. See
    /// <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/>.
    /// </summary>
    /// <param name="profile">A built-in profile; <see cref="Qwen3EmbeddingProfile.Embedding06B"/> when null.</param>
    /// <param name="options">Cache directory, endpoint and verification.</param>
    /// <param name="progress">Optional per-file progress.</param>
    /// <param name="cancellationToken">Cancels the download.</param>
    public static async Task<ModelSnapshot> DownloadAsync(
        Qwen3EmbeddingProfile? profile = null, ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        profile ??= Qwen3EmbeddingProfile.Embedding06B;
        using var store = new ModelAssetStore(options);
        return await store.DownloadAsync(profile.RuntimeLock, progress, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Offline: the cached files of <paramref name="profile"/>, verified per <see cref="ModelAcquisitionOptions.Verification"/>. Never downloads.</summary>
    /// <exception cref="ModelAssetMissingException">Not acquired yet; the message says how to download.</exception>
    /// <exception cref="ModelAssetIntegrityException">A cached file is corrupt.</exception>
    public static ModelSnapshot OpenLocal(Qwen3EmbeddingProfile? profile = null, ModelAcquisitionOptions? options = null)
    {
        profile ??= Qwen3EmbeddingProfile.Embedding06B;
        using var store = new ModelAssetStore(options);
        return store.OpenLocal(profile.RuntimeLock);
    }

    private sealed class AssetsInstance(Qwen3EmbeddingProfile profile) : IModelAssets<ModelSnapshot>
    {
        public ModelAssetLock Lock => profile.RuntimeLock;

        public Task<ModelSnapshot> DownloadAsync(ModelAcquisitionOptions? options = null, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default) =>
            Qwen3EmbeddingModelAssets.DownloadAsync(profile, options, progress, cancellationToken);

        public ModelSnapshot OpenLocal(ModelAcquisitionOptions? options = null) => Qwen3EmbeddingModelAssets.OpenLocal(profile, options);

        public ModelSnapshot OpenDirectory(string directory, CacheVerification verification = CacheVerification.Sha256) =>
            profile.OpenDirectory(directory, verification);
    }
}
