using Starlings.Acquisition;
using Starlings.Tokenizers;

namespace Starlings.Models.Bert.Tests;

/// <summary>
/// The route's tokenizer over the pinned gte-reranker-modernbert-base <c>tokenizer.json</c>, which declares truncation
/// to 8000 and Fixed 8000 padding: <see cref="BertSession.ForProfile"/> replaces both, so a pair is not padded and the
/// limit is the profile's 8192, as transformers does per call. The ids are those of huggingface/tokenizers 0.23.2. Skips
/// without the asset (tools/fetch-tokenizer-assets.sh provides it).
/// </summary>
public sealed class ModernBertTokenizerTests
{
    private const int Sep = 50282;
    private const int Pad = 50283;

    private static readonly BertModelProfile Profile = BertModelProfile.BuiltIn("gte-reranker-modernbert-base");

    private static string AssetPath()
    {
        string root = BuiltInProfileTests.RepositoryRoot();
        string models = Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(root, ".models");
        string path = Path.Combine(models, Profile.SourceLock.Repository, Profile.SourceLock.Revision, BertModelProfile.TokenizerFile);
        Assert.SkipUnless(File.Exists(path), $"{path} not found; run tools/fetch-tokenizer-assets.sh.");
        return path;
    }

    [Fact]
    public void TheFileItselfPadsToItsDeclaredLength()
    {
        // The hazard the route has to clear: without ForProfile a five-token pair comes back 8000 long.
        Assert.Equal(8000, Tokenizer.FromFile(AssetPath()).Encode("q", "d").Count);
    }

    [Fact]
    public void AShortPairIsNotPaddedAndHasNoSegmentIds()
    {
        var tokenizer = BertSession.ForProfile(Tokenizer.FromFile(AssetPath()), Profile.MaxTokens);

        var encoding = tokenizer.Encode("what is the capital of China?", "Beijing");

        Assert.Equal([50281, 5371, 310, 253, 5347, 273, 4135, 32, Sep, 4678, 16741, Sep], encoding.Ids);
        Assert.All(encoding.AttentionMask, value => Assert.Equal(1, value));
        Assert.All(encoding.TypeIds, value => Assert.Equal(0, value));
    }

    [Fact]
    public void ALongPairIsTruncatedToTheProfileLimitNotTheFiles()
    {
        var tokenizer = BertSession.ForProfile(Tokenizer.FromFile(AssetPath()), Profile.MaxTokens);
        string document = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog ", 1000));

        var encoding = tokenizer.Encode("how to implement quick sort in python?", document);

        Assert.Equal(8192, encoding.Count);
        Assert.Equal(Sep, encoding.Ids[^1]);
        Assert.DoesNotContain(Pad, encoding.Ids);
    }
}
