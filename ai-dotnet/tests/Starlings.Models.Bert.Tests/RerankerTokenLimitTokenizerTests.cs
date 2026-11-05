using Starlings.Acquisition;
using Starlings.Tokenizers;

namespace Starlings.Models.Bert.Tests;

/// <summary>
/// The token limit over the pinned <c>tokenizer.json</c> of the reranker profiles: truncation against transformers and
/// the floor against the tokenizer's own special tokens. <see cref="ModernBertTokenizerTests"/> holds gte-reranker's
/// truncation. Skips without the assets (tools/fetch-tokenizer-assets.sh provides them).
/// </summary>
public sealed class RerankerTokenLimitTokenizerTests
{
    private const string Query = "how to implement quick sort in python?";

    private static Tokenizer Pinned(BertModelProfile profile, int limit)
    {
        string root = BuiltInProfileTests.RepositoryRoot();
        string models = Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(root, ".models");
        string path = Path.Combine(models, profile.SourceLock.Repository, profile.SourceLock.Revision, BertModelProfile.TokenizerFile);
        Assert.SkipUnless(File.Exists(path), $"{path} not found; run tools/fetch-tokenizer-assets.sh.");
        return BertSession.ForProfile(Tokenizer.FromFile(path), limit);
    }

    // transformers 5.18.0 at each profile's revision: tokenizer(query, document, truncation=True, max_length=16).
    public static TheoryData<string, int[]> LongDocumentAt16 => new()
    {
        { "ms-marco", [101, 2129, 2000, 10408, 4248, 4066, 1999, 102, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 102] },
        { "bge-reranker-v2-m3", [0, 3642, 47, 29479, 63773, 12096, 23, 2, 2, 70, 63773, 119455, 6, 147797, 88203, 2] },
    };

    public static TheoryData<string, int[]> LongQueryAt16 => new()
    {
        { "ms-marco", [101, 1053, 1053, 1053, 1053, 1053, 1053, 1053, 1053, 1053, 1053, 1053, 1053, 102, 1040, 102] },
        { "bge-reranker-v2-m3", [0, 8096, 8096, 8096, 8096, 8096, 8096, 8096, 8096, 8096, 8096, 8096, 2, 2, 104, 2] },
    };

    [Theory]
    [MemberData(nameof(LongDocumentAt16))]
    public void ALongDocumentTruncatesLikeTransformersMaxLength(string id, int[] expected)
    {
        var tokenizer = Pinned(BertModelProfile.BuiltIn(id), 16);
        string document = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog ", 1000));

        Assert.Equal(expected, tokenizer.Encode(Query, document).Ids);
    }

    [Theory]
    [MemberData(nameof(LongQueryAt16))]
    public void ALongQueryTruncatesLikeTransformersMaxLength(string id, int[] expected)
    {
        var tokenizer = Pinned(BertModelProfile.BuiltIn(id), 16);

        Assert.Equal(expected, tokenizer.Encode(string.Concat(Enumerable.Repeat("q ", 40)), "d").Ids);
    }

    [Theory]
    [InlineData("gte-reranker-modernbert-base", 3)]
    [InlineData("ms-marco", 3)]
    [InlineData("bge-reranker-v2-m3", 4)]
    public void TheFloorComesFromTheTokenizersPairSpecialTokens(string id, int pairSpecialTokens)
    {
        var profile = BertModelProfile.BuiltIn(id);
        int counted = Pinned(profile, profile.MaxTokens).WithTruncation(null).Encode(string.Empty, string.Empty).Count;
        int floor = counted + 2;

        Assert.Equal(pairSpecialTokens, counted);
        BertSession.ValidateTokenFloor(floor, counted);
        Assert.Throws<ArgumentOutOfRangeException>(() => BertSession.ValidateTokenFloor(floor - 1, counted));

        // At the floor both texts keep a token and the pair ends on its separator.
        var encoding = Pinned(profile, floor).Encode(Query, "the quick brown fox");
        Assert.Equal(floor, encoding.Count);
        Assert.Equal(profile.SepTokenId, encoding.Ids[^1]);
    }
}
