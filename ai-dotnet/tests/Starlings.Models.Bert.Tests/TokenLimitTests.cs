namespace Starlings.Models.Bert.Tests;

/// <summary>
/// The reranker's optional token limit, without a model: the bounds on the value, the refusal of a longer pre-tokenized
/// row, and its capability status. <see cref="ModernBertTokenizerTests"/> holds text truncation to the reference.
/// </summary>
public sealed class TokenLimitTests
{
    private static readonly BertModelProfile Profile = BertModelProfile.BuiltIn("gte-reranker-modernbert-base");

    private static BertInput Row(int length)
    {
        var ids = new long[length];
        ids[0] = Profile.ClsTokenId;
        ids[^1] = Profile.SepTokenId;
        return new BertInput(ids, Enumerable.Repeat(1L, length).ToArray(), new long[length]);
    }

    private static void Validate(BertInput row, int limit) =>
        BertSession.ValidateInput(row, limit, Profile.VocabularySize, Profile.PadTokenId, Profile.ClsTokenId, Profile.SepTokenId);

    [Theory]
    [InlineData(null)]
    [InlineData(1)]
    [InlineData(512)]
    [InlineData(8192)]
    public void TheLimitIsOptionalAndWithinTheProfiles(int? maxTokens) => BertSession.ValidateTokenLimit(maxTokens, 8192);

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(8193)]
    public void CreateRefusesALimitOutsideTheProfilesBeforeTouchingAssets(int maxTokens)
    {
        var refused = Assert.Throws<ArgumentOutOfRangeException>(() => BertReranker.Create(Profile, "no-such-snapshot", maxTokens: maxTokens));

        Assert.Equal("maxTokens", refused.ParamName);
    }

    [Theory]
    [InlineData(4, 3, false)]
    [InlineData(5, 3, true)]
    [InlineData(5, 4, false)]
    [InlineData(6, 4, true)]
    public void TheLimitLeavesOneTokenForEachText(int maxTokens, int pairSpecialTokens, bool allowed)
    {
        void Check() => BertSession.ValidateTokenFloor(maxTokens, pairSpecialTokens);

        if (allowed)
        {
            Check();
        }
        else
        {
            Assert.Equal("maxTokens", Assert.Throws<ArgumentOutOfRangeException>(Check).ParamName);
        }
    }

    [Fact]
    public void APreTokenizedRowOverTheLimitIsRefusedNotTruncated()
    {
        Validate(Row(16), 16);

        Assert.Throws<ArgumentException>(() => Validate(Row(17), 16));
    }

    [Theory]
    [InlineData("gte-reranker-modernbert-base/onnx-fp32-cpu")]
    [InlineData("ms-marco-minilm-l6-v2/onnx-fp32-cpu")]
    [InlineData("bge-reranker-v2-m3/onnx-fp32-cpu")]
    public void EveryRerankerProfileStatesTheCapability(string profile)
    {
        Assert.Equal("experimental", BertCapabilities.TokenLimit);
        Assert.Equal(BertCapabilities.TokenLimit, BertCapabilities.ProfileCapabilities[profile].Capabilities["token_limit"].ToString().ToLowerInvariant());
    }
}
