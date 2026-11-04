using Starlings.Acquisition;
using Starlings.Tokenizers;

namespace Starlings.Models.Bert.Tests;

/// <summary>
/// The pinned Starlings.Tokenizers loads the paraphrase-multilingual tokenizer.json (WhitespaceSplit in a Sequence with
/// Metaspace, bare Precompiled normalizer) and gives the ids of huggingface/tokenizers (recorded on #182). Skips without
/// the asset (tools/fetch-tokenizer-assets.sh provides it).
/// </summary>
public sealed class ParaphraseMultilingualTokenizerTests
{
    [Theory]
    [InlineData("Hello world", new[] { 0, 35378, 8999, 2 })]
    [InlineData("  Ünïcödé   spaces\tand\nnewlines ", new[] { 0, 8578, 19, 9392, 238, 21166, 446, 32628, 7, 136, 3525, 59801, 2 })]
    [InlineData("日本語のテキスト、句読点。", new[] { 0, 6, 98449, 154, 193508, 37, 27683, 55009, 2391, 30, 2 })]
    [InlineData("ﬁ ligature ① Ⅻ ＡＢＣ", new[] { 0, 809, 19881, 6644, 106, 39198, 47457, 2 })]
    [InlineData("Ein schöner Tag in München!", new[] { 0, 2991, 182319, 4837, 23, 43769, 38, 2 })]
    public void TokenIdsMatchTheReference(string text, int[] expected)
    {
        var profile = BertModelProfile.BuiltIn("paraphrase-multilingual");
        string root = BuiltInProfileTests.RepositoryRoot();
        string models = Environment.GetEnvironmentVariable(ModelAcquisitionOptions.CacheDirectoryEnvironmentVariable) is { Length: > 0 } env
            ? env
            : Path.Combine(root, ".models");
        string path = Path.Combine(models, profile.SourceLock.Repository, profile.SourceLock.Revision, BertModelProfile.TokenizerFile);
        Assert.SkipUnless(File.Exists(path), $"{path} not found; run tools/fetch-tokenizer-assets.sh.");

        var tokenizer = Tokenizer.FromFile(path);

        Assert.Equal(expected, tokenizer.Encode(text).Ids);
    }
}
