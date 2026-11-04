using System.Security.Cryptography;
using Starlings.Acquisition;
using Starlings.Generation;
using Starlings.Tokenizers;

namespace Starlings.Models.Phi4;

/// <summary>
/// The pinned <c>microsoft/Phi-4-mini-instruct</c> checkpoint, independent of a runtime: repository and commit, the
/// special token ids the template and the owned tokenizer use, the decoder shape from <c>config.json</c>, its
/// <see cref="ModelProfile"/>, the asset lock and <see cref="ChatDefinition"/> for a generic chat loader.
/// <c>model-pipeline/routes/genai-chat-cpu-v1/contracts.json</c> holds the same values; a test compares them.
/// </summary>
public static class Phi4Checkpoint
{
    private const string ContractsResource = "Starlings.Models.Phi4.Contracts.json";

    private static readonly Lazy<ChatModelDefinition> _chatDefinition = new(LoadChatDefinition);

    /// <summary>Hub repository.</summary>
    public const string Repository = "microsoft/Phi-4-mini-instruct";

    /// <summary>Pinned full commit SHA.</summary>
    public const string Revision = "cfbefacb99257ffa30c83adab238a50856ac3083";

    /// <summary>The <c>Starlings.Tokenizers</c> profile of the checkpoint's <c>tokenizer.json</c>.</summary>
    public const string TokenizerProfile = "phi4-bpe-chat";

    /// <summary>SHA-256 of the pinned <c>tokenizer.json</c>.</summary>
    public const string TokenizerSha256 = "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea";

    /// <summary>The <c>model_type</c> of <c>config.json</c> and of the GenAI configuration.</summary>
    public const string ModelType = "phi3";

    /// <summary>Rows of the embedding and the logits (the tokenizer's ids end at 200028).</summary>
    public const int VocabularySize = 200064;

    /// <summary><c>&lt;|endoftext|&gt;</c>, also the upstream BOS and pad id.</summary>
    public const int EndOfTextTokenId = 199999;

    /// <summary><c>&lt;|assistant|&gt;</c>.</summary>
    public const int AssistantTokenId = 200019;

    /// <summary><c>&lt;|end|&gt;</c>.</summary>
    public const int EndTokenId = 200020;

    /// <summary><c>&lt;|user|&gt;</c>.</summary>
    public const int UserTokenId = 200021;

    /// <summary><c>&lt;|system|&gt;</c>.</summary>
    public const int SystemTokenId = 200022;

    /// <summary>
    /// An embedding row with no token, used as the pad id: the upstream pad id is <c>&lt;|endoftext|&gt;</c>, and
    /// ONNX Runtime GenAI would drop a literal one from a prompt's attention.
    /// </summary>
    public const int PadTokenIdOverlay = 200063;

    /// <summary>Hidden-state width.</summary>
    public const int HiddenSize = 3072;

    /// <summary>Decoder layers.</summary>
    public const int Layers = 32;

    /// <summary>Attention heads.</summary>
    public const int AttentionHeads = 24;

    /// <summary>Key/value heads.</summary>
    public const int KeyValueHeads = 8;

    /// <summary>Attention head width.</summary>
    public const int HeadSize = 128;

    /// <summary>The checkpoint's context, prompt and completion together (<c>max_position_embeddings</c>).</summary>
    public const int ContextLength = 131072;

    /// <summary>
    /// Positions below this use the short rope factors (<c>original_max_position_embeddings</c>). The evidence covers
    /// only these; longer context is untested.
    /// </summary>
    public const int ShortFactorPositions = 4096;

    /// <summary>The checkpoint's identity and decoder shape; EOS ids from <c>generation_config.json</c>.</summary>
    public static ModelProfile Profile { get; } = new()
    {
        Id = "phi4-mini-instruct",
        Repository = Repository,
        Revision = Revision,
        TokenizerProfile = TokenizerProfile,
        TokenizerSha256 = TokenizerSha256,
        ChatTemplateSha256 = Phi4ChatTemplate.TemplateSha256,
        EosTokenIds = [EndOfTextTokenId, EndTokenId],
        VocabularySize = VocabularySize,
        HiddenSize = HiddenSize,
        Layers = Layers,
        HeadSize = HeadSize,
        ContextLength = ContextLength,
        License = "MIT",
    };

    /// <summary>
    /// The asset lock of the Hub files a GenAI bundle is built from, at <see cref="Revision"/> (about 7.7 GB of bf16
    /// weights). The checkpoint's Python modeling code is not locked: nothing here runs it. This package does not
    /// download it; pass it to
    /// <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/>.
    /// </summary>
    public static ModelAssetLock Lock { get; } = new("phi4-mini-instruct-checkpoint", Repository, Revision,
    [
        new ModelAssetFile("added_tokens.json", 249, "d4f2aceb0f20b71dd1f4bcc7e052e4412946bf281840b8f83d39f259571af486"),
        new ModelAssetFile("config.json", 2_504, "ac65d86061d3d0d704ee2511fd0eb8713ef19eb6eedba17c3080a4165d5b933b"),
        new ModelAssetFile("generation_config.json", 168, "3e3f48753753f92d2b958679151861d4fd7bf26e4dfc41fd47056116c4914dcd"),
        new ModelAssetFile("merges.txt", 2_418_348, "856ce61180bb689282eed6b3a6838bb1f438399be23aefe9d20eb379791fb4ad"),
        new ModelAssetFile("model-00001-of-00002.safetensors", 4_903_637_712, "bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6"),
        new ModelAssetFile("model-00002-of-00002.safetensors", 2_768_428_504, "7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983"),
        new ModelAssetFile("model.safetensors.index.json", 16_273, "613a98d5e5716ca96fa75931abedc9c5a5d95f488ce4d62df71e639fe3ac6c59"),
        new ModelAssetFile("special_tokens_map.json", 587, "aff38493227d813e29fcf8406e8e90062f1f031aa47d589325e9c31d89ac7cc3"),
        new ModelAssetFile("tokenizer.json", 15_524_095, TokenizerSha256),
        new ModelAssetFile("tokenizer_config.json", 2_932, "9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db"),
        new ModelAssetFile("vocab.json", 3_910_310, "6cb65a857824fa6615bb1782d95d882617a8bbce1da0317118586b36f39e98bd"),
    ]);

    /// <summary>
    /// The ONNX Runtime GenAI bundle lock: the six files the pinned builder produces from the checkpoint
    /// (<c>model-pipeline/reference/genai-chat/build_bundle.py</c>). No hosted copy exists: a user builds the bundle, and
    /// the loader verifies it against this lock.
    /// </summary>
    public static IReadOnlyList<ChatBundleFile> BundleFiles { get; } = Array.AsReadOnly(new ChatBundleFile[]
    {
        new("genai_config.json", "9187e9d71430a2ae23bdeb04a4042920f4e8fcfbe49c3fe9ae046eff6a1b329b", 1821),
        new("model.onnx", "b6b05c0d65000207940950184db3d29d8e84052470dac52f0f1d89df5555efb3", 52088021),
        new("model.onnx.data", "19aae69a1272c1bbf3a13935b371476700d5b12eb8290ee282631a39d8f67430", 15344140288),
        new("tokenizer.json", TokenizerSha256, 15524095),
        new("tokenizer_config.json", "9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db", 2932),
        new("chat_template.jinja", "febf589225c9728ab791f52e8897d7607a823d45368f0a4c92fa68997b40cce9", 423),
    });

    /// <summary>
    /// What a generic chat loader needs: this checkpoint's entry of the route contract table (embedded in the assembly),
    /// the template port, the bundle lock and the tokenizer check. Read on first use: a malformed table is a packaging
    /// defect and throws <see cref="InvalidDataException"/>.
    /// </summary>
    public static ChatModelDefinition ChatDefinition => _chatDefinition.Value;

    /// <summary>Verify and open the pinned <c>tokenizer.json</c> without loading model weights.</summary>
    /// <exception cref="InvalidDataException">The file's bytes or token identities differ from the pin.</exception>
    public static Tokenizer OpenTokenizer(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using (FileStream stream = File.OpenRead(path))
        {
            if (Convert.ToHexStringLower(SHA256.HashData(stream)) != TokenizerSha256)
            {
                throw new InvalidDataException("Tokenizer bytes differ from the admitted original.");
            }
        }

        Tokenizer tokenizer = Tokenizer.FromFile(path);
        if (tokenizer.TokenToId("<|endoftext|>") != EndOfTextTokenId || tokenizer.TokenToId("<|end|>") != EndTokenId ||
            tokenizer.TokenToId("<|assistant|>") != AssistantTokenId || tokenizer.TokenToId("<|user|>") != UserTokenId ||
            tokenizer.TokenToId("<|system|>") != SystemTokenId || tokenizer.IdToToken(PadTokenIdOverlay) is not null)
        {
            throw new InvalidDataException("Original tokenizer identity or unused padding row differs.");
        }

        return tokenizer;
    }

    private static ChatModelDefinition LoadChatDefinition()
    {
        using Stream stream = typeof(Phi4Checkpoint).Assembly.GetManifestResourceStream(ContractsResource)
            ?? throw new InvalidDataException($"The route contract table '{ContractsResource}' is missing from the assembly.");
        using var reader = new StreamReader(stream);
        return new ChatModelDefinition(ChatRouteContract.ParseTable(reader.ReadToEnd())[Repository], Phi4ChatTemplate.Instance, BundleFiles, OpenTokenizer);
    }
}
