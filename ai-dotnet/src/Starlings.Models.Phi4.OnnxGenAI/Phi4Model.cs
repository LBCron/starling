using System.Collections.Immutable;
using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Tokenizers;

namespace Starlings.Models.Phi4;

/// <summary>A rendered prompt and its token ids.</summary>
/// <param name="Text">The template's text.</param>
/// <param name="TokenIds">The owned tokenizer's ids, with no added special tokens.</param>
public sealed record Phi4Prompt(string Text, ImmutableArray<int> TokenIds);

/// <summary>Experimental CPU load settings; no provider fallback.</summary>
public sealed record Phi4ModelOptions
{
    /// <summary>Bounded native thread count, from 1 through 2.</summary>
    public int IntraOpThreads { get; init; } = 2;
}

/// <summary>Text-only generation over the pinned Phi-4-mini original and a hash-verified experimental CPU fp32 bundle.</summary>
public sealed class Phi4Model : IDisposable
{
    private readonly GenAIModel _backend;
    private readonly TextGenerator _generator;

    private Phi4Model(Phi4Profile profile, GenAIModel backend, Tokenizer tokenizer)
    {
        Profile = profile;
        _backend = backend;
        Tokenizer = tokenizer;
        _generator = new TextGenerator(backend, tokenizer);
    }

    /// <summary>The loaded profile.</summary>
    public Phi4Profile Profile { get; }

    /// <summary>Owned tokenizer; no foreign tokenizer is used in production.</summary>
    public Tokenizer Tokenizer { get; }

    /// <summary>Verify the bundle against the profile's pinned lock and load CPU fp32. No implicit download occurs.</summary>
    /// <exception cref="InvalidDataException">A bundle file or the GenAI configuration differs from the pin.</exception>
    public static Phi4Model Load(Phi4Profile profile, string bundleDirectory, Phi4ModelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentException.ThrowIfNullOrWhiteSpace(bundleDirectory);
        options ??= new Phi4ModelOptions();
        if (options.IntraOpThreads is < 1 or > 2)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "CPU profiles permit one or two native threads.");
        }

        string directory = Path.GetFullPath(bundleDirectory);
        VerifyBundle(profile, directory);
        Tokenizer tokenizer = Phi4Checkpoint.OpenTokenizer(Path.Combine(directory, "tokenizer.json"));

        var backend = GenAIModel.Load(directory, new GenAIModelOptions
        {
            Provider = GenAIExecutionProvider.Cpu,
            IntraOpThreads = options.IntraOpThreads,
            PadTokenId = Phi4Checkpoint.PadTokenIdOverlay,
        });
        try
        {
            if (backend.ModelType != Phi4Checkpoint.ModelType || backend.VocabularySize != Phi4Checkpoint.VocabularySize ||
                backend.ContextLength != Phi4Checkpoint.ContextLength ||
                !backend.EosTokenIds.Order().SequenceEqual(Phi4Checkpoint.Profile.EosTokenIds))
            {
                throw new InvalidDataException("GenAI configuration differs from the admitted architecture, vocabulary, or EOS contract.");
            }

            return new Phi4Model(profile, backend, tokenizer);
        }
        catch
        {
            backend.Dispose();
            throw;
        }
    }

    /// <summary>Render with the owned template and tokenize without added special tokens.</summary>
    public Phi4Prompt CreatePrompt(IReadOnlyList<ChatTurn> turns)
    {
        string text = Phi4ChatTemplate.Render(turns);
        return new Phi4Prompt(text, Tokenizer.Encode(text, addSpecialTokens: false).Ids);
    }

    /// <summary>Generate in a fresh independent request.</summary>
    public GenerationResult Generate(IReadOnlyList<ChatTurn> turns, GenerationOptions? options = null,
        CancellationToken cancellationToken = default) => Generate(CreatePrompt(turns), options, cancellationToken);

    /// <summary>Generate from a prepared prompt.</summary>
    public GenerationResult Generate(Phi4Prompt prompt, GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return _generator.Generate(prompt.TokenIds, options, cancellationToken);
    }

    /// <summary>Stream with the same fresh-request, stop and usage contract as non-streaming generation.</summary>
    public IEnumerable<GenerationUpdate> GenerateStreaming(Phi4Prompt prompt, GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return _generator.GenerateStreaming(prompt.TokenIds, options, cancellationToken);
    }

    /// <summary>Advanced diagnostic sequence; each call returns an independent disposable cache.</summary>
    public GenAISequence CreateSequence(GenAISequenceOptions options) => _backend.CreateSequence(options);

    /// <summary>Dispose the shared model; existing sequences retain native lifetime until disposed.</summary>
    public void Dispose() => _backend.Dispose();

    private static void VerifyBundle(Phi4Profile profile, string directory)
    {
        foreach (Phi4BundleFile pinned in profile.BundleFiles)
        {
            string path = Path.Combine(directory, pinned.Path);
            if (!File.Exists(path) || new FileInfo(path).Length != pinned.Size || Sha256(path) != pinned.Sha256)
            {
                throw new InvalidDataException($"Bundle asset mismatch: {pinned.Path}.");
            }
        }

        using JsonDocument config = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "genai_config.json")));
        JsonElement decoder = config.RootElement.GetProperty("model").GetProperty("decoder");
        if (decoder.GetProperty("hidden_size").GetInt32() != Phi4Checkpoint.HiddenSize ||
            decoder.GetProperty("num_hidden_layers").GetInt32() != Phi4Checkpoint.Layers ||
            decoder.GetProperty("num_attention_heads").GetInt32() != Phi4Checkpoint.AttentionHeads ||
            decoder.GetProperty("num_key_value_heads").GetInt32() != Phi4Checkpoint.KeyValueHeads ||
            decoder.GetProperty("head_size").GetInt32() != Phi4Checkpoint.HeadSize || decoder.GetProperty("filename").GetString() != "model.onnx")
        {
            throw new InvalidDataException("GenAI decoder geometry differs from the admitted original configuration.");
        }
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
