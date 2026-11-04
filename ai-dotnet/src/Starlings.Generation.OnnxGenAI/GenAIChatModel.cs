using System.Collections.Immutable;
using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Tokenizers;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>A rendered prompt and its token ids.</summary>
/// <param name="Text">The template's text.</param>
/// <param name="TokenIds">The owned tokenizer's ids, with no added special tokens.</param>
public sealed record GenAIChatPrompt(string Text, ImmutableArray<int> TokenIds);

/// <summary>Experimental CPU load settings; no provider fallback.</summary>
public sealed record GenAIChatModelOptions
{
    /// <summary>Bounded native thread count, from 1 through 2.</summary>
    public int IntraOpThreads { get; init; } = 2;
}

/// <summary>
/// Text chat generation for any checkpoint of the <c>genai-chat-cpu-v1</c> route, over a hash-verified CPU fp32 bundle:
/// everything specific to a model comes from its <see cref="ChatModelDefinition"/>. Experimental: the API may change while
/// the route has one consumer. Nothing downloads: the caller supplies the bundle directory.
/// </summary>
public sealed class GenAIChatModel : IDisposable
{
    private readonly GenAIModel _backend;
    private readonly TextGenerator _generator;

    private GenAIChatModel(ChatModelDefinition definition, GenAIModel backend, Tokenizer tokenizer)
    {
        Definition = definition;
        _backend = backend;
        Tokenizer = tokenizer;
        _generator = new TextGenerator(backend, tokenizer);
    }

    /// <summary>The loaded definition.</summary>
    public ChatModelDefinition Definition { get; }

    /// <summary>Owned tokenizer; no foreign tokenizer is used in production.</summary>
    public Tokenizer Tokenizer { get; }

    /// <summary>Verify the bundle against the definition's lock and contract, then load CPU fp32. No implicit download occurs.</summary>
    /// <exception cref="InvalidDataException">The template, a bundle file or the GenAI configuration differs from the definition.</exception>
    public static GenAIChatModel Load(ChatModelDefinition definition, string bundleDirectory, GenAIChatModelOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentException.ThrowIfNullOrWhiteSpace(bundleDirectory);
        options ??= new GenAIChatModelOptions();
        if (options.IntraOpThreads is < 1 or > 2)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "CPU profiles permit one or two native threads.");
        }

        ChatRouteContract contract = definition.Contract;
        if (definition.Template.TemplateSha256 != contract.TemplateSha256)
        {
            throw new InvalidDataException("The chat template port differs from the route contract.");
        }

        string directory = Path.GetFullPath(bundleDirectory);
        VerifyBundle(definition, directory);
        Tokenizer tokenizer = definition.OpenTokenizer(Path.Combine(directory, "tokenizer.json"));
        if (contract.EosTokenIds.Any(id => tokenizer.IdToToken(id) is null) ||
            (contract.PadTokenIdOverlay is int pad && tokenizer.IdToToken(pad) is not null))
        {
            throw new InvalidDataException("The tokenizer lacks an EOS id, or the pad overlay row holds a token.");
        }

        var backend = GenAIModel.Load(directory, new GenAIModelOptions
        {
            Provider = GenAIExecutionProvider.Cpu,
            IntraOpThreads = options.IntraOpThreads,
            PadTokenId = contract.PadTokenIdOverlay,
        });
        try
        {
            if (backend.ModelType != contract.ModelType || backend.VocabularySize != contract.VocabularySize ||
                backend.ContextLength != contract.ContextLength ||
                !backend.EosTokenIds.Order().SequenceEqual(contract.EosTokenIds.Order()))
            {
                throw new InvalidDataException("GenAI configuration differs from the contract's architecture, vocabulary, or EOS ids.");
            }

            return new GenAIChatModel(definition, backend, tokenizer);
        }
        catch
        {
            backend.Dispose();
            throw;
        }
    }

    /// <summary>Render with the owned template and tokenize without added special tokens.</summary>
    /// <exception cref="NotSupportedException">Tools or thinking, where the contract's status is <c>unsupported</c>.</exception>
    public GenAIChatPrompt CreatePrompt(IReadOnlyList<ChatTurn> turns, IReadOnlyList<ToolDefinition>? tools = null, bool enableThinking = false)
    {
        CheckCapabilities(Definition.Contract, tools, enableThinking);
        string text = Definition.Template.Render(turns, tools, enableThinking);
        return new GenAIChatPrompt(text, Tokenizer.Encode(text, addSpecialTokens: false).Ids);
    }

    /// <summary>Generate in a fresh independent request.</summary>
    public GenerationResult Generate(IReadOnlyList<ChatTurn> turns, GenerationOptions? options = null,
        CancellationToken cancellationToken = default) => Generate(CreatePrompt(turns), options, cancellationToken);

    /// <summary>Generate from a prepared prompt.</summary>
    public GenerationResult Generate(GenAIChatPrompt prompt, GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return _generator.Generate(prompt.TokenIds, options, cancellationToken);
    }

    /// <summary>Stream with the same fresh-request, stop and usage contract as non-streaming generation.</summary>
    public IEnumerable<GenerationUpdate> GenerateStreaming(GenAIChatPrompt prompt, GenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return _generator.GenerateStreaming(prompt.TokenIds, options, cancellationToken);
    }

    /// <summary>Advanced diagnostic sequence; each call returns an independent disposable cache.</summary>
    public GenAISequence CreateSequence(GenAISequenceOptions options) => _backend.CreateSequence(options);

    /// <summary>Dispose the shared model; existing sequences retain native lifetime until disposed.</summary>
    public void Dispose() => _backend.Dispose();

    internal static void CheckCapabilities(ChatRouteContract contract, IReadOnlyList<ToolDefinition>? tools, bool enableThinking)
    {
        if (tools is { Count: > 0 } && contract.Tools == "unsupported")
        {
            throw new NotSupportedException($"Tools are unsupported for {contract.Id}.");
        }

        if (enableThinking && contract.Thinking == "unsupported")
        {
            throw new NotSupportedException($"Thinking is unsupported for {contract.Id}.");
        }
    }

    private static void VerifyBundle(ChatModelDefinition definition, string directory)
    {
        foreach (string required in (string[])["genai_config.json", "tokenizer.json"])
        {
            if (!definition.BundleFiles.Any(file => file.Path == required))
            {
                throw new InvalidDataException($"The bundle lock has no {required}.");
            }
        }

        foreach (ChatBundleFile pinned in definition.BundleFiles)
        {
            string path = Path.Combine(directory, pinned.Path);
            if (!File.Exists(path) || new FileInfo(path).Length != pinned.Size || Sha256(path) != pinned.Sha256)
            {
                throw new InvalidDataException($"Bundle asset mismatch: {pinned.Path}.");
            }
        }

        ChatRouteContract contract = definition.Contract;
        using JsonDocument config = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "genai_config.json")));
        JsonElement decoder = config.RootElement.GetProperty("model").GetProperty("decoder");
        if (decoder.GetProperty("hidden_size").GetInt32() != contract.HiddenSize ||
            decoder.GetProperty("num_hidden_layers").GetInt32() != contract.Layers ||
            decoder.GetProperty("num_attention_heads").GetInt32() != contract.AttentionHeads ||
            decoder.GetProperty("num_key_value_heads").GetInt32() != contract.KeyValueHeads ||
            decoder.GetProperty("head_size").GetInt32() != contract.HeadSize || decoder.GetProperty("filename").GetString() != "model.onnx")
        {
            throw new InvalidDataException("GenAI decoder geometry differs from the route contract.");
        }
    }

    private static string Sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
