using System.Collections.ObjectModel;
using System.Text.Json;
using Starlings.Tokenizers;

namespace Starlings.Generation;

/// <summary>One file of a bundle lock.</summary>
/// <param name="Path">File name in the bundle directory.</param>
/// <param name="Sha256">Lower-case hex SHA-256 of the file.</param>
/// <param name="Size">Size in bytes.</param>
public sealed record ChatBundleFile(string Path, string Sha256, long Size);

/// <summary>
/// What a chat family supplies to a generic loader: its entry of the route contract table and the parts the table does
/// not hold. Experimental: the API may change while the route has one consumer.
/// </summary>
/// <param name="Contract">The repository's entry of the route contract table.</param>
/// <param name="Template">The owned template port; its digest must equal <see cref="ChatRouteContract.TemplateSha256"/>.</param>
/// <param name="BundleFiles">The bundle lock: every file a loader requires, with its digest and size.</param>
/// <param name="OpenTokenizer">Verifies the <c>tokenizer.json</c> at the given path and opens it with the owned tokenizer.</param>
public sealed record ChatModelDefinition(
    ChatRouteContract Contract,
    IChatTemplate Template,
    IReadOnlyList<ChatBundleFile> BundleFiles,
    Func<string, Tokenizer> OpenTokenizer);

/// <summary>
/// One entry of a data-driven chat route's contract table (<c>model-pipeline/routes/genai-chat-cpu-v1/contracts.json</c>):
/// a pinned checkpoint's decoder shape, special ids, template digest and capability statuses. The onboarding plugin reads
/// the same file. An entry is not a support claim. Experimental: the API may change while the route has one consumer.
/// </summary>
public sealed class ChatRouteContract
{
    private const string Format = "starlings.genai-chat-route-contracts";
    private const string Route = "genai-chat-cpu-v1";
    private const string ProfileSuffix = "/ortgenai-fp32-cpu";

    private static readonly string[] EntryFields = ["id", "revision", "behavior"];

    private static readonly string[] BehaviorFields =
    [
        "architecture", "model_type", "config", "vocab_size", "eos_token_ids", "add_bos_token", "pad_token_id_overlay",
        "pad_overlay_reason", "tokenizer_profile", "template", "template_sha256", "max_new_tokens", "max_context_tokens",
        "cache", "tools", "thinking", "multimodal", "long_context",
    ];

    private static readonly string[] ConfigFields =
    [
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size",
        "max_position_embeddings", "original_max_position_embeddings", "partial_rotary_factor", "rope_type", "tie_word_embeddings",
    ];

    private ChatRouteContract(string repository, string id, string revision, JsonElement behavior)
    {
        JsonElement config = behavior.GetProperty("config");
        JsonElement pad = behavior.GetProperty("pad_token_id_overlay");
        Repository = repository;
        Id = id;
        Revision = revision;
        Architecture = behavior.GetProperty("architecture").GetString()!;
        ModelType = behavior.GetProperty("model_type").GetString()!;
        HiddenSize = config.GetProperty("hidden_size").GetInt32();
        Layers = config.GetProperty("num_hidden_layers").GetInt32();
        AttentionHeads = config.GetProperty("num_attention_heads").GetInt32();
        KeyValueHeads = config.GetProperty("num_key_value_heads").GetInt32();
        HeadSize = config.GetProperty("head_dim").GetInt32();
        ContextLength = config.GetProperty("max_position_embeddings").GetInt32();
        VocabularySize = behavior.GetProperty("vocab_size").GetInt32();
        EosTokenIds = Array.AsReadOnly(behavior.GetProperty("eos_token_ids").EnumerateArray().Select(token => token.GetInt32()).ToArray());
        AddBosToken = behavior.GetProperty("add_bos_token").GetBoolean();
        PadTokenIdOverlay = pad.ValueKind == JsonValueKind.Null ? null : pad.GetInt32();
        TokenizerProfile = behavior.GetProperty("tokenizer_profile").GetString()!;
        Template = behavior.GetProperty("template").GetString()!;
        TemplateSha256 = behavior.GetProperty("template_sha256").GetString()!;
        MaxNewTokens = behavior.GetProperty("max_new_tokens").GetInt32();
        MaxContextTokens = behavior.GetProperty("max_context_tokens").GetInt32();
        Tools = behavior.GetProperty("tools").GetString()!;
        Thinking = behavior.GetProperty("thinking").GetString()!;
        Multimodal = behavior.GetProperty("multimodal").GetString()!;
        LongContext = behavior.GetProperty("long_context").GetString()!;
    }

    /// <summary>Hub repository, the table's key.</summary>
    public string Repository { get; }

    /// <summary>Portable profile id.</summary>
    public string Id { get; }

    /// <summary>Pinned full commit SHA.</summary>
    public string Revision { get; }

    /// <summary>The built-in transformers architecture of the reference.</summary>
    public string Architecture { get; }

    /// <summary>The <c>model_type</c> of <c>config.json</c> and of the GenAI configuration.</summary>
    public string ModelType { get; }

    /// <summary>Hidden-state width.</summary>
    public int HiddenSize { get; }

    /// <summary>Decoder layers.</summary>
    public int Layers { get; }

    /// <summary>Attention heads.</summary>
    public int AttentionHeads { get; }

    /// <summary>Key/value heads.</summary>
    public int KeyValueHeads { get; }

    /// <summary>Attention head width.</summary>
    public int HeadSize { get; }

    /// <summary>The checkpoint's context, prompt and completion together (<c>max_position_embeddings</c>).</summary>
    public int ContextLength { get; }

    /// <summary>Rows of the embedding and the logits.</summary>
    public int VocabularySize { get; }

    /// <summary>The ids that end a turn, as the table lists them.</summary>
    public IReadOnlyList<int> EosTokenIds { get; }

    /// <summary>Whether the tokenizer adds a BOS.</summary>
    public bool AddBosToken { get; }

    /// <summary>The pad id overlaid on the GenAI configuration; null keeps the bundle's.</summary>
    public int? PadTokenIdOverlay { get; }

    /// <summary>The owned tokenizer pipeline.</summary>
    public string TokenizerProfile { get; }

    /// <summary>Template id.</summary>
    public string Template { get; }

    /// <summary>SHA-256 of the upstream template string.</summary>
    public string TemplateSha256 { get; }

    /// <summary>Default completion limit of a corpus row.</summary>
    public int MaxNewTokens { get; }

    /// <summary>Prompt plus completion stay below this in the evidence; a loader does not enforce it.</summary>
    public int MaxContextTokens { get; }

    /// <summary>Status of tool calling: <c>supported</c>, <c>unsupported</c>, <c>experimental</c> or <c>untested</c>.</summary>
    public string Tools { get; }

    /// <summary>Status of the thinking switch, spelled as <see cref="Tools"/>.</summary>
    public string Thinking { get; }

    /// <summary>Status of multimodal input, spelled as <see cref="Tools"/>.</summary>
    public string Multimodal { get; }

    /// <summary>Status of context past <see cref="MaxContextTokens"/>, spelled as <see cref="Tools"/>.</summary>
    public string LongContext { get; }

    /// <summary>
    /// Parse the route contract table with the rules of <c>validate_contracts</c> in <c>onboarding_genai_chat.py</c>, apart
    /// from its tokenizer profile allow-list. Keys are repositories, in table order.
    /// </summary>
    /// <exception cref="InvalidDataException">The table is malformed or breaks a rule.</exception>
    public static IReadOnlyDictionary<string, ChatRouteContract> ParseTable(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        try
        {
            using JsonDocument document = JsonDocument.Parse(json);
            JsonElement root = document.RootElement;
            RequireFields(root, ["format", "format_version", "route", "contracts"], "Chat route contract table");
            if (!root.GetProperty("format").ValueEquals(Format) || !root.GetProperty("route").ValueEquals(Route) ||
                root.GetProperty("format_version").GetRawText() != "1")
            {
                throw new InvalidDataException($"Expected {Format} version 1 for {Route}.");
            }

            JsonElement entries = root.GetProperty("contracts");
            if (entries.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("contracts must be an object keyed by repository.");
            }

            var contracts = new Dictionary<string, ChatRouteContract>(StringComparer.Ordinal);
            foreach (JsonProperty entry in entries.EnumerateObject())
            {
                if (!contracts.TryAdd(entry.Name, ParseEntry(entry.Name, entry.Value)))
                {
                    throw new InvalidDataException($"Duplicate repository {entry.Name}.");
                }
            }

            return contracts.Count == 0
                ? throw new InvalidDataException("contracts must not be empty.")
                : new ReadOnlyDictionary<string, ChatRouteContract>(contracts);
        }
        catch (Exception exception) when (exception is JsonException or InvalidOperationException or FormatException)
        {
            throw new InvalidDataException($"Malformed chat route contract table: {exception.Message}", exception);
        }
    }

    private static ChatRouteContract ParseEntry(string repository, JsonElement entry)
    {
        string where = $"Chat route contract '{repository}'";
        string[] parts = repository.Split('/');
        if (parts.Length != 2 || parts[0].Length == 0 || parts[1].Length == 0)
        {
            throw new InvalidDataException($"{where}: the key must be a repository id <owner>/<name>.");
        }

        RequireFields(entry, EntryFields, where);
        RequireKinds(where, entry, JsonValueKind.String, "id", "revision");
        string revision = entry.GetProperty("revision").GetString()!;
        if (revision.Length != 40 || !IsLowerHex(revision))
        {
            throw new InvalidDataException($"{where}: revision must be a full commit SHA.");
        }

        string id = entry.GetProperty("id").GetString()!;
        if (!id.EndsWith(ProfileSuffix, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"{where}: id must name an ortgenai-fp32-cpu profile.");
        }

        JsonElement behavior = entry.GetProperty("behavior");
        RequireFields(behavior, BehaviorFields, where);
        JsonElement config = behavior.GetProperty("config");
        RequireFields(config, ConfigFields, where);
        RequireKinds(where, behavior, JsonValueKind.String, "architecture", "model_type", "tokenizer_profile", "template", "template_sha256", "cache",
            "tools", "thinking", "multimodal", "long_context");
        RequireIntegers(where, behavior, "vocab_size", "max_new_tokens", "max_context_tokens");
        RequireIntegers(where, config, "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "intermediate_size", "max_position_embeddings", "original_max_position_embeddings");
        RequireKinds(where, config, JsonValueKind.Number, "partial_rotary_factor");
        RequireKinds(where, config, JsonValueKind.String, "rope_type");
        foreach (JsonElement flag in new[] { behavior.GetProperty("add_bos_token"), config.GetProperty("tie_word_embeddings") })
        {
            if (flag.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            {
                throw new InvalidDataException($"{where}: add_bos_token and tie_word_embeddings must be booleans.");
            }
        }

        int vocabulary = behavior.GetProperty("vocab_size").GetInt32();
        JsonElement eos = behavior.GetProperty("eos_token_ids");
        if (eos.ValueKind != JsonValueKind.Array || eos.GetArrayLength() == 0 ||
            eos.EnumerateArray().Any(token => token.ValueKind != JsonValueKind.Number || !token.TryGetInt32(out int value) || value < 0 || value >= vocabulary))
        {
            throw new InvalidDataException($"{where}: eos_token_ids must be ids inside the vocabulary.");
        }

        JsonElement pad = behavior.GetProperty("pad_token_id_overlay");
        JsonElement reason = behavior.GetProperty("pad_overlay_reason");
        if (pad.ValueKind == JsonValueKind.Null)
        {
            if (reason.ValueKind != JsonValueKind.Null)
            {
                throw new InvalidDataException($"{where}: pad_overlay_reason must be null without a pad overlay.");
            }
        }
        else if (reason.ValueKind != JsonValueKind.String || pad.ValueKind != JsonValueKind.Number || !pad.TryGetInt32(out int padId) ||
            padId < 0 || padId >= vocabulary || eos.EnumerateArray().Any(token => token.GetInt32() == padId))
        {
            throw new InvalidDataException($"{where}: the pad overlay must be a row inside the vocabulary, outside the EOS ids, with a reason.");
        }

        string digest = behavior.GetProperty("template_sha256").GetString()!;
        if (digest.Length != 64 || !IsLowerHex(digest))
        {
            throw new InvalidDataException($"{where}: template_sha256 must be a lowercase SHA-256.");
        }

        foreach (string capability in (string[])["tools", "thinking", "multimodal", "long_context"])
        {
            if (behavior.GetProperty(capability).GetString() is not ("supported" or "unsupported" or "experimental" or "untested"))
            {
                throw new InvalidDataException($"{where}: {capability} needs an explicit status.");
            }
        }

        if (behavior.GetProperty("max_context_tokens").GetInt32() > config.GetProperty("original_max_position_embeddings").GetInt32())
        {
            throw new InvalidDataException($"{where}: max_context_tokens must stay inside the short rope factors.");
        }

        return new ChatRouteContract(repository, id, revision, behavior);
    }

    private static void RequireFields(JsonElement element, string[] fields, string where)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"{where}: expected an object.");
        }

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonProperty property in element.EnumerateObject())
        {
            if (!seen.Add(property.Name) || Array.IndexOf(fields, property.Name) < 0)
            {
                throw new InvalidDataException($"{where}: duplicate or unknown field '{property.Name}'.");
            }
        }

        if (seen.Count != fields.Length)
        {
            throw new InvalidDataException($"{where}: fields must be {string.Join(", ", fields)}.");
        }
    }

    private static void RequireKinds(string where, JsonElement element, JsonValueKind kind, params string[] fields)
    {
        foreach (string field in fields)
        {
            if (element.GetProperty(field).ValueKind != kind)
            {
                throw new InvalidDataException($"{where}: {field} must be a {kind}.");
            }
        }
    }

    private static void RequireIntegers(string where, JsonElement element, params string[] fields)
    {
        foreach (string field in fields)
        {
            if (element.GetProperty(field).ValueKind != JsonValueKind.Number || !element.GetProperty(field).TryGetInt32(out _))
            {
                throw new InvalidDataException($"{where}: {field} must be an integer.");
            }
        }
    }

    private static bool IsLowerHex(string text) => text.All(character => char.IsAsciiHexDigit(character) && !char.IsAsciiLetterUpper(character));
}
