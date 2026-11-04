using System.Security.Cryptography;
using System.Text;

namespace Starlings.Generation.OnnxGenAI.Tests;

/// <summary>The generic loader's checks on an invented family and a small fake bundle: no model bytes, no native load.</summary>
public sealed class GenAIChatModelTests : IDisposable
{
    private const string Decoder = """{"model":{"decoder":{"hidden_size":8,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_size":4,"filename":"model.onnx"}}}""";

    private readonly string _directory = Directory.CreateTempSubdirectory("genai-chat-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private sealed class TokenizerReached : Exception;

    private sealed class FakeTemplate(string digest) : IChatTemplate
    {
        public string TemplateSha256 => digest;

        public bool SupportsThinking => false;

        public string Render(IReadOnlyList<ChatTurn> turns, IReadOnlyList<ToolDefinition>? tools = null, bool enableThinking = false) => "text";
    }

    private static ChatRouteContract Contract(string tools = "unsupported", string thinking = "unsupported") => ChatRouteContract.ParseTable($$"""
        {
          "format": "starlings.genai-chat-route-contracts", "format_version": 1, "route": "genai-chat-cpu-v1",
          "contracts": {
            "example/Tiny-Chat": {
              "id": "tiny-chat/ortgenai-fp32-cpu", "revision": "{{new string('a', 40)}}",
              "behavior": {
                "architecture": "TinyForCausalLM", "model_type": "tiny",
                "config": {
                  "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                  "intermediate_size": 16, "max_position_embeddings": 64, "original_max_position_embeddings": 64,
                  "partial_rotary_factor": 1.0, "rope_type": "default", "tie_word_embeddings": false
                },
                "vocab_size": 100, "eos_token_ids": [1], "add_bos_token": false,
                "pad_token_id_overlay": null, "pad_overlay_reason": null,
                "tokenizer_profile": "tiny-bpe", "template": "tiny-text-v1", "template_sha256": "{{new string('b', 64)}}",
                "max_new_tokens": 8, "max_context_tokens": 32, "cache": "fresh-request",
                "tools": "{{tools}}", "thinking": "{{thinking}}", "multimodal": "unsupported", "long_context": "untested"
              }
            }
          }
        }
        """)["example/Tiny-Chat"];

    private ChatModelDefinition Definition(string config = Decoder, string? templateDigest = null, string[]? lockedFiles = null)
    {
        var locked = new List<ChatBundleFile>();
        foreach ((string name, string text) in new[] { ("genai_config.json", config), ("tokenizer.json", "{}"), ("extra.bin", "x") })
        {
            File.WriteAllText(Path.Combine(_directory, name), text, new UTF8Encoding(false));
            if (lockedFiles is null || lockedFiles.Contains(name))
            {
                locked.Add(new ChatBundleFile(name, Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text))), text.Length));
            }
        }

        return new ChatModelDefinition(Contract(), new FakeTemplate(templateDigest ?? new string('b', 64)), locked, _ => throw new TokenizerReached());
    }

    [Fact]
    public void AnIntactBundleReachesTheTokenizerCheck() =>
        Assert.Throws<TokenizerReached>(() => GenAIChatModel.Load(Definition(), _directory));

    [Fact]
    public void RefusesAnotherTemplate() =>
        Assert.Throws<InvalidDataException>(() => GenAIChatModel.Load(Definition(templateDigest: new string('c', 64)), _directory));

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void RefusesAThreadCountOutsideTheBound(int threads) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => GenAIChatModel.Load(Definition(), _directory, new GenAIChatModelOptions { IntraOpThreads = threads }));

    [Theory]
    [InlineData("missing")]
    [InlineData("size")]
    [InlineData("digest")]
    public void RefusesABundleFileThatDiffersFromTheLock(string breakage)
    {
        ChatModelDefinition definition = Definition();
        string path = Path.Combine(_directory, "extra.bin");
        switch (breakage)
        {
            case "missing": File.Delete(path); break;
            case "size": File.WriteAllText(path, "xx"); break;
            default: File.WriteAllText(path, "y"); break;
        }

        Assert.Contains("extra.bin", Assert.Throws<InvalidDataException>(() => GenAIChatModel.Load(definition, _directory)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RefusesALockWithoutTheConfigurationOrTokenizer() =>
        Assert.Throws<InvalidDataException>(() => GenAIChatModel.Load(Definition(lockedFiles: ["extra.bin"]), _directory));

    [Theory]
    [InlineData("hidden_size\":8", "hidden_size\":9")]
    [InlineData("num_hidden_layers\":2", "num_hidden_layers\":3")]
    [InlineData("num_attention_heads\":2", "num_attention_heads\":3")]
    [InlineData("num_key_value_heads\":1", "num_key_value_heads\":2")]
    [InlineData("head_size\":4", "head_size\":5")]
    [InlineData("model.onnx", "other.onnx")]
    public void RefusesAGeometryThatDiffersFromTheContract(string from, string to) =>
        Assert.Throws<InvalidDataException>(() => GenAIChatModel.Load(Definition(Decoder.Replace(from, to, StringComparison.Ordinal)), _directory));

    [Fact]
    public void CapabilityCheckFollowsTheContract()
    {
        ToolDefinition[] tools = [ToolDefinition.Function("lookup")];
        Assert.Throws<NotSupportedException>(() => GenAIChatModel.CheckCapabilities(Contract(), tools, enableThinking: false));
        Assert.Throws<NotSupportedException>(() => GenAIChatModel.CheckCapabilities(Contract(), null, enableThinking: true));
        GenAIChatModel.CheckCapabilities(Contract(), null, enableThinking: false);
        GenAIChatModel.CheckCapabilities(Contract(tools: "experimental", thinking: "experimental"), tools, enableThinking: true);
    }
}
