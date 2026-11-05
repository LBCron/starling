using System.Text.Json;
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;

namespace Starlings.Models.Phi4.Tests;

/// <summary>The definition against the route's contract table, and the loader's refusals (no model bytes needed).</summary>
public sealed class Phi4DefinitionTests
{
    [Fact]
    public void DefinitionEqualsTheRouteContract()
    {
        using JsonDocument table = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Data", "contracts.json")));
        JsonElement entry = table.RootElement.GetProperty("contracts").GetProperty(Phi4Checkpoint.Repository);
        JsonElement behavior = entry.GetProperty("behavior");
        JsonElement config = behavior.GetProperty("config");

        Assert.Equal(Phi4Checkpoint.ChatDefinition.Contract.Id, entry.GetProperty("id").GetString());
        Assert.Equal(Phi4Checkpoint.Revision, entry.GetProperty("revision").GetString());
        Assert.Equal(Phi4Checkpoint.ModelType, behavior.GetProperty("model_type").GetString());
        Assert.Equal(Phi4Checkpoint.VocabularySize, behavior.GetProperty("vocab_size").GetInt32());
        Assert.Equal(Phi4Checkpoint.PadTokenIdOverlay, behavior.GetProperty("pad_token_id_overlay").GetInt32());
        Assert.Equal(Phi4Checkpoint.TokenizerProfile, behavior.GetProperty("tokenizer_profile").GetString());
        Assert.Equal(Phi4ChatTemplate.TemplateSha256, behavior.GetProperty("template_sha256").GetString());
        Assert.Equal(Phi4Checkpoint.ShortFactorPositions, behavior.GetProperty("max_context_tokens").GetInt32());
        Assert.Equal(Phi4Checkpoint.Profile.EosTokenIds, behavior.GetProperty("eos_token_ids").EnumerateArray().Select(e => e.GetInt32()));
        Assert.Equal(Phi4Checkpoint.HiddenSize, config.GetProperty("hidden_size").GetInt32());
        Assert.Equal(Phi4Checkpoint.Layers, config.GetProperty("num_hidden_layers").GetInt32());
        Assert.Equal(Phi4Checkpoint.AttentionHeads, config.GetProperty("num_attention_heads").GetInt32());
        Assert.Equal(Phi4Checkpoint.KeyValueHeads, config.GetProperty("num_key_value_heads").GetInt32());
        Assert.Equal(Phi4Checkpoint.HeadSize, config.GetProperty("head_dim").GetInt32());
        Assert.Equal(Phi4Checkpoint.ContextLength, config.GetProperty("max_position_embeddings").GetInt32());
        Assert.Equal(Phi4Checkpoint.ShortFactorPositions, config.GetProperty("original_max_position_embeddings").GetInt32());
    }

    [Fact]
    public void ChatDefinitionReadsTheEmbeddedTable()
    {
        ChatRouteContract contract = Phi4Checkpoint.ChatDefinition.Contract;
        Assert.Equal(Phi4Checkpoint.Repository, contract.Repository);
        Assert.Equal(Phi4Checkpoint.Revision, contract.Revision);
        Assert.Equal(Phi4Checkpoint.ModelType, contract.ModelType);
        Assert.Equal(Phi4Checkpoint.VocabularySize, contract.VocabularySize);
        Assert.Equal(Phi4Checkpoint.PadTokenIdOverlay, contract.PadTokenIdOverlay);
        Assert.Equal(Phi4Checkpoint.ContextLength, contract.ContextLength);
        Assert.Equal(Phi4ChatTemplate.TemplateSha256, contract.TemplateSha256);
        Assert.Same(Phi4ChatTemplate.Instance, Phi4Checkpoint.ChatDefinition.Template);
        Assert.Equal("unsupported", contract.Tools);
        Assert.Equal("unsupported", contract.Thinking);
        Assert.Equal("unsupported", contract.Multimodal);
        Assert.Equal("untested", contract.LongContext);
    }

    [Fact]
    public void BundleLockHasTheSixFiles()
    {
        Assert.Equal(
            (string[])["chat_template.jinja", "genai_config.json", "model.onnx", "model.onnx.data", "tokenizer.json", "tokenizer_config.json"],
            Phi4Checkpoint.BundleFiles.Select(f => f.Path).Order(StringComparer.Ordinal));
        Assert.Equal(Phi4Checkpoint.TokenizerSha256, Phi4Checkpoint.BundleFiles.Single(f => f.Path == "tokenizer.json").Sha256);
    }

    [Fact]
    public void LoaderRefusesAnIncompleteBundle()
    {
        string empty = Directory.CreateTempSubdirectory("phi4-bundle-").FullName;
        try
        {
            Assert.Throws<InvalidDataException>(() => GenAIChatModel.Load(Phi4Checkpoint.ChatDefinition, empty));
        }
        finally
        {
            Directory.Delete(empty);
        }
    }

    [Fact]
    public void NothingIsSupportedAndTheKnownGapsHaveAStatus()
    {
        Assert.DoesNotContain(CapabilityStatus.Supported, Phi4Capabilities.All.Values);
        Assert.Equal(CapabilityStatus.Unsupported, Phi4Capabilities.All["tool_calling"]);
        Assert.Equal(CapabilityStatus.Unsupported, Phi4Capabilities.All["vision"]);
        Assert.Equal(CapabilityStatus.Untested, Phi4Capabilities.All["long_context"]);
    }
}
