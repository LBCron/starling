using System.Text.Json;
using Starlings.Generation;
using Starlings.Tokenizers;

namespace Starlings.Models.Phi4.Tests;

/// <summary>
/// The template port against the checkpoint's <c>chat_template</c> rendered by transformers 5.18.0
/// (<c>Data/template-reference.jsonl</c>, written by <c>model-pipeline/reference/genai-chat/make_template_reference.py</c>
/// over <c>corpora/five-models/chat.jsonl</c> and <c>corpora/phi4/template.jsonl</c>), and the owned tokenizer's prompt
/// ids against the transformers tokenizer's.
/// </summary>
public sealed class Phi4ChatTemplateTests
{
    private static readonly Lazy<Dictionary<string, JsonElement>> _cases = new(ReadCases);
    private static readonly Lazy<Tokenizer?> _tokenizer = new(LoadTokenizer);

    public static TheoryData<string> CaseIds => [.. _cases.Value.Keys];

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void RendersLikeTheReference(string id)
    {
        JsonElement row = _cases.Value[id];
        bool generationPrompt = !row.TryGetProperty("add_generation_prompt", out JsonElement value) || value.GetBoolean();

        Assert.Equal(row.GetProperty("text").GetString(), Phi4ChatTemplate.Render(Turns(row.GetProperty("messages")), generationPrompt));
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void PromptIdsMatchTheReferenceTokenizer(string id)
    {
        Tokenizer? tokenizer = _tokenizer.Value;
        Assert.SkipWhen(tokenizer is null, $"Run tools/fetch-tokenizer-assets.sh for {Phi4Checkpoint.Repository} tokenizer.json.");
        JsonElement row = _cases.Value[id];
        int[] expected = [.. row.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32())];

        Assert.Equal(expected, tokenizer!.Encode(row.GetProperty("text").GetString()!, addSpecialTokens: false).Ids.ToArray());
    }

    [Fact]
    public void CoversTheRoleTagsAndTheEndOfTextLiteral()
    {
        Assert.True(_cases.Value.Count >= 34);
        int[][] prompts = [.. _cases.Value.Values.Select(r => r.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray())];
        foreach (int token in (int[])[Phi4Checkpoint.SystemTokenId, Phi4Checkpoint.UserTokenId, Phi4Checkpoint.AssistantTokenId, Phi4Checkpoint.EndTokenId, Phi4Checkpoint.EndOfTextTokenId])
        {
            Assert.Contains(prompts, ids => ids.Contains(token));
        }
    }

    [Fact]
    public void RejectsWhatThePortDoesNotCarry()
    {
        Assert.Throws<ArgumentException>(() => Phi4ChatTemplate.Render([]));
        Assert.Throws<NotSupportedException>(() => Phi4ChatTemplate.Render([ChatTurn.User("a"), ChatTurn.Tool("b")]));
        var call = new ToolCall("get_weather", JsonDocument.Parse("""{"city": "Paris"}""").RootElement.Clone());
        Assert.Throws<NotSupportedException>(() => Phi4ChatTemplate.Render([ChatTurn.User("a"), ChatTurn.Assistant("", [call])]));
        Assert.Throws<NotSupportedException>(() => Phi4ChatTemplate.Instance.Render([ChatTurn.User("a")], [ToolDefinition.Function("f")]));
        Assert.Throws<ArgumentException>(() => Phi4ChatTemplate.Instance.Render([ChatTurn.User("a")], enableThinking: true));
    }

    [Fact]
    public void InterfaceHasNoThinkingSwitch()
    {
        Assert.Equal(Phi4ChatTemplate.TemplateSha256, Phi4ChatTemplate.Instance.TemplateSha256);
        Assert.False(Phi4ChatTemplate.Instance.SupportsThinking);
        Assert.Equal("<|user|>a<|end|><|assistant|>", Phi4ChatTemplate.Instance.Render([ChatTurn.User("a")]));
    }

    private static IReadOnlyList<ChatTurn> Turns(JsonElement messages) =>
    [
        .. messages.EnumerateArray().Select(m =>
        {
            string content = m.GetProperty("content").GetString()!;
            string? reasoning = m.TryGetProperty("reasoning_content", out var r) ? r.GetString() : null;
            return m.GetProperty("role").GetString() switch
            {
                "system" => ChatTurn.System(content),
                "user" => ChatTurn.User(content),
                "assistant" => ChatTurn.Assistant(content, reasoning),
                var role => throw new InvalidDataException($"Unexpected role {role}."),
            };
        }),
    ];

    private static Dictionary<string, JsonElement> ReadCases()
    {
        var cases = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(Path.Combine(AppContext.BaseDirectory, "Data", "template-reference.jsonl")))
        {
            JsonElement row = JsonDocument.Parse(line).RootElement.Clone();
            cases.Add(row.GetProperty("id").GetString()!, row);
        }

        return cases;
    }

    // $STARLING_MODELS/<repo>/<revision>/tokenizer.json, else <repo root>/.models (tools/fetch-tokenizer-assets.sh).
    private static Tokenizer? LoadTokenizer()
    {
        string? root = Environment.GetEnvironmentVariable("STARLING_MODELS");
        if (string.IsNullOrEmpty(root))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "CLAUDE.md")))
            {
                dir = dir.Parent;
            }

            if (dir is null)
            {
                return null;
            }

            root = Path.Combine(dir.FullName, ".models");
        }

        string path = Path.Combine(root, Phi4Checkpoint.Repository, Phi4Checkpoint.Revision, "tokenizer.json");
        return File.Exists(path) ? Phi4Checkpoint.OpenTokenizer(path) : null;
    }
}
