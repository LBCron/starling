// Ported from the chat_template string of tokenizer_config.json of microsoft/Phi-4-mini-instruct at
// cfbefacb99257ffa30c83adab238a50856ac3083, Copyright (c) Microsoft Corporation, MIT. Modified: rewritten in C#,
// restricted to text system, user and assistant turns.

using System.Text;
using Starlings.Generation;

namespace Starlings.Models.Phi4;

/// <summary>
/// The Phi-4-mini chat template (<c>chat_template</c> of <c>tokenizer_config.json</c> of
/// <c>microsoft/Phi-4-mini-instruct</c> at <c>cfbefacb</c>) for text turns: <c>&lt;|role|&gt;content&lt;|end|&gt;</c>
/// per turn, then <c>&lt;|assistant|&gt;</c> (or <c>&lt;|endoftext|&gt;</c> without a generation prompt).
/// </summary>
/// <remarks>
/// <para>Text is inserted as given, as the template does: a special token's text in a turn (for example
/// <c>&lt;|end|&gt;</c>) becomes that special token when tokenized, and the tokenizer's <c>rstrip</c> removes the
/// whitespace that follows a role tag, so a turn's leading whitespace does not reach the model. Turn order is not
/// checked; the template has no rule for it. An assistant turn's <see cref="ChatTurn.ReasoningContent"/> is ignored,
/// as the template ignores it.</para>
/// <para>Not supported (rejected with <see cref="NotSupportedException"/>): the template's tool section
/// (<c>tools</c> on a system message), tool calls and tool turns. Phi-4-mini has no thinking switch.</para>
/// </remarks>
public static class Phi4ChatTemplate
{
    /// <summary>SHA-256 of the upstream <c>chat_template</c> string, as UTF-8, this renderer follows (423 bytes).</summary>
    public const string TemplateSha256 = "febf589225c9728ab791f52e8897d7607a823d45368f0a4c92fa68997b40cce9";

    /// <summary>The renderer as the runtime-independent <see cref="IChatTemplate"/>.</summary>
    public static IChatTemplate Instance { get; } = new Renderer();

    /// <summary>Renders <paramref name="turns"/> as the template does with <c>add_generation_prompt=True</c> (or false).</summary>
    /// <param name="turns">The conversation (at least one turn).</param>
    /// <param name="addGenerationPrompt">Append the assistant generation prompt (default true).</param>
    /// <exception cref="ArgumentException">The conversation is empty.</exception>
    /// <exception cref="NotSupportedException">A tool turn or tool calls.</exception>
    public static string Render(IReadOnlyList<ChatTurn> turns, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(turns);
        if (turns.Count == 0)
        {
            throw new ArgumentException("The conversation needs at least one turn.", nameof(turns));
        }

        var text = new StringBuilder();
        foreach (ChatTurn turn in turns)
        {
            ArgumentNullException.ThrowIfNull(turn);
            if (turn.ToolCalls is { Count: > 0 })
            {
                throw new NotSupportedException("Tool calls are unsupported for Phi-4-mini.");
            }

            string tag = turn.Role switch
            {
                ChatTurnRole.System => "<|system|>",
                ChatTurnRole.User => "<|user|>",
                ChatTurnRole.Assistant => "<|assistant|>",
                _ => throw new NotSupportedException("Tool turns are unsupported for Phi-4-mini."),
            };
            text.Append(tag).Append(turn.Content).Append("<|end|>");
        }

        return text.Append(addGenerationPrompt ? "<|assistant|>" : "<|endoftext|>").ToString();
    }

    private sealed class Renderer : IChatTemplate
    {
        public string TemplateSha256 => Phi4ChatTemplate.TemplateSha256;

        public bool SupportsThinking => false;

        public string Render(IReadOnlyList<ChatTurn> turns, IReadOnlyList<ToolDefinition>? tools = null, bool enableThinking = false)
        {
            if (tools is { Count: > 0 })
            {
                throw new NotSupportedException("Tools are unsupported for Phi-4-mini.");
            }

            if (enableThinking)
            {
                throw new ArgumentException("Phi-4-mini has no thinking switch.", nameof(enableThinking));
            }

            return Phi4ChatTemplate.Render(turns);
        }
    }
}
