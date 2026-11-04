namespace Starlings.Models;

/// <summary>
/// The machine-readable capability record of one model profile, generated at build time from the profile's registry
/// manifest (<c>model-pipeline/registry/&lt;profile&gt;/manifest.yaml</c>) by <c>model-pipeline catalog</c>, so the
/// package and the published catalog (<c>docs/catalog/catalog.json</c>) cannot disagree.
/// </summary>
/// <remarks>
/// <para>
/// <b>Contract fields.</b> <see cref="ProfileId"/>, <see cref="Maturity"/>, <see cref="MaxInputTokens"/>,
/// <see cref="Truncation"/>, <see cref="Dimension"/> and <see cref="QualifiedProviders"/> are part of a Stable
/// profile's contract: they change only with a new profile or a release note under the support policy.
/// </para>
/// <para>
/// <b>Informational fields.</b> <see cref="Tasks"/>, <see cref="Platforms"/>, <see cref="StopTokenIds"/> and
/// <see cref="Capabilities"/> describe the profile as its manifest records it today; read the package README for what
/// a status covers. A capability the manifest does not record is absent from <see cref="Capabilities"/>, and absent
/// never means supported.
/// </para>
/// <para>
/// <b>Generation fields.</b> <see cref="Generation"/> is informational and Experimental API: it can change in any release
/// until a release note promotes a field to the contract (<c>docs/kickoff.md</c>, "Capability record fields").
/// </para>
/// <para>
/// Collections are read-only and compared by reference in record equality.
/// </para>
/// </remarks>
public sealed record ModelProfileCapabilities
{
    /// <summary>The profile id, <c>&lt;family&gt;/&lt;profile&gt;</c> (manifest <c>id</c>). Contract.</summary>
    public required string ProfileId { get; init; }

    /// <summary>The profile's maturity (manifest <c>maturity</c>). Contract.</summary>
    public required ProfileMaturity Maturity { get; init; }

    /// <summary>Tokens per input at most, special tokens included (manifest <c>behavior.max_input_tokens</c>). Contract.</summary>
    public required int MaxInputTokens { get; init; }

    /// <summary>What happens to longer input (manifest <c>behavior.truncation</c>). Contract.</summary>
    public required TruncationPolicy Truncation { get; init; }

    /// <summary>
    /// Floats per output embedding by default (manifest <c>behavior.defaults.dimension</c>), or null for a profile that
    /// produces no embedding. Contract.
    /// </summary>
    public int? Dimension { get; init; }

    /// <summary>
    /// Execution providers (<c>cpu</c>, <c>cuda</c>, ...) with at least one platform row whose status is
    /// <see cref="CapabilityStatus.Supported"/> (manifest <c>compatibility</c>), sorted. Empty for a profile with no
    /// qualified platform. Contract.
    /// </summary>
    public required IReadOnlyList<string> QualifiedProviders { get; init; }

    /// <summary>The profile's tasks (manifest <c>behavior.tasks</c>, for example <c>embedding</c>). Informational.</summary>
    public IReadOnlyList<string> Tasks { get; init; } = [];

    /// <summary>Every platform row of the manifest's <c>compatibility</c>, in manifest order. Informational.</summary>
    public IReadOnlyList<ProfilePlatform> Platforms { get; init; } = [];

    /// <summary>
    /// Token ids that end generation (manifest <c>behavior.defaults.stop_token_ids</c>); empty for profiles that do not
    /// generate. Informational.
    /// </summary>
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];

    /// <summary>The manifest's <c>capabilities</c> map, key for key. Informational.</summary>
    public IReadOnlyDictionary<string, CapabilityStatus> Capabilities { get; init; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal);

    /// <summary>
    /// How the profile serves generation features (manifest <c>behavior.generation</c>), or null for a profile whose
    /// tasks do not include <c>generation</c>. Informational, Experimental API.
    /// </summary>
    public ProfileGeneration? Generation { get; init; }
}

/// <summary>
/// How a generation profile serves each generation feature (manifest <c>behavior.generation</c>). Informational,
/// Experimental API. Each value says how a feature works, not how far it is qualified: the status stays in
/// <see cref="ModelProfileCapabilities.Capabilities"/>, and the validator and the contract tests hold each value to its
/// row there. A value that is not <c>None</c> or <c>Untested</c> does not mean supported.
/// </summary>
/// <param name="ToolCallMode">Follows the capability <c>tool_calling</c>.</param>
/// <param name="StructuredOutput">Follows the capability <c>structured_output</c> or <c>guided_decoding</c>.</param>
/// <param name="StateReuse">Follows the capability <c>conversation_state_reuse</c> or <c>prefix_reuse</c>.</param>
/// <param name="CachingStyle">Follows the same capability as <paramref name="StateReuse"/>.</param>
/// <param name="MaxOutputTokens">
/// Upper bound on the completion tokens of one request, or null when the capability <c>max_new_tokens</c> is untested,
/// unsupported or not recorded. The bound is shared with the prompt: it equals
/// <see cref="ModelProfileCapabilities.MaxInputTokens"/>, and a request whose prompt plus completion exceed that is
/// refused, so passing it unchanged as a completion limit fails for any non-empty prompt. It is not a qualified
/// completion length.
/// </param>
public sealed record ProfileGeneration(
    ProfileToolCallMode ToolCallMode,
    ProfileStructuredOutput StructuredOutput,
    ProfileStateReuse StateReuse,
    ProfileCachingStyle CachingStyle,
    int? MaxOutputTokens);

/// <summary>How a profile offers tool calls (manifest <c>behavior.generation.tool_call_mode</c>).</summary>
public enum ProfileToolCallMode
{
    /// <summary>The profile has no tool calls (<c>none</c>; the capability is unsupported).</summary>
    None,

    /// <summary>Tools are rendered into the chat template and calls are parsed from the completion (<c>template</c>).</summary>
    TemplateRendered,

    /// <summary>No sufficient evidence: the capability is untested or not recorded (<c>untested</c>).</summary>
    Untested,
}

/// <summary>How a profile constrains its output (manifest <c>behavior.generation.structured_output</c>).</summary>
public enum ProfileStructuredOutput
{
    /// <summary>Nothing constrains decoding (<c>none</c>; the capability is unsupported).</summary>
    None,

    /// <summary>A grammar constrains decoding (<c>grammar</c>).</summary>
    Grammar,

    /// <summary>No sufficient evidence: the capability is untested or not recorded (<c>untested</c>).</summary>
    Untested,
}

/// <summary>Reuse of prefix or conversation state across requests (manifest <c>behavior.generation.state_reuse</c>).</summary>
public enum ProfileStateReuse
{
    /// <summary>Each request starts from fresh state (<c>none</c>; the capability is unsupported).</summary>
    None,

    /// <summary>No sufficient evidence: the capability is untested or not recorded (<c>untested</c>).</summary>
    Untested,
}

/// <summary>Prompt caching across requests (manifest <c>behavior.generation.caching_style</c>).</summary>
public enum ProfileCachingStyle
{
    /// <summary>No prompt is cached across requests (<c>none</c>; the capability is unsupported).</summary>
    None,

    /// <summary>No sufficient evidence: the capability is untested or not recorded (<c>untested</c>).</summary>
    Untested,
}

/// <summary>Maturity of a model profile (manifest <c>maturity</c>; the levels of docs/support-policy.md).</summary>
public enum ProfileMaturity
{
    /// <summary>Nothing beyond the recorded evidence; the profile may change or be removed in any release.</summary>
    Experimental,

    /// <summary>The declared scope passes G0, G1, G2 and G6 in full and G3 to G5 and G7 as checklists.</summary>
    Preview,

    /// <summary>Every gate G0 to G7 passed, regression support, and the contract in the package README.</summary>
    Stable,

    /// <summary>Existing pinned combinations get fixes; new upstream features are not pursued.</summary>
    Maintenance,

    /// <summary>A replacement and a support end date are published; security fixes only until then.</summary>
    Deprecated,
}

/// <summary>What a profile does with input longer than <see cref="ModelProfileCapabilities.MaxInputTokens"/> (manifest <c>behavior.truncation</c>).</summary>
public enum TruncationPolicy
{
    /// <summary>The input is rejected (<see cref="InputTooLongException"/> or the package's subclass of it); nothing is truncated (<c>error</c>).</summary>
    Rejects,

    /// <summary>Tokens past the limit are dropped from the end (<c>truncate-right</c>).</summary>
    TruncatesRight,

    /// <summary>Tokens past the limit are dropped from the start (<c>truncate-left</c>).</summary>
    TruncatesLeft,

    /// <summary>The input is split into chunks of at most the limit (<c>chunk</c>).</summary>
    Chunks,
}

/// <summary>One platform row of a profile's compatibility (manifest <c>compatibility</c>).</summary>
/// <param name="Os">Operating system: <c>linux</c>, <c>windows</c> or <c>macos</c>.</param>
/// <param name="Arch">Architecture: <c>x64</c> or <c>arm64</c>.</param>
/// <param name="Provider">Execution provider, for example <c>cpu</c>, <c>cuda</c>, <c>wasm</c> or <c>webgpu</c>.</param>
/// <param name="Status">The row's status.</param>
/// <param name="Browser">The browser of a browser row (providers <c>wasm</c> and <c>webgpu</c>), else null.</param>
/// <param name="Verified">The date the status was last verified, when recorded.</param>
public sealed record ProfilePlatform(string Os, string Arch, string Provider, CapabilityStatus Status, string? Browser = null, DateOnly? Verified = null)
{
    /// <summary>The catalog's label for the row: <c>&lt;os&gt;-&lt;arch&gt; &lt;provider&gt;</c>, plus the browser on browser rows.</summary>
    public string Label => $"{Os}-{Arch} {Provider}" + (Browser is null ? string.Empty : " " + Browser);
}
