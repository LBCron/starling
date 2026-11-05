namespace Starlings.Models.Phi4;

/// <summary>Support status of one capability. A missing entry never means supported.</summary>
public enum CapabilityStatus
{
    /// <summary>Qualified for the advertised maturity.</summary>
    Supported,

    /// <summary>Not available; requests for it are rejected.</summary>
    Unsupported,

    /// <summary>Implemented and tested against a reference, not yet qualified.</summary>
    Experimental,

    /// <summary>May work, but no evidence exists.</summary>
    Untested,
}

/// <summary>Capability descriptor for this package version.</summary>
/// <remarks>
/// <see cref="All"/> is package-wide. Nothing is Supported: no gate has run and no reference is protected
/// (<c>model-pipeline/admissions/phi4-mini.md</c>). Everything not listed here is Unsupported.
/// </remarks>
public static class Phi4Capabilities
{
    /// <summary>Status per capability.</summary>
    public static IReadOnlyDictionary<string, CapabilityStatus> All { get; } = new Dictionary<string, CapabilityStatus>(StringComparer.Ordinal)
    {
        ["chat_template.text"] = CapabilityStatus.Experimental,
        ["checkpoint_lock"] = CapabilityStatus.Experimental,
        ["generation.greedy.cpu_fp32"] = CapabilityStatus.Experimental,
        ["streaming"] = CapabilityStatus.Experimental,
        ["cancellation"] = CapabilityStatus.Experimental,
        ["stop_tokens_and_sequences"] = CapabilityStatus.Experimental,
        ["long_context"] = CapabilityStatus.Untested,        // prompt plus completion at or past 4,096 tokens: long rope factors
        ["sampling"] = CapabilityStatus.Untested,            // the decode loop has it; no Phi comparison ran
        ["logprobs"] = CapabilityStatus.Untested,
        ["guided_decoding"] = CapabilityStatus.Untested,
        ["cuda"] = CapabilityStatus.Unsupported,             // the loader is CPU only
        ["fp16"] = CapabilityStatus.Unsupported,
        ["tool_calling"] = CapabilityStatus.Unsupported,     // the template's tool section is rejected
        ["thinking"] = CapabilityStatus.Unsupported,         // the template has no switch
        ["vision"] = CapabilityStatus.Unsupported,           // the checkpoint is text only
        ["extensions_ai"] = CapabilityStatus.Unsupported,
        ["aot"] = CapabilityStatus.Untested,
    };
}
