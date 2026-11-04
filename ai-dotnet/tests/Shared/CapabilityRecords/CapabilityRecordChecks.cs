using System.Globalization;
using Starlings.Models;

namespace Starlings.TestSupport;

/// <summary>
/// Checks a package's generated capability records (<see cref="ModelProfileCapabilities"/>, from
/// <c>model-pipeline catalog</c>) against the registry manifests they were generated from, field for field and
/// capability key for key, read by an independent parser (<see cref="RegistryManifest"/>).
/// </summary>
internal static class CapabilityRecordChecks
{
    /// <summary>
    /// <paramref name="records"/> holds exactly the profiles of <paramref name="manifests"/>, and every record equals
    /// its manifest: maturity, max input tokens, truncation, dimension, tasks, qualified providers, platform rows, stop
    /// token ids, the whole capabilities map and the generation fields.
    /// </summary>
    public static void AssertRecordsEqualManifests(IReadOnlyDictionary<string, ModelProfileCapabilities> records, IEnumerable<RegistryManifest> manifests)
    {
        var expected = manifests.ToList();
        Assert.Equal(expected.Select(m => m.Id).Order(StringComparer.Ordinal), records.Keys.Order(StringComparer.Ordinal));
        foreach (var manifest in expected)
        {
            var record = records[manifest.Id];
            string id = manifest.Id;
            Assert.Equal(id, record.ProfileId);
            Assert.True(manifest.Maturity == Name(record.Maturity), $"{id}: maturity {manifest.Maturity}, record {record.Maturity}");
            Assert.Equal(int.Parse(manifest.Behavior["max_input_tokens"], CultureInfo.InvariantCulture), record.MaxInputTokens);
            Assert.True(manifest.Behavior["truncation"] == Truncation(record.Truncation), $"{id}: truncation {manifest.Behavior["truncation"]}, record {record.Truncation}");
            int? dimension = manifest.Behavior.TryGetValue("dimension", out var d) ? int.Parse(d, CultureInfo.InvariantCulture) : null;
            Assert.Equal(dimension, record.Dimension);
            int[] stops = manifest.Behavior.TryGetValue("stop_token_ids", out var s) && s.Length > 0
                ? [.. s.Split(',').Select(x => int.Parse(x.Trim(), CultureInfo.InvariantCulture))]
                : [];
            Assert.Equal(stops, record.StopTokenIds);
            Assert.Equal(manifest.Behavior["tasks"].Split(',').Select(t => t.Trim()), record.Tasks);

            Assert.Equal(manifest.CompatibilityRows.Count, record.Platforms.Count);
            for (int i = 0; i < record.Platforms.Count; i++)
            {
                var row = manifest.CompatibilityRows[i];
                var platform = record.Platforms[i];
                Assert.Equal(row["os"], platform.Os);
                Assert.Equal(row["arch"], platform.Arch);
                Assert.Equal(row["provider"], platform.Provider);
                Assert.Equal(row["status"], Name(platform.Status));
                Assert.Equal(row.GetValueOrDefault("browser"), platform.Browser);
                Assert.Equal(row.TryGetValue("verified", out var v) ? DateOnly.Parse(v, CultureInfo.InvariantCulture) : null, platform.Verified);
            }

            Assert.Equal(
                manifest.CompatibilityRows.Where(r => r["status"] == "supported").Select(r => r["provider"]).Distinct().Order(StringComparer.Ordinal),
                record.QualifiedProviders);

            Assert.Equal(manifest.Capabilities.Keys.Order(StringComparer.Ordinal), record.Capabilities.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, status) in manifest.Capabilities)
            {
                Assert.True(status == Name(record.Capabilities[key]), $"{id} {key}: manifest {status}, record {Name(record.Capabilities[key])}");
            }

            if (record.Maturity == ProfileMaturity.Stable)
            {
                // The validator refuses Stable with an untested capability; the record follows the same rule.
                Assert.DoesNotContain(CapabilityStatus.Untested, record.Capabilities.Values);
            }

            AssertGenerationEqualsManifest(manifest, record);
        }
    }

    /// <summary>
    /// The record's generation fields equal the manifest's, field by field, and each follows its capability row. A profile
    /// has them exactly when its tasks include generation.
    /// </summary>
    private static void AssertGenerationEqualsManifest(RegistryManifest manifest, ModelProfileCapabilities record)
    {
        string id = manifest.Id;
        bool generates = record.Tasks.Contains("generation");
        Assert.True(generates == manifest.Behavior.ContainsKey("generation.tool_call_mode"), $"{id}: tasks and behavior.generation disagree");
        if (record.Generation is not { } generation)
        {
            Assert.False(generates, $"{id}: a generation profile without Generation");
            return;
        }

        Assert.True(generates, $"{id}: Generation on a profile that does not generate");
        Assert.Equal(manifest.Behavior["generation.tool_call_mode"], Name(generation.ToolCallMode));
        Assert.Equal(manifest.Behavior["generation.structured_output"], Name(generation.StructuredOutput));
        Assert.Equal(manifest.Behavior["generation.state_reuse"], Name(generation.StateReuse));
        Assert.Equal(manifest.Behavior["generation.caching_style"], Name(generation.CachingStyle));
        string maxOutput = manifest.Behavior["generation.max_output_tokens"];
        Assert.Equal(maxOutput == "untested" ? null : int.Parse(maxOutput, CultureInfo.InvariantCulture), generation.MaxOutputTokens);

        // Every value traces to a capability row: a second, independent reading of the validator's rule.
        Assert.Equal(Expected(record, "template", "tool_calling"), Name(generation.ToolCallMode));
        Assert.Equal(Expected(record, "grammar", "structured_output", "guided_decoding"), Name(generation.StructuredOutput));
        Assert.Equal(Expected(record, null, "conversation_state_reuse", "prefix_reuse"), Name(generation.StateReuse));
        Assert.Equal(Expected(record, null, "conversation_state_reuse", "prefix_reuse"), Name(generation.CachingStyle));
        var maxNewTokens = Row(record, "max_new_tokens");
        Assert.Equal(maxNewTokens is CapabilityStatus.Supported or CapabilityStatus.Experimental ? record.MaxInputTokens : null, generation.MaxOutputTokens);

        if (record.Maturity == ProfileMaturity.Stable)
        {
            Assert.NotEqual(ProfileToolCallMode.Untested, generation.ToolCallMode);
            Assert.NotEqual(ProfileStructuredOutput.Untested, generation.StructuredOutput);
            Assert.NotEqual(ProfileStateReuse.Untested, generation.StateReuse);
            Assert.NotEqual(ProfileCachingStyle.Untested, generation.CachingStyle);
            Assert.NotNull(generation.MaxOutputTokens);
        }
    }

    /// <summary>The first recorded of <paramref name="keys"/>, or null when the profile records none of them.</summary>
    private static CapabilityStatus? Row(ModelProfileCapabilities record, params string[] keys)
    {
        foreach (string key in keys)
        {
            if (record.Capabilities.TryGetValue(key, out var status))
            {
                return status;
            }
        }

        return null;
    }

    /// <summary>The generation value a capability row allows; <paramref name="available"/> is null where no value exists yet.</summary>
    private static string? Expected(ModelProfileCapabilities record, string? available, params string[] keys) => Row(record, keys) switch
    {
        null or CapabilityStatus.Untested => "untested",
        CapabilityStatus.Unsupported => "none",
        _ => available,
    };

    /// <summary>A generation enum value as the manifests spell it.</summary>
    private static string Name(Enum value) => value is ProfileToolCallMode.TemplateRendered ? "template" : value.ToString().ToLowerInvariant();

    /// <summary>A status as the manifests spell it.</summary>
    public static string Name(CapabilityStatus status) => status.ToString().ToLowerInvariant();

    private static string Name(ProfileMaturity maturity) => maturity.ToString().ToLowerInvariant();

    private static string Truncation(TruncationPolicy policy) => policy switch
    {
        TruncationPolicy.Rejects => "error",
        TruncationPolicy.TruncatesRight => "truncate-right",
        TruncationPolicy.TruncatesLeft => "truncate-left",
        TruncationPolicy.Chunks => "chunk",
        _ => throw new ArgumentOutOfRangeException(nameof(policy), policy, null),
    };
}
