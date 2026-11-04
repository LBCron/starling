namespace Starlings.Models.Phi4;

/// <summary>One file of a profile's bundle lock.</summary>
/// <param name="Path">File name in the bundle directory.</param>
/// <param name="Sha256">Lower-case hex SHA-256 of the file.</param>
/// <param name="Size">Size in bytes.</param>
public sealed record Phi4BundleFile(string Path, string Sha256, long Size);

/// <summary>
/// The pinned original (<see cref="Phi4Checkpoint"/>) with its ONNX Runtime GenAI bundle lock: the six files the pinned
/// builder produces from it (<c>model-pipeline/reference/genai-chat/build_bundle.py</c>). No hosted copy exists: a user
/// builds the bundle, and <see cref="Phi4Model.Load"/> verifies it against this lock.
/// </summary>
public sealed class Phi4Profile
{
    private Phi4Profile(string id, Phi4BundleFile[] bundleFiles)
    {
        Id = id;
        BundleFiles = Array.AsReadOnly(bundleFiles);
    }

    /// <summary>CPU fp32 text chat.</summary>
    public static Phi4Profile Mini { get; } = new("phi4-mini-instruct/ortgenai-fp32-cpu",
        [
            new("genai_config.json", "9187e9d71430a2ae23bdeb04a4042920f4e8fcfbe49c3fe9ae046eff6a1b329b", 1821),
            new("model.onnx", "b6b05c0d65000207940950184db3d29d8e84052470dac52f0f1d89df5555efb3", 52088021),
            new("model.onnx.data", "19aae69a1272c1bbf3a13935b371476700d5b12eb8290ee282631a39d8f67430", 15344140288),
            new("tokenizer.json", "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea", 15524095),
            new("tokenizer_config.json", "9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db", 2932),
            new("chat_template.jinja", "febf589225c9728ab791f52e8897d7607a823d45368f0a4c92fa68997b40cce9", 423),
        ]);

    /// <summary>Portable candidate profile id.</summary>
    public string Id { get; }

    /// <summary>The bundle lock: every file <see cref="Phi4Model.Load"/> requires, with its pinned digest and size.</summary>
    public IReadOnlyList<Phi4BundleFile> BundleFiles { get; }

    /// <summary>Resolve a candidate id, rejecting other profiles.</summary>
    public static Phi4Profile FromId(string id) => id switch
    {
        "phi4-mini-instruct/ortgenai-fp32-cpu" => Mini,
        _ => throw new NotSupportedException($"Unsupported Phi profile: {id}"),
    };
}
