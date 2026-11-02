using System.Globalization;
using System.Text.Json;
using Starlings.Models.Qwen3Embedding;

return ToolMain.Run(ToolBody);

// The dotnet stage of the qwen3-embedding-onnx-cpu-v1 onboarding route. Checks the candidate contract against the
// library's built-in profile (upstream revision, hosted graph digests, prompts, special ids, template), opens the
// released directory through the profile's lock, then for every reference case records the library's token ids and its
// output from the text (one right-padded batch per MaxBatchSize rows) and from the reference ids. Writes
// <out>/dotnet.json. Exit 2 on a usage error, 1 on a contract mismatch.
int ToolBody()
{
    var options = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int i = 0; i < args.Length; i += 2)
    {
        if (i + 1 >= args.Length || args[i] is not ("--profile" or "--release" or "--reference" or "--out" or "--threads")
            || !options.TryAdd(args[i], args[i + 1]))
        {
            Console.Error.WriteLine("usage: --profile <candidate.json> --release <dir> --reference <reference.json> --out <dir> [--threads 2]");
            return 2;
        }
    }

    if (Required.Any(key => !options.ContainsKey(key)))
    {
        Console.Error.WriteLine("missing --profile, --release, --reference, or --out");
        return 2;
    }

    int threads = options.TryGetValue("--threads", out var rawThreads) ? int.Parse(rawThreads, CultureInfo.InvariantCulture) : 2;
    using var candidate = JsonDocument.Parse(File.ReadAllText(options["--profile"]));
    var problems = ContractProblems(candidate.RootElement, out var profile);
    if (problems.Count > 0)
    {
        foreach (string problem in problems)
        {
            Console.Error.WriteLine("contract: " + problem);
        }

        return 1;
    }

    using var reference = JsonDocument.Parse(File.ReadAllText(options["--reference"]));
    var root = reference.RootElement;
    var cases = root.GetProperty("cases").EnumerateArray().ToArray();
    var settings = new Qwen3EmbeddingOptions { Threads = threads, MaxBatchSize = 8 };
    string release = Path.GetFullPath(options["--release"]);
    var result = new Dictionary<string, object>
    {
        ["profile"] = profile!.Id,
        ["threads"] = threads,
        ["max_batch_size"] = settings.MaxBatchSize,
        ["max_batch_tokens"] = settings.MaxBatchTokens!,
    };
    var rows = new List<Dictionary<string, object>>();
    if (profile.Task == Qwen3EmbeddingTask.Embedding)
    {
        using var embedder = Qwen3TextEmbedder.Create(release, settings, profile);
        foreach (var group in cases.GroupBy(c => c.GetProperty("prompt").GetString()!))
        {
            var items = group.ToArray();
            float[][] fromText = embedder.EmbedBatch(items.Select(c => c.GetProperty("text").GetString()!).ToArray(), group.Key);
            for (int i = 0; i < items.Length; i++)
            {
                var c = items[i];
                var input = embedder.Tokenize(c.GetProperty("text").GetString()!, group.Key);
                long[] ids = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt64()).ToArray();
                float[] fromIds = embedder.EmbedTokens([new Qwen3Input(ids, Enumerable.Repeat(1L, ids.Length).ToArray())])[0];
                rows.Add(new()
                {
                    ["id"] = c.GetProperty("id").GetString()!,
                    ["ids"] = input.InputIds.ToArray(),
                    ["embedding_text"] = fromText[i],
                    ["embedding_ids"] = fromIds,
                });
            }
        }

        string[] queries = ["What is the capital of China?", "Explain gravity"];
        string[] documents =
        [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        ];
        float[][] q = embedder.EmbedBatch(queries, "query");
        float[][] d = embedder.EmbedBatch(documents, "none");
        result["card_cosines"] = q.Select(a => d.Select(b => (double)a.Zip(b, (x, y) => x * y).Sum()).ToArray()).ToArray();
    }
    else
    {
        using var reranker = Qwen3Reranker.Create(release, settings, profile);
        var inputs = cases.Select(c => reranker.Tokenize(
            c.GetProperty("query").GetString()!, c.GetProperty("document").GetString()!, c.GetProperty("instruction").GetString()!)).ToArray();
        float[] fromText = reranker.ScoreTokens(inputs);
        for (int i = 0; i < cases.Length; i++)
        {
            long[] ids = cases[i].GetProperty("ids").EnumerateArray().Select(e => e.GetInt64()).ToArray();
            float fromIds = reranker.ScoreTokens([new Qwen3Input(ids, Enumerable.Repeat(1L, ids.Length).ToArray())])[0];
            rows.Add(new()
            {
                ["id"] = cases[i].GetProperty("id").GetString()!,
                ["ids"] = inputs[i].InputIds.ToArray(),
                ["score_text"] = fromText[i],
                ["score_ids"] = fromIds,
            });
        }

        // The public Score API (default instruction) on the card pairs, one call per query.
        var publicScores = new List<Dictionary<string, object>>();
        var card = cases.Where(c => c.GetProperty("id").GetString()!.StartsWith("card-", StringComparison.Ordinal)).ToArray();
        foreach (var group in card.GroupBy(c => c.GetProperty("query").GetString()!))
        {
            var items = group.ToArray();
            float[] scores = reranker.Score(group.Key, items.Select(c => c.GetProperty("document").GetString()!).ToArray());
            for (int i = 0; i < items.Length; i++)
            {
                publicScores.Add(new() { ["id"] = items[i].GetProperty("id").GetString()!, ["score"] = scores[i] });
            }
        }

        result["public_scores"] = publicScores;
        var tokenization = Qwen3Tokenization.Open(profile, Path.Combine(release, Qwen3EmbeddingProfile.TokenizerFile));
        result["prefix_ids"] = tokenization.RerankPrefixIds.ToArray();
        result["suffix_ids"] = tokenization.RerankSuffixIds.ToArray();
    }

    result["cases"] = rows;
    string output = Path.GetFullPath(options["--out"]);
    Directory.CreateDirectory(output);
    File.WriteAllText(Path.Combine(output, "dotnet.json"), JsonSerializer.Serialize(result, Indented) + "\n");
    Console.WriteLine($"{profile.Id}: {rows.Count} cases");
    return 0;
}

// The candidate must be exactly the built-in profile's contract.

static List<string> ContractProblems(JsonElement candidate, out Qwen3EmbeddingProfile? profile)
{
    var problems = new List<string>();
    var origin = candidate.GetProperty("origin");
    string repository = origin.GetProperty("repo").GetString()!;
    profile = Qwen3EmbeddingProfile.BuiltIn.FirstOrDefault(p => p.Repository == repository);
    if (profile is null)
    {
        problems.Add($"no built-in profile for {repository}");
        return problems;
    }

    void Same(string field, object? actual, object? expected)
    {
        if (!Equals(actual, expected))
        {
            problems.Add($"{field}: candidate {actual}, library {expected}");
        }
    }

    var behavior = candidate.GetProperty("behavior");
    Same("id", candidate.GetProperty("id").GetString(), profile.Id + "/onnx-fp32-cpu");
    Same("task", candidate.GetProperty("task").GetString(), profile.Task == Qwen3EmbeddingTask.Embedding ? "embeddings" : "reranking");
    Same("origin.revision", origin.GetProperty("revision").GetString(), profile.Revision);
    Same("behavior.tokenizer_profile", behavior.GetProperty("tokenizer_profile").GetString(), profile.TokenizerProfile);
    Same("behavior.max_tokens", behavior.GetProperty("max_tokens").GetInt32(), profile.MaxTokens);
    Same("behavior.hidden_size", behavior.GetProperty("hidden_size").GetInt32(), Qwen3EmbeddingProfile.HiddenSize);
    Same("behavior.vocabulary_size", behavior.GetProperty("vocabulary_size").GetInt32(), Qwen3EmbeddingProfile.VocabularySize);
    Same("behavior.pad_token_id", behavior.GetProperty("pad_token_id").GetInt32(), Qwen3EmbeddingProfile.PadTokenId);
    var hosting = behavior.GetProperty("hosting");
    Same("behavior.hosting.repo", hosting.GetProperty("repo").GetString(), profile.HostingRepository);
    Same("behavior.hosting.revision", hosting.GetProperty("revision").GetString(), profile.HostingRevision);
    foreach (var file in hosting.GetProperty("files").EnumerateArray())
    {
        string path = file.GetProperty("path").GetString()!;
        var locked = profile.RuntimeLock.Files.FirstOrDefault(f => f.Path == path);
        Same($"behavior.hosting.files[{path}].sha256", file.GetProperty("sha256").GetString(), locked?.Sha256);
        Same($"behavior.hosting.files[{path}].size", file.GetProperty("size").GetInt64(), locked?.Size);
    }

    var tokenizer = origin.GetProperty("files").EnumerateArray().First(f => f.GetProperty("path").GetString() == Qwen3EmbeddingProfile.TokenizerFile);
    Same("origin.files[tokenizer.json].sha256", tokenizer.GetProperty("sha256").GetString(), profile.RuntimeLock.GetFile(Qwen3EmbeddingProfile.TokenizerFile).Sha256);
    if (profile.Task == Qwen3EmbeddingTask.Embedding)
    {
        Same("behavior.eos_token_id", behavior.GetProperty("eos_token_id").GetInt32(), Qwen3EmbeddingProfile.EndOfTextTokenId);
        Same("behavior.dimension", behavior.GetProperty("dimension").GetInt32(), Qwen3EmbeddingProfile.HiddenSize);
        var prompts = behavior.GetProperty("prompts").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!);
        Same("behavior.prompts", string.Join("|", prompts.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => p.Key + "=" + p.Value)),
            string.Join("|", profile.Prompts.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => p.Key + "=" + p.Value)));
    }
    else
    {
        Same("behavior.yes_token_id", behavior.GetProperty("yes_token_id").GetInt32(), Qwen3EmbeddingProfile.YesTokenId);
        Same("behavior.no_token_id", behavior.GetProperty("no_token_id").GetInt32(), Qwen3EmbeddingProfile.NoTokenId);
        Same("behavior.default_instruction", behavior.GetProperty("default_instruction").GetString(), Qwen3EmbeddingProfile.DefaultRerankInstruction);
        Same("behavior.prefix", behavior.GetProperty("prefix").GetString(), Qwen3EmbeddingProfile.RerankPrefix);
        Same("behavior.suffix", behavior.GetProperty("suffix").GetString(), Qwen3EmbeddingProfile.RerankSuffix);
    }

    return problems;
}

internal static partial class Program
{
    private static readonly string[] Required = ["--profile", "--release", "--reference", "--out"];

    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };
}
