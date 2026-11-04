using System.Globalization;
using CommunityToolkit.VectorData.InMemory;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Starlings.Acquisition;
using Starlings.Models;
using Starlings.Models.Bert;
using Starlings.Models.E5;
using Starlings.Models.E5.Extensions.AI;

const int Candidates = 5; // the embedder's top-k, read by the reranker
const int Best = 3;       // what the sample prints after reranking

try
{
    return await RunAsync();
}
catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException))
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

async Task<int> RunAsync()
{
    if (args.Length == 0 || args[0] is not ("download" or "search"))
        return Usage();

    string? cache = null;
    string? question = null;
    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] == "--cache" && i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            cache = args[++i];
        else if (!args[i].StartsWith("--", StringComparison.Ordinal) && args[0] == "search" && question is null)
            question = args[i];
        else
            return Usage();
    }

    var acquisition = new ModelAcquisitionOptions { CacheDirectory = cache };
    BertModelProfile rerankerProfile = BertModelProfile.BuiltIn("bge-reranker-v2-m3");
    using var store = new ModelAssetStore(acquisition);
    if (args[0] == "download")
    {
        var e5Files = await E5ModelAssets.DownloadAsync(acquisition);
        Console.WriteLine($"Verified embedder: {e5Files.Snapshot.Directory}");
        ModelSnapshot rerankerFiles = await store.DownloadAsync(rerankerProfile.RuntimeLock);
        Console.WriteLine($"Verified reranker: {rerankerFiles.Directory}");
        return 0;
    }

    // No automatic prefix: the vector property supplies "passage: ", and SearchAsync supplies "query: ".
    using IEmbeddingGenerator<string, Embedding<float>> generator =
        E5EmbeddingGenerator.Create(E5ModelAssets.OpenLocal(acquisition));
    using var reranker = BertReranker.Create(rerankerProfile, store.OpenLocal(rerankerProfile.RuntimeLock).Directory);
    using var collection = new InMemoryCollection<string, Passage>("passages", new()
    {
        EmbeddingGenerator = generator,
    });
    await collection.EnsureCollectionExistsAsync();
    Passage[] passages =
    [
        new() { Id = "rhine", Text = "Der Rhein fließt durch Köln." },
        new() { Id = "cologne-cathedral", Text = "Der Kölner Dom ist das Wahrzeichen von Köln." },
        new() { Id = "danube", Text = "Le Danube traverse Vienne et Budapest avant de rejoindre la mer Noire." },
        new() { Id = "seine", Text = "The Seine flows through Paris on its way to the English Channel." },
        new() { Id = "tokyo", Text = "東京は日本の首都です。" },
        new() { Id = "sumida", Text = "隅田川は東京を流れて東京湾に注ぐ。" },
        new() { Id = "bread", Text = "Para hacer pan se necesita harina, agua, sal y levadura." },
        new() { Id = "everest", Text = "Mount Everest is the highest mountain above sea level." },
    ];
    // The collection embeds the string vector property on upsert; no manual vector generation is needed.
    await collection.UpsertAsync(passages);
    Console.WriteLine($"Indexed {passages.Length} multilingual passages in memory.");
    question ??= "Which river flows through Cologne?";
    Console.WriteLine($"Query: {question}");

    // First stage: the embedder's top-k. Second stage: RerankAsync reads the search to the end and scores all the
    // candidates in one call, with the plain query and passage text (no e5 prefixes). `top` could cut the result here;
    // this sample keeps every candidate so that it can print both orders.
    IReadOnlyList<RerankedItem<VectorSearchResult<Passage>>> ranked = await reranker.RerankAsync(
        question,
        collection.SearchAsync("query: " + question, top: Candidates),
        hit => hit.Record.Text);

    Console.WriteLine($"e5 top {Candidates} (cosine):");
    foreach (var r in ranked.OrderBy(r => r.Index))
    {
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"  #{r.Index + 1} {r.Item.Score:F4}  {r.Item.Record.Id}: {r.Item.Record.Text}"));
    }

    // Raw logits, comparable only within this call; not probabilities.
    Console.WriteLine($"bge-reranker-v2-m3 top {Best} (logit, was e5 #):");
    foreach (var r in ranked.Take(Best))
    {
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"  {r.Score:F4}  (was #{r.Index + 1})  {r.Item.Record.Id}: {r.Item.Record.Text}"));
    }
    return 0;
}

static int Usage()
{
    Console.Error.WriteLine("usage: E5RerankSearch download [--cache <dir>] | search [--cache <dir>] [\"question\"]");
    return 2;
}

internal sealed class Passage
{
    [VectorStoreKey]
    public required string Id { get; set; }

    [VectorStoreData]
    public required string Text { get; set; }

    // A string vector property tells VectorData to use the collection's embedding generator.
    [VectorStoreVector(384, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public string Vector => "passage: " + Text;
}
