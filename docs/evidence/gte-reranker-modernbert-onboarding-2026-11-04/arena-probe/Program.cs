// Peak memory and latency of one profile on .NET, through BertReranker or BertTextEmbedder, the public API. One setting
// per process, because VmHWM never falls.
//
// Usage: ArenaProbe <snapshot> <built-in id> <threads> <runs> <arena|noarena> <tokens> [batch] [maxTokens]
//
// "arena" and "noarena" set the AppContext switch Starlings.Models.Bert.CpuMemoryArena. <tokens> is the pair's length
// after truncation, as the Python probe ort_peak.py builds it; [batch] is the rows per call; [maxTokens] goes to
// BertReranker.Create instead of the profile's limit.
using System.Diagnostics;
using Starlings.Models.Bert;

string snapshot = args[0], id = args[1], arena = args[4];
int threads = int.Parse(args[2]), runs = int.Parse(args[3]), tokens = int.Parse(args[5]);
int batch = args.Length > 6 ? int.Parse(args[6]) : 1;
int? maxTokens = args.Length > 7 ? int.Parse(args[7]) : null;
AppContext.SetSwitch("Starlings.Models.Bert.CpuMemoryArena", arena == "arena");

static (double Hwm, double Rss) Memory()
{
    double hwm = 0, rss = 0;
    foreach (string line in File.ReadLines("/proc/self/status"))
    {
        if (line.StartsWith("VmHWM:", StringComparison.Ordinal))
        {
            hwm = double.Parse(line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)[1]) / 1e6;
        }
        else if (line.StartsWith("VmRSS:", StringComparison.Ordinal))
        {
            rss = double.Parse(line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)[1]) / 1e6;
        }
    }

    return (hwm, rss);
}

const string Query = "how to implement quick sort in python?";
string document = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog ", 1000));
var profile = BertModelProfile.BuiltIn(id);
var clock = Stopwatch.StartNew();
if (profile.Task == BertTask.Reranking)
{
    using var reranker = BertReranker.Create(profile, snapshot, threads, maxBatchSize: batch, maxTokens: maxTokens ?? (tokens < profile.MaxTokens ? tokens : null));
    var memory = Memory();
    Console.WriteLine($"{id} {arena} threads {threads} session arena {reranker.Diagnostics.CpuMemoryArena} maxTokens {reranker.MaxTokens} load {clock.Elapsed.TotalSeconds:F1}s peak {memory.Hwm:F2}G rss {memory.Rss:F2}G");
    var pair = new BertPair(Query, document);
    int length = reranker.Tokenize(pair).Length;
    if (length != tokens)
    {
        throw new InvalidOperationException($"pair is {length} tokens, not {tokens}");
    }

    var pairs = Enumerable.Repeat(pair, batch).ToArray();
    for (int run = 1; run <= runs; run++)
    {
        clock.Restart();
        float[] scores = reranker.ScoreBatch(pairs);
        double seconds = clock.Elapsed.TotalSeconds;
        memory = Memory();
        Console.WriteLine($"tokens {length} rows {batch} run {run} logit {scores[0]:F6} time {seconds:F3}s peak {memory.Hwm:F2}G rss {memory.Rss:F2}G");
    }
}
else
{
    using var embedder = BertTextEmbedder.Create(profile, snapshot, threads, maxBatchSize: batch);
    var memory = Memory();
    Console.WriteLine($"{id} {arena} threads {threads} session arena {embedder.Diagnostics.CpuMemoryArena} load {clock.Elapsed.TotalSeconds:F1}s peak {memory.Hwm:F2}G rss {memory.Rss:F2}G");
    var texts = Enumerable.Repeat(string.Join(' ', document.Split(' ').Take(tokens)), batch).ToArray();
    for (int run = 1; run <= runs; run++)
    {
        clock.Restart();
        float[][] vectors = embedder.EmbedBatch(texts);
        double seconds = clock.Elapsed.TotalSeconds;
        memory = Memory();
        Console.WriteLine($"rows {batch} run {run} first {vectors[0][0]:F6} time {seconds:F3}s peak {memory.Hwm:F2}G rss {memory.Rss:F2}G");
    }
}
