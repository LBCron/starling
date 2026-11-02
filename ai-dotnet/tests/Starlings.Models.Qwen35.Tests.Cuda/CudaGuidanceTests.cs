using System.Text.Json;
using System.Text.RegularExpressions;
using Starlings.Generation;
using Starlings.Models.Qwen35;

namespace Starlings.Models.Qwen35.Tests.Cuda;

/// <summary>
/// Guided decoding (llguidance) on the CUDA provider, from the variant directory as the lock lays it out (with the guidance
/// files since hosting revision r2). Greedy; outputs are checked against the constraint, not a reference.
/// </summary>
public abstract class CudaGuidanceChecks
{
    private static readonly Qwen35Tool Weather = Qwen35Tool.Function(
        "get_weather",
        "Current weather for a city.",
        JsonDocument.Parse("""
        {"type": "object", "properties": {
          "city": {"type": "string"},
          "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }, "required": ["city"]}
        """).RootElement.Clone());

    private static readonly Qwen35Tool Timer = Qwen35Tool.Function(
        "set_timer",
        "Starts a countdown timer.",
        JsonDocument.Parse("""{"type": "object", "properties": {"minutes": {"type": "integer"}}, "required": ["minutes"]}""").RootElement.Clone());

    protected abstract Qwen35ChatModel Model { get; }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void TheDirectoryHasTheGuidanceFiles() => Assert.Empty(Model.Info.GuidanceMissingFiles);

    [Fact]
    public void ARegexIsFollowed()
    {
        var r = Model.Generate(
            [Qwen35Message.User("Write a product code.")],
            new Qwen35ChatOptions { MaxNewTokens = 16, Guidance = GenerationGuidance.Regex("FORCED-[0-9]{3}") },
            Ct);
        Report("regex", r);
        Assert.Matches("^FORCED-[0-9]{3}$", r.Text);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
    }

    [Fact]
    public void AJsonSchemaIsFollowed()
    {
        const string Schema = """{"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "additionalProperties": false}""";
        var r = Model.Generate(
            [Qwen35Message.User("Name the capital of France and its country as JSON.")],
            new Qwen35ChatOptions { MaxNewTokens = 48, Guidance = GenerationGuidance.JsonSchema(Schema) },
            Ct);
        Report("json_schema", r);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
        using var json = JsonDocument.Parse(r.Text);
        Assert.Equal(JsonValueKind.String, json.RootElement.GetProperty("city").ValueKind);
        Assert.Equal(JsonValueKind.String, json.RootElement.GetProperty("country").ValueKind);
        Assert.Equal(2, json.RootElement.EnumerateObject().Count());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheRequiredToolGrammarYieldsOneParsedCall(bool fastForward)
    {
        Qwen35Tool[] tools = [Weather, Timer];
        var options = new Qwen35ChatOptions
        {
            MaxNewTokens = 64,
            Tools = tools,
            Guidance = Qwen35ToolGrammar.Build(tools, new Qwen35ToolGrammarOptions { FastForward = fastForward }),
        };
        var r = Model.Generate([Qwen35Message.User("What's the weather in Porto?")], options, Ct);
        Report(fastForward ? "tool grammar, fast-forward" : "tool grammar", r);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
        var parsed = Qwen35ToolCallParser.Parse(r.Text, options);
        var call = Assert.Single(parsed.ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal(string.Empty, parsed.Text);
    }

    private static void Report(string name, GenerationResult r) =>
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"{name}: {r.Usage.CompletionTokens} tokens, {r.FinishReason}: {Regex.Escape(r.Text)}");
}

/// <summary>fp32x on CUDA (TF32 off).</summary>
[Collection("cuda")]
public sealed class Fp32xCudaGuidanceTests(Fp32xCudaFixture fixture) : CudaGuidanceChecks, IClassFixture<Fp32xCudaFixture>
{
    protected override Qwen35ChatModel Model
    {
        get
        {
            Assert.SkipWhen(fixture.Model is null, fixture.Reason);
            return fixture.Model!;
        }
    }
}

/// <summary>fp16x on CUDA.</summary>
[Collection("cuda")]
public sealed class Fp16xCudaGuidanceTests(Fp16xCudaFixture fixture) : CudaGuidanceChecks, IClassFixture<Fp16xCudaFixture>
{
    protected override Qwen35ChatModel Model
    {
        get
        {
            Assert.SkipWhen(fixture.Model is null, fixture.Reason);
            return fixture.Model!;
        }
    }
}
