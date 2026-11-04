using System.Text.Json;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>
/// The IChatClient on a model on CPU, from the variant directory as the lock lays it out (with ONNX Runtime GenAI's
/// guidance files since hosting revision r2). Skips without the model.
/// </summary>
public abstract class Qwen35GuidedChatClientFixtureBase : IDisposable
{
    protected Qwen35GuidedChatClientFixtureBase(Qwen35Variant variant)
    {
        var (files, reason) = Qwen35TestAssets.FindModel(variant);
        Reason = reason;
        if (files is not null)
        {
            Client = Qwen35ChatClient.Create(files);
        }
    }

    public Qwen35ChatClient? Client { get; }

    public string Reason { get; }

    public void Dispose()
    {
        Client?.Dispose();
        GC.SuppressFinalize(this);
    }
}

/// <summary>The fp32x model.</summary>
public sealed class Qwen35GuidedChatClientFixture() : Qwen35GuidedChatClientFixtureBase(Qwen35Variant.Fp32x);

/// <summary>The int4 model (profile qwen3.5-0.8b/ortgenai-int4-cpu, issue #163).</summary>
public sealed class Qwen35Int4GuidedChatClientFixture() : Qwen35GuidedChatClientFixtureBase(Qwen35Variant.Int4);

/// <summary>Runs alone, after the parallel classes have released their models: its fixture loads another model (fp32x 3.1 GB, int4 about 0.58 GB).</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GuidedModelTestGroup
{
    public const string Name = "guided model (not parallel)";
}

/// <summary>Guided IChatClient requests on the real model (greedy; outputs are checked against the constraint).</summary>
public abstract class GuidedRealModelChecks(Qwen35GuidedChatClientFixtureBase fixture)
{
    private Qwen35ChatClient Client
    {
        get
        {
            Assert.SkipWhen(fixture.Client is null, fixture.Reason);
            return fixture.Client!;
        }
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    /// <summary>Writes the constrained output to the test log, so the evidence records what the model returned.</summary>
    private static void Record(string request, ChatResponse response)
    {
        string contents = string.Join(" | ", response.Messages.SelectMany(m => m.Contents).Select(c => c switch
        {
            FunctionCallContent call => $"call {call.Name}({JsonSerializer.Serialize(call.Arguments)})",
            TextContent text => $"text {JsonSerializer.Serialize(text.Text)}",
            _ => c.GetType().Name,
        }));
        TestContext.Current.TestOutputHelper?.WriteLine($"{request}: finish={response.FinishReason}; {contents}");
    }

    [Fact]
    public async Task RequireAnyReturnsAFunctionCall()
    {
        var weather = AIFunctionFactory.Create((string city) => $"Sunny in {city}", "get_weather", "Current weather for a city.");
        var time = AIFunctionFactory.Create(() => "12:00", "get_time", "The current time.");
        var response = await Client.GetResponseAsync(
            "What's the weather in Porto?",
            new ChatOptions { Tools = [weather, time], ToolMode = ChatToolMode.RequireAny, MaxOutputTokens = 64 },
            Ct);
        Record("RequireAny", response);
        var call = Assert.IsType<FunctionCallContent>(Assert.Single(response.Messages.Single().Contents));
        Assert.Equal("get_weather", call.Name);
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
    }

    [Fact]
    public async Task RequireSpecificReturnsACallOfTheNamedFunction()
    {
        var weather = AIFunctionFactory.Create((string city) => $"Sunny in {city}", "get_weather", "Current weather for a city.");
        var time = AIFunctionFactory.Create(() => "12:00", "get_time", "The current time.");
        var response = await Client.GetResponseAsync(
            "What time is it in Porto?",
            new ChatOptions { Tools = [weather, time], ToolMode = ChatToolMode.RequireSpecific("get_weather"), MaxOutputTokens = 64 },
            Ct);
        Record("RequireSpecific(get_weather)", response);
        var call = Assert.IsType<FunctionCallContent>(Assert.Single(response.Messages.Single().Contents));
        Assert.Equal("get_weather", call.Name);
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
    }

    [Fact]
    public async Task AJsonSchemaResponseFormatReturnsMatchingJson()
    {
        var schema = JsonDocument.Parse("""{"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": false}""").RootElement;
        var response = await Client.GetResponseAsync(
            "What is 6 times 7? Reply as JSON.",
            new ChatOptions { ResponseFormat = ChatResponseFormat.ForJsonSchema(schema), MaxOutputTokens = 32 },
            Ct);
        Record("JSON schema ResponseFormat", response);
        using var json = JsonDocument.Parse(response.Text);
        Assert.Equal(JsonValueKind.Number, json.RootElement.GetProperty("answer").ValueKind);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
    }
}

/// <summary>The fp32x model.</summary>
[Collection(GuidedModelTestGroup.Name)]
public sealed class GuidedRealModelTests(Qwen35GuidedChatClientFixture fixture)
    : GuidedRealModelChecks(fixture), IClassFixture<Qwen35GuidedChatClientFixture>;

/// <summary>The int4 model: the same constraints as fp32x, with no tolerance or quality floor.</summary>
[Collection(GuidedModelTestGroup.Name)]
public sealed class Int4GuidedRealModelTests(Qwen35Int4GuidedChatClientFixture fixture)
    : GuidedRealModelChecks(fixture), IClassFixture<Qwen35Int4GuidedChatClientFixture>;
