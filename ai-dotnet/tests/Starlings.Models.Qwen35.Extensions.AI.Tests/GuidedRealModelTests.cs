using System.Text.Json;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>
/// The IChatClient on the fp32x model on CPU, from the variant directory as the lock lays it out (with ONNX Runtime
/// GenAI's guidance files since hosting revision r2). Skips without the model.
/// </summary>
public sealed class Qwen35GuidedChatClientFixture : IDisposable
{
    public Qwen35GuidedChatClientFixture()
    {
        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Fp32x);
        Reason = reason;
        if (files is not null)
        {
            Client = Qwen35ChatClient.Create(files);
        }
    }

    public Qwen35ChatClient? Client { get; }

    public string Reason { get; }

    public void Dispose() => Client?.Dispose();
}

/// <summary>Runs alone, after the parallel classes have released their models: its fixture loads another 3.1 GB model.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GuidedModelTestGroup
{
    public const string Name = "guided model (not parallel)";
}

/// <summary>Guided IChatClient requests on the real model (greedy; outputs are checked against the constraint).</summary>
[Collection(GuidedModelTestGroup.Name)]
public sealed class GuidedRealModelTests(Qwen35GuidedChatClientFixture fixture) : IClassFixture<Qwen35GuidedChatClientFixture>
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
