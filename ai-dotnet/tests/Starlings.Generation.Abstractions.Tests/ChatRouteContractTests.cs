using System.Text.Json.Nodes;

namespace Starlings.Generation.Tests;

/// <summary>The contract table parser's refusals, on an invented family so nothing depends on one model.</summary>
public sealed class ChatRouteContractTests
{
    private const string Repository = "example/Tiny-Chat";

    private static JsonObject Table() => JsonNode.Parse($$"""
        {
          "format": "starlings.genai-chat-route-contracts",
          "format_version": 1,
          "route": "genai-chat-cpu-v1",
          "contracts": {
            "{{Repository}}": {
              "id": "tiny-chat/ortgenai-fp32-cpu",
              "revision": "{{new string('a', 40)}}",
              "behavior": {
                "architecture": "TinyForCausalLM",
                "model_type": "tiny",
                "config": {
                  "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                  "intermediate_size": 16, "max_position_embeddings": 64, "original_max_position_embeddings": 64,
                  "partial_rotary_factor": 1.0, "rope_type": "default", "tie_word_embeddings": false
                },
                "vocab_size": 100,
                "eos_token_ids": [1, 2],
                "add_bos_token": true,
                "pad_token_id_overlay": null,
                "pad_overlay_reason": null,
                "tokenizer_profile": "tiny-bpe",
                "template": "tiny-text-v1",
                "template_sha256": "{{new string('b', 64)}}",
                "max_new_tokens": 8,
                "max_context_tokens": 32,
                "cache": "fresh-request",
                "tools": "experimental",
                "thinking": "unsupported",
                "multimodal": "unsupported",
                "long_context": "untested"
              }
            }
          }
        }
        """)!.AsObject();

    private static JsonObject Behavior(JsonObject table) => table["contracts"]![Repository]!["behavior"]!.AsObject();

    private static ChatRouteContract Parse(JsonObject table) => ChatRouteContract.ParseTable(table.ToJsonString())[Repository];

    [Fact]
    public void ParsesAnEntryWithoutAPadOverlay()
    {
        ChatRouteContract contract = Parse(Table());
        Assert.Equal("tiny-chat/ortgenai-fp32-cpu", contract.Id);
        Assert.Equal("tiny", contract.ModelType);
        Assert.Equal([1, 2], contract.EosTokenIds);
        Assert.Null(contract.PadTokenIdOverlay);
        Assert.Equal(4, contract.HeadSize);
        Assert.Equal(64, contract.ContextLength);
        Assert.Equal("experimental", contract.Tools);
        Assert.Equal("untested", contract.LongContext);
    }

    [Fact]
    public void ParsesAPadOverlayWithItsReason()
    {
        JsonObject table = Table();
        Behavior(table)["pad_token_id_overlay"] = 99;
        Behavior(table)["pad_overlay_reason"] = "unused row";
        Assert.Equal(99, Parse(table).PadTokenIdOverlay);
    }

    public static TheoryData<string> Breaks =>
    [
        "format", "version", "route", "empty", "unknown-entry-field", "short-revision", "other-profile-id", "missing-behavior-field",
        "unknown-config-field", "string-vocabulary", "empty-eos", "eos-outside", "pad-outside", "pad-is-eos", "pad-without-reason",
        "reason-without-pad", "bad-digest", "status-missing", "status-unknown", "long-context-bound",
    ];

    [Theory]
    [MemberData(nameof(Breaks))]
    public void RefusesABrokenTable(string breakage)
    {
        JsonObject table = Table();
        JsonObject behavior = Behavior(table);
        JsonObject entry = table["contracts"]![Repository]!.AsObject();
        switch (breakage)
        {
            case "format": table["format"] = "other"; break;
            case "version": table["format_version"] = 2; break;
            case "route": table["route"] = "other"; break;
            case "empty": table["contracts"] = new JsonObject(); break;
            case "unknown-entry-field": entry["extra"] = 1; break;
            case "short-revision": entry["revision"] = "abc"; break;
            case "other-profile-id": entry["id"] = "tiny-chat/onnx-fp32-cpu"; break;
            case "missing-behavior-field": behavior.Remove("cache"); break;
            case "unknown-config-field": behavior["config"]!["extra"] = 1; break;
            case "string-vocabulary": behavior["vocab_size"] = "100"; break;
            case "empty-eos": behavior["eos_token_ids"] = new JsonArray(); break;
            case "eos-outside": behavior["eos_token_ids"] = new JsonArray(100); break;
            case "pad-outside": behavior["pad_token_id_overlay"] = 100; behavior["pad_overlay_reason"] = "x"; break;
            case "pad-is-eos": behavior["pad_token_id_overlay"] = 1; behavior["pad_overlay_reason"] = "x"; break;
            case "pad-without-reason": behavior["pad_token_id_overlay"] = 99; break;
            case "reason-without-pad": behavior["pad_overlay_reason"] = "x"; break;
            case "bad-digest": behavior["template_sha256"] = new string('B', 64); break;
            case "status-missing": behavior.Remove("tools"); break;
            case "status-unknown": behavior["tools"] = "maybe"; break;
            case "long-context-bound": behavior["max_context_tokens"] = 65; break;
            default: throw new ArgumentOutOfRangeException(nameof(breakage));
        }

        Assert.Throws<InvalidDataException>(() => Parse(table));
    }

    [Fact]
    public void RefusesDuplicateKeysAndInvalidJson()
    {
        string duplicate = Table().ToJsonString().Replace("\"vocab_size\":100", "\"vocab_size\":100,\"vocab_size\":100", StringComparison.Ordinal);
        Assert.Throws<InvalidDataException>(() => ChatRouteContract.ParseTable(duplicate));
        Assert.Throws<InvalidDataException>(() => ChatRouteContract.ParseTable("{"));
    }
}
