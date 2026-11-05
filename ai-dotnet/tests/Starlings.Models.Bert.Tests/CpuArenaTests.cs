namespace Starlings.Models.Bert.Tests;

/// <summary>
/// Flips a process-wide <see cref="AppContext"/> switch. The collection has parallelization disabled, so xunit runs it
/// on its own after the parallel ones and no other test can build a session while the switch is on.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ProcessSwitchTestGroup
{
    public const string Name = "process switch (not parallel)";
}

/// <summary>ONNX Runtime's CPU arena on this route: off unless the process turns it back on.</summary>
[Collection(ProcessSwitchTestGroup.Name)]
public sealed class CpuArenaTests
{
    private static bool? SessionArena() => BertSession.SessionOptions(threads: 2, expectedSha256: "digest", verifiedSha256: "digest").CpuMemoryArena;

    [Fact]
    public void TheArenaIsOffUnlessTheSwitchTurnsItOn()
    {
        Assert.False(SessionArena());

        AppContext.SetSwitch(BertSession.CpuMemoryArenaSwitch, true);
        try
        {
            Assert.True(SessionArena());
        }
        finally
        {
            AppContext.SetSwitch(BertSession.CpuMemoryArenaSwitch, false);
        }

        Assert.False(SessionArena());
    }
}
