namespace Starlings.Models.Bert.Tests;

/// <summary>ONNX Runtime's CPU arena on this route: off unless the process turns it back on.</summary>
public sealed class CpuArenaTests
{
    [Fact]
    public void TheArenaIsOffUnlessTheSwitchTurnsItOn()
    {
        Assert.False(BertSession.CpuMemoryArena());

        AppContext.SetSwitch(BertSession.CpuMemoryArenaSwitch, true);
        try
        {
            Assert.True(BertSession.CpuMemoryArena());
        }
        finally
        {
            AppContext.SetSwitch(BertSession.CpuMemoryArenaSwitch, false);
        }

        Assert.False(BertSession.CpuMemoryArena());
    }
}
