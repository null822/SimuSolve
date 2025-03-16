using OpenTK.Compute.OpenCL;

namespace CLObjects;

public static class ClUtil
{
    public static (CLDevice, CLContext, CLCommandQueue) InitializeOpenCl(int platformIndex, int deviceIndex, IntPtr commandQueueProperties = 0)
    {
        CL.GetPlatformIds(out var platforms);
        var platform = platforms[platformIndex];
        
        CL.GetDeviceIds(platform, DeviceType.All, out var devices);
        var device = devices[deviceIndex];
        
        var context = CL.CreateContext([], [device], 0, 0, out var contextCode);
        if (contextCode != CLResultCode.Success) throw new Exception($"Failed to create CL Context: {contextCode}");
        
        var commandQueue = CL.CreateCommandQueueWithProperties(context, device, commandQueueProperties, out var commandQueueCode);
        if (commandQueueCode != CLResultCode.Success) throw new Exception($"Failed to create CL Command Queue: {commandQueueCode}");

        return (device, context, commandQueue);
    }
}