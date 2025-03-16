using OpenTK.Compute.OpenCL;

namespace CLObjects;

public class ClException(string message) : Exception(message)
{
    public static void ThrowIfNotSuccess(CLResultCode code, string message)
    {
        if (code != CLResultCode.Success)
            throw new ClException(message);
    }
}