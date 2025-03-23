using CLObjects;
using OpenTK.Compute.OpenCL;

namespace SimuSolve;

public static class Kernels
{
    public static Kernel Splitter { get; private set; } = null!;
    public static Kernel ScaleCalculator { get; private set; } = null!;
    public static Kernel Scaler { get; private set; } = null!;
    public static Kernel Eliminator { get; private set; } = null!;
    public static Kernel UnknownSolver { get; private set; } = null!;
    public static Kernel ResultCopier { get; private set; } = null!;
    
    public static Kernel BufferCleaner { get; private set; } = null!;
    
    public static void CreateKernels(CLContext context, CLDevice device)
    {
        Splitter = new Kernel(context, device, "splitter.cl");
        ScaleCalculator = new Kernel(context, device, "scale_calculator.cl");
        Scaler = new Kernel(context, device, "scaler.cl");
        Eliminator = new Kernel(context, device, "eliminator.cl");
        UnknownSolver = new Kernel(context, device, "unknown_solver.cl");
        ResultCopier = new Kernel(context, device, "result_copier.cl");
        
        BufferCleaner = new Kernel(context, device, "buffer_cleaner.cl");
    }

    public static void Clear<T>(this Buffer<T> buffer, CLCommandQueue commandQueue, nuint size) where T : unmanaged
    {
        BufferCleaner.SetArg("buffer", buffer);
        BufferCleaner.EnqueueNdRanged(commandQueue, [size]);
    }
}