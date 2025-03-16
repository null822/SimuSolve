using System.Reflection;
using System.Text;
using OpenTK.Compute.OpenCL;

namespace CLObjects;

public class Kernel
{
    private readonly CLContext _context;
    private readonly CLDevice _device;
    
    private readonly CLProgram _program;
    private readonly CLKernel _kernel;

    private readonly Dictionary<string, uint> _argNames = [];

    public CLProgram ClProgram => _program;
    
    public CLKernel ClKernel => _kernel;
    public IntPtr Handle => _kernel.Handle;
    
    
    
    public Kernel(CLContext context, CLDevice device, string kernelFileName, string kernelName = "Main")
    {
        _context = context;
        _device = device;

        _program = CreateProgram(kernelFileName);
        _kernel = CreateKernel(_program, kernelName);
        
        LoadArguments();
    }
    
    public Kernel(CLContext context, CLDevice device, CLProgram program, string kernelName)
    {
        _context = context;
        _device = device;

        _program = program;
        _kernel = CreateKernel(program, kernelName);
        
        LoadArguments();
    }

    private void LoadArguments()
    {
        CL.GetKernelInfo(_kernel, KernelInfo.NumberOfArguments, out var countData);
        var count = BitConverter.ToInt32(countData);
        
        for (var i = 0u; i < count; i++)
        {
            CL.GetKernelArgInfo(_kernel, i, KernelArgInfo.Name, out var nameData);
            var name = Encoding.UTF8.GetString(nameData[..^1]); // remove null character at end
            _argNames.Add(name, i);
        }
    }
    
    public void SetArg<T>(uint index, T value) where T : unmanaged
    {
        CL.SetKernelArg(_kernel, index, value);
    }
    
    public void SetArg<T>(uint index, Buffer<T> value) where T : unmanaged
    {
        CL.SetKernelArg(_kernel, index, value.Handle);
    }
    
    public void SetArg<T>(string name, T value) where T : unmanaged => SetArg(_argNames[name], value);
    public void SetArg<T>(string name, Buffer<T> value) where T : unmanaged => SetArg(_argNames[name], value);

    public void EnqueueNdRanged(CLCommandQueue commandQueue,
        UIntPtr[] globalSize, UIntPtr[]? globalOffset = null, UIntPtr[]? localSize = null)
    {
        var dimensions = (uint)globalSize.Length;
        
        var enqueueKernelCode = CL.EnqueueNDRangeKernel(
            commandQueue, 
            _kernel,
            dimensions,
            globalOffset ?? new UIntPtr[dimensions],
            globalSize,
            localSize ?? new UIntPtr[dimensions].Select(_ => (UIntPtr)1).ToArray(),
            0,
            null,
            out _);
        
        ClException.ThrowIfNotSuccess(enqueueKernelCode, $"Failed to enqueue Kernel: {enqueueKernelCode}");
    }

    #region Kernel Creation
    
    private static CLKernel CreateKernel(CLProgram program, string kernelName)
    {
        var kernel = CL.CreateKernel(program, kernelName, out var kernelCode);
        ClException.ThrowIfNotSuccess(kernelCode, $"Failed to create kernel: {kernelCode}");
        return kernel;
    }
    
    private CLProgram CreateProgram(string kernelFileName)
    {
        var kernelString = ReadKernelString(kernelFileName);
        var program = CL.CreateProgramWithSource(_context, kernelString, out var programCode);
        ClException.ThrowIfNotSuccess(programCode, $"Failed to create CL program: {programCode}");
        
        var buildCode = CL.BuildProgram(program, 1, [_device], "-cl-kernel-arg-info", 0, 0);
        
        if (buildCode != CLResultCode.Success)
        {
            CL.GetProgramBuildInfo(program, _device, ProgramBuildInfo.Log, out var log);
            Console.WriteLine(Encoding.UTF8.GetString(log));
            
            throw new ClException($"Failed to build program: {buildCode}");
        }
        
        return program;
    }
    
    private static string ReadKernelString(string kernelName)
    {
        var assembly = Assembly.GetEntryAssembly() ?? throw new Exception("Unable to locate application Assembly");
        var stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.kernels.{kernelName}");
        if (stream == null) throw new Exception($"Unable to read kernel source {kernelName}");
        var reader = new StreamReader(stream, true);
        return reader.ReadToEnd();
    }
    
    #endregion
}