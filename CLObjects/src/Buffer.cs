using OpenTK.Compute.OpenCL;

namespace CLObjects;

public class Buffer<T> where T : unmanaged
{
    private readonly CLBuffer _buffer;

    public CLBuffer ClBuffer => _buffer;
    public IntPtr Handle => _buffer.Handle;
    
    public Buffer(CLContext context, MemoryFlags flags, UIntPtr size)
    {
        _buffer = CL.CreateBuffer(context,
            flags,
            size,
            0,
            out var bufferCode);
        ClException.ThrowIfNotSuccess(bufferCode, $"Failed to create CL Buffer: {bufferCode}");
    }
    
    public Buffer(CLContext context, MemoryFlags flags, T[] contents)
    {
        _buffer = CL.CreateBuffer(context,
            flags,
            contents,
            out var bufferCode);
        ClException.ThrowIfNotSuccess(bufferCode, $"Failed to create CL Buffer: {bufferCode}");
    }
    
    public unsafe TMapped* Map<TMapped>(CLCommandQueue commandQueue, nuint size, nuint offset = 0, MapFlags flags = MapFlags.Read)
        where TMapped : unmanaged
    {
        var memory = (TMapped*)CL.EnqueueMapBuffer(commandQueue,
            _buffer,
            true,
            flags,
            offset,
            size,
            0,
            null,
            out _,
            out var mapBufferCode).ToPointer();
        ClException.ThrowIfNotSuccess(mapBufferCode, $"Failed to map CL buffer: {mapBufferCode}");

        return memory;
    }
}