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
    
    public unsafe T* Map(CLCommandQueue commandQueue, nuint size, nuint offset = 0, MapFlags flags = MapFlags.Read)
    {
        var memory = (T*)CL.EnqueueMapBuffer(commandQueue,
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

    public unsafe T[] ToArray(CLCommandQueue commandQueue, int length, nuint tSize)
    {
        var mapped = Map(commandQueue, (nuint)length * tSize);
        var array = new T[length];
        for (var i = 0; i < length; i++)
        {
            array[i] = mapped[i];
        }
        return array;
    }
    
    public unsafe void Print(CLCommandQueue commandQueue, int length, int tSize, int colLength = -1, Func<T, string?>? toString = null)
    {
        toString ??= arg => arg.ToString();
        
        if (colLength == -1) colLength = length;
        
        Console.Write('[');
        var buf = Map(commandQueue, (nuint)(length * tSize));
        for (var i = 0; i < length; i++)
        {
            if (i % colLength == 0)
            {
                Console.Write($"{Environment.NewLine}    ");
            }
            
            Console.Write($"{toString.Invoke(buf[i])}, ");
        }
        Console.WriteLine($"{Environment.NewLine}]");
    }
}