using System.Diagnostics;
using System.Text;
using CLObjects;
using OpenTK.Compute.OpenCL;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Mandelbrot;

public static class Program
{
    private static CLDevice _device;
    
    private static CLContext _context;
    private static CLCommandQueue _commandQueue;
    
    public static unsafe void Main()
    {
        (_device, _context, _commandQueue) = ClUtil.InitializeOpenCl(Constants.PlatformIndex, Constants.DeviceIndex);
        
        CL.GetDeviceInfo(_device, DeviceInfo.Name, out var deviceNameBytes);
        Console.WriteLine($"Running on Device: {Encoding.UTF8.GetString(deviceNameBytes)}");
        
        const ulong totalPixels = Constants.Width * Constants.Height;
        
        var kernel = new Kernel(_context, _device, "mandelbrot.cl", "Mandelbrot");
        var buffer = new Buffer<double>(_context, MemoryFlags.ReadWrite, checked((UIntPtr)(totalPixels * sizeof(double))));
        
        kernel.SetArg(0, Constants.X0);
        kernel.SetArg(1, Constants.Y0);
        kernel.SetArg(2, Constants.X1);
        kernel.SetArg(3, Constants.Y1);
        kernel.SetArg(4, (int)Constants.Width);
        kernel.SetArg(5, (int)Constants.Height);
        kernel.SetArg(6, (int)Constants.MaxIterations);
        kernel.SetArg(7, buffer.Handle);
        
        Console.WriteLine("Running Kernel");
        var s = new Stopwatch();
        s.Start();
        
        kernel.EnqueueNdRanged(_commandQueue, [Constants.Width, Constants.Height]);
        
        s.Stop();
        Console.WriteLine($"Kernel Finished in {s.Elapsed.TotalMicroseconds:N}us");
        Console.WriteLine($"Downloading data from GPU");
        s.Restart();
        
        var buf = buffer.Map<double>(_commandQueue, checked((UIntPtr)(totalPixels * sizeof(double))));
        
        s.Stop();
        Console.WriteLine($"Image downloaded from GPU in {s.Elapsed.TotalMicroseconds:N}us");
        Console.WriteLine("Calculating Colors");
        s.Restart();
        
        var colors = new byte[totalPixels];
        for (var i = 0uL; i < totalPixels; i++)
        {
            var value = buf[i];
            // var v =
                // (((value >> 0) & 0x1) << 7) |
                // (((value >> 1) & 0x1) << 6) |
                // (((value >> 2) & 0x1) << 5) |
                // (((value >> 3) & 0x1) << 4) |
                // (((value >> 4) & 0x1) << 3) |
                // (((value >> 5) & 0x1) << 2) |
                // (((value >> 6) & 0x1) << 1) |
                // (((value >> 7) & 0x1) << 0);
            
            // var l = v / 255f;

            colors[i] = (byte)(Math.Log(value, 1.0001) * 256);
        }
        s.Stop();
        Console.WriteLine($"Color Calculation took {s.Elapsed.TotalMicroseconds:N}us");
        
        var image = Image.LoadPixelData<L8>(Configuration.Default, colors, (int)Constants.Width, (int)Constants.Height);
        image.SaveAsPng("mandelbrot.png");
    }
}
