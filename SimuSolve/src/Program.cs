using System.Numerics;
using System.Text;
using CLObjects;
using OpenTK.Compute.OpenCL;

namespace SimuSolve;

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
        
        const int coeffCount = Constants.CoeffCount;
        const int totalColCount = coeffCount + 1;
        const int totalRowCount = coeffCount * 2;
        
        const int scaleLength = 2 * coeffCount; // 2n
        const int coeffLength = 2 * coeffCount * coeffCount + 2 * coeffCount; // 2n^2 + 2n
        
        
        double[] coefficients = [ // length = coeffLength
         // Z  = Aa + Bb + Cc + Dd
            4,   1,   2,   3,   4,
            9,   7,   7,   4,   2,
            0,   5,   3,   7,   1,
            5,   3,   5,   9,   8,
            
            // second (initially empty) section
            0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,
        ];
        
        
        
        /*
         * a =  2.02919708029197
         * b = -0.386861313868613
         * c = -1.54744525547445
         * d =  1.84671532846715
         */
        
        // create buffers
        var scaleBuffer = new Buffer<double>(_context, MemoryFlags.HostReadOnly, scaleLength * sizeof(double)); // 2n
        var coeffBuffer1 = new Buffer<double>(_context, MemoryFlags.CopyHostPtr, coefficients); // 2n^2 + 2n
        var coeffBuffer2 = new Buffer<double>(_context, MemoryFlags.HostReadOnly, coeffLength * sizeof(double)); // 2n^2 + 2n
        
        // create kernels
        var splitterKernel = new Kernel(_context, _device, "splitter.cl");
        var scaleCalculatorKernel = new Kernel(_context, _device, "scale_calculator.cl");
        var scalerKernel = new Kernel(_context, _device, "scaler.cl");
        var eliminatorKernel = new Kernel(_context, _device, "eliminator.cl");
        var unknownSolverKernel = new Kernel(_context, _device, "unknown_solver.cl");
        
        
        // set constant kernel arguments
        
        splitterKernel.SetArg("totalColCount", totalColCount);
        
        scaleCalculatorKernel.SetArg("scaleBuffer", scaleBuffer);
        scaleCalculatorKernel.SetArg("totalColCount", totalColCount);
        
        unknownSolverKernel.SetArg("totalColCount", totalColCount);
        
        scalerKernel.SetArg("scaleBuffer", scaleBuffer);
        scalerKernel.SetArg("totalColCount", totalColCount);
        
        eliminatorKernel.SetArg("colCount", totalColCount);

        
        var coeffSource = coeffBuffer1;
        var coeffDestination = coeffBuffer2;

        var blockCount = (uint)1;
        var rowCount = (uint)coeffCount;
        var colCount = (uint)totalColCount;

        for (var i = 0; i < coeffCount - 1; i++)
        {
            // fork on powers of 2
            if (BitOperations.IsPow2(rowCount))
            {
                splitterKernel.SetArg("buffer", coeffSource);
                splitterKernel.SetArg("rowCount", rowCount);
                splitterKernel.SetArg("colCount", colCount);
                
                splitterKernel.EnqueueNdRanged(_commandQueue, [blockCount, rowCount, colCount]);
                
                blockCount *= 2;
            }
            
            // calculate scale value for each row
            scaleCalculatorKernel.SetArg("valueBuffer", coeffSource);
            scaleCalculatorKernel.SetArg("rowCount", rowCount);
            scaleCalculatorKernel.SetArg("targetValue", totalColCount-1 - i); // target the last term in each row
            scaleCalculatorKernel.EnqueueNdRanged(_commandQueue, [blockCount, rowCount]);
            
            
            // apply scale to all actively used coefficients in every row
            scalerKernel.SetArg("valueBuffer", coeffSource);
            scalerKernel.SetArg("rowCount", rowCount);
            scalerKernel.EnqueueNdRanged(_commandQueue, [blockCount, rowCount, colCount]);
            
            // shrink the block size by 1 in both dimensions. this is done here to make the eliminator not do extra work
            rowCount--;
            colCount--;
            
            // subtract rows (equations) to eliminate the last term in each row
            eliminatorKernel.SetArg("srcBuffer", coeffSource);
            eliminatorKernel.SetArg("desBuffer", coeffDestination);
            eliminatorKernel.SetArg("rowCount", rowCount);
            eliminatorKernel.EnqueueNdRanged(_commandQueue, [blockCount, rowCount, colCount]);
            
            (coeffSource, coeffDestination) = (coeffDestination, coeffSource);
        }
        
        unknownSolverKernel.SetArg("valueBuffer", coeffSource);
        unknownSolverKernel.EnqueueNdRanged(_commandQueue, [blockCount]);
        
        Console.WriteLine($"blocks: {blockCount}, rows: {rowCount}, cols: {colCount}");
        
        Console.WriteLine("Scale: ");
        PrintBuffer(scaleBuffer, scaleLength, sizeof(double));
        
        Console.WriteLine("Coeffs: ");
        PrintBuffer(coeffSource, coeffLength, sizeof(double), totalColCount);
        
        var buf = coeffSource.Map<double>(_commandQueue, coeffLength * sizeof(double));

        for (var i = 0; i < coeffCount; i++)
        {
            var solution = buf[i * 2 * totalColCount];
            var coefficient = (char)('a' + i);
            Console.WriteLine($"{coefficient} = {solution}");
        }

    }

    private static unsafe void PrintBuffer<T>(Buffer<T> buffer, int length, int tSize, int colCount = -1) where T : unmanaged
    {
        if (colCount == -1) colCount = length;
        
        Console.Write('[');
        var buf = buffer.Map<T>(_commandQueue, (nuint)(length * tSize));
        for (var i = 0; i < length; i++)
        {
            if (i % colCount == 0)
            {
                Console.Write($"{Environment.NewLine}    ");
            }
            
            Console.Write($"{buf[i]}, ");
        }
        Console.WriteLine($"{Environment.NewLine}]");
    }
    
    private static unsafe void PrintEquations(Buffer<double> coeffBuffer, int coeffCount)
    {
        var colCount = coeffCount + 1;
        
        var coeffSize = (nuint)((2 * coeffCount * coeffCount + 2 * coeffCount) * sizeof(double));
        var coeffs = coeffBuffer.Map<double>(_commandQueue, coeffSize);
        
        for (var row = 0; row < coeffCount * 2; row++)
        {
            var equation = new StringBuilder();
            for (var col = 0; col < coeffCount; col++)
            {
                var coeff = coeffs[((row * colCount) + col + 1)]; // +1 since const terms are at index 0
                
                var signCoeff = coeff < 0 ? '-' : '+';
                var absCoeff = Math.Abs(coeff);
                
                equation.Append($"{signCoeff} {absCoeff:0.000}# ");
            }

            var constTerm = coeffs[row * colCount + 0];
            var signConstTerm = constTerm < 0 ? '-' : '+';
            var absConstTerm = Math.Abs(constTerm);
            equation.Append($"= {signConstTerm}{absConstTerm:0.000}");
            
            Console.WriteLine(equation);
        }
        
    }
}
