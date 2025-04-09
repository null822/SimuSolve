using System.Diagnostics;
using System.Numerics;
using System.Text;
using CLObjects;
using OpenTK.Compute.OpenCL;

namespace SimuSolve;

// TODO: non-power-of-2 coeff count optimisations

public static class Program
{
    private static CLDevice _device;
    
    private static CLContext _context;
    private static CLCommandQueue _commandQueue;
    
    public static void Main(string[] args)
    {
        (_device, _context, _commandQueue) = ClUtil.InitializeOpenCl(Constants.PlatformIndex, Constants.DeviceIndex);
        Kernels.CreateKernels(_context, _device);
        
        CL.GetDeviceInfo(_device, DeviceInfo.Name, out var deviceNameBytes);
        Console.WriteLine($"Running on Device: {Encoding.UTF8.GetString(deviceNameBytes)}");
        
        var inputPath =  args.Length >= 1 ? args[0] : "data/input.csv";
        var outputPath = args.Length >= 2 ? args[1] : "data/output.csv";
        
        // load input file
        var inputString = File.ReadAllText(inputPath).Replace(" ", "");
        var inputRows = inputString.Split('\n');
        var coeffCount = (uint)inputRows.Length;
        
        var coefficients = new double[coeffCount,coeffCount];
        var constTerms = new double[coeffCount];
        for (var row = 0; row < coeffCount; row++)
        {
            var inputRow = inputRows[row].Split(',');
            
            // const terms stored in the first column
            if (!double.TryParse(inputRow[0], out constTerms[row]))
                constTerms[row] = double.NaN;
            
            // all other coefficients are stored in the next columns
            for (var col = 0; col < coeffCount; col++)
            {
                if (!double.TryParse(inputRow[col + 1], out coefficients[row, col]))
                    coefficients[row, col] = double.NaN;
            }
            
        }
        
        // solve the simultaneous equations
        var solutions = Solve(coeffCount, coefficients, constTerms);
        
        
        if (solutions.Length <= 64)
        {
            Console.WriteLine("Solutions: ");
            for (var i = 0; i < solutions.Length; i++)
            {
                Console.WriteLine($"a_{i} = {solutions[i]:g4}");
            }
        }
        
        // write solutions to output file
        var outputString = new StringBuilder();
        for (var i = 0; i < coeffCount; i++)
        {
            var solution = solutions[i];
            outputString.Append($"{solution:R}\n");
        }
        outputString.Remove(outputString.Length - 1, 1); // remove trailing newline
        File.WriteAllText(outputPath, outputString.ToString());
        outputString.Clear();
        
        Console.WriteLine($"Solutions Saved to \"{outputPath}\"");
    }
    
    private static double[] Solve(uint coeffCount, double[,] inputCoefficients, double[] inputConstTerms)
    {
        var maxCoeffCount = BitOperations.RoundUpToPowerOf2(coeffCount);
        
        var totalRowCount = maxCoeffCount * 2;
        var totalColCount = coeffCount + 1;
        
        var coeffLength = totalColCount * totalRowCount;
        
        
        // pack data into coefficient buffer
        var coefficients = new double[coeffLength];
        for (var row = 0; row < coeffCount; row++)
        {
            coefficients[row * totalColCount] = inputConstTerms[row];
            for (var col = 0; col < coeffCount; col++)
            {
                coefficients[row * totalColCount + col + 1] = inputCoefficients[row, col];
            }
        }
        
        
        var s = new Stopwatch();
        s.Start();
        
        // create buffers
        var scaleBuffer = new Buffer<double>(_context, MemoryFlags.HostReadOnly, totalRowCount * sizeof(double));
        var coeffBuffer1 = new Buffer<double>(_context, MemoryFlags.CopyHostPtr, coefficients);
        var coeffBuffer2 = new Buffer<double>(_context, MemoryFlags.HostReadOnly, coeffLength * sizeof(double));
        
        // set constant kernel arguments
        Kernels.Splitter.SetArg("totalColCount", totalColCount);
        Kernels.ScaleCalculator.SetArg("scaleBuffer", scaleBuffer);
        Kernels.ScaleCalculator.SetArg("totalColCount", totalColCount);
        Kernels.Scaler.SetArg("scaleBuffer", scaleBuffer);
        Kernels.Scaler.SetArg("totalColCount", totalColCount);
        Kernels.Eliminator.SetArg("colCount", totalColCount);
        Kernels.UnknownSolver.SetArg("totalColCount", totalColCount);
        Kernels.ResultCopier.SetArg("outputBuffer", scaleBuffer);
        Kernels.ResultCopier.SetArg("totalColCount", totalColCount);
        Kernels.ResultCopier.SetArg("center", maxCoeffCount / 2);
        Kernels.ResultCopier.SetArg("coeffCountDiff", maxCoeffCount - coeffCount);
        
        
        var coeffSource = coeffBuffer1; // the source buffer
        var coeffDestination = coeffBuffer2; // the destination buffer

        var boundaryRowCount = maxCoeffCount * 2; // amount of rows between adjacent blocks
        var blockCount = 1u; // amount of blocks
        var rowCount = coeffCount; // amount of rows in a block
        var colCount = totalColCount; // amount of cols in a block
        
        // force a premature split if the coeff count is not a power of 2
        if (!BitOperations.IsPow2(coeffCount))
        {
            Kernels.Splitter.SetArg("buffer", coeffSource);
            Kernels.Splitter.SetArg("rowCount", maxCoeffCount);
            Kernels.Splitter.SetArg("colCount", totalColCount);
            Kernels.Splitter.EnqueueNdRanged(_commandQueue, [1, coeffCount, totalColCount]);
            
            blockCount *= 2;
            boundaryRowCount /= 2;
        }
        
        for (var i = 0; i < coeffCount - 1; i++)
        {
            // fork on powers of 2
            if (BitOperations.IsPow2(rowCount))
            {
                Kernels.Splitter.SetArg("buffer", coeffSource);
                Kernels.Splitter.SetArg("rowCount", rowCount);
                Kernels.Splitter.SetArg("colCount", colCount);
                
                Kernels.Splitter.EnqueueNdRanged(_commandQueue, [blockCount, rowCount, colCount]);
                
                blockCount *= 2;
                boundaryRowCount /= 2;
            }
            
            // calculate scale value for each row in each block
            Kernels.ScaleCalculator.SetArg("srcBuffer", coeffSource);
            Kernels.ScaleCalculator.SetArg("boundaryRowCount", boundaryRowCount);
            Kernels.ScaleCalculator.SetArg("targetValue", colCount - 1); // target the last term in each row
            Kernels.ScaleCalculator.EnqueueNdRanged(_commandQueue, [blockCount, rowCount]);
            
            // apply scale to all actively used coefficients in every row in each block
            Kernels.Scaler.SetArg("srcBuffer", coeffSource);
            Kernels.Scaler.SetArg("boundaryRowCount", boundaryRowCount);
            Kernels.Scaler.EnqueueNdRanged(_commandQueue, [blockCount, rowCount, colCount]);
            
            // shrink the block size by 1 in both dimensions. this is done here to make the eliminator not do extra work
            rowCount--;
            colCount--;
            
            // subtract rows (equations) to eliminate the last term in each row
            Kernels.Eliminator.SetArg("srcBuffer", coeffSource);
            Kernels.Eliminator.SetArg("desBuffer", coeffDestination);
            Kernels.Eliminator.SetArg("boundaryRowCount", boundaryRowCount);
            Kernels.Eliminator.EnqueueNdRanged(_commandQueue, [blockCount, rowCount, colCount]);
            
            (coeffSource, coeffDestination) = (coeffDestination, coeffSource);
        }
        
        Kernels.UnknownSolver.SetArg("valueBuffer", coeffSource);
        Kernels.UnknownSolver.EnqueueNdRanged(_commandQueue, [coeffCount]); // maxCoeffCount = blockCount here
        
        Kernels.ResultCopier.SetArg("coeffBuffer", coeffSource);
        Kernels.ResultCopier.EnqueueNdRanged(_commandQueue, [coeffCount]);
        
        s.Stop();
        Console.WriteLine($"Time: {s.Elapsed.TotalMilliseconds}ms");
        Console.WriteLine("Copying Results from GPU Memory");
        var results = scaleBuffer.ToArray(_commandQueue, (int)coeffCount, sizeof(double));
        
        return results;
    }
}
