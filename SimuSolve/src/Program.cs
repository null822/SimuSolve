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
    
    public static void Main()
    {
        (_device, _context, _commandQueue) = ClUtil.InitializeOpenCl(Constants.PlatformIndex, Constants.DeviceIndex);
        Kernels.CreateKernels(_context, _device);
        
        CL.GetDeviceInfo(_device, DeviceInfo.Name, out var deviceNameBytes);
        Console.WriteLine($"Running on Device: {Encoding.UTF8.GetString(deviceNameBytes)}");
        
        const uint coeffCount = 50;
        var (inputCoefficients, inputConstTerms) = GetCoefficients(coeffCount, true);
        
        var inputString = new StringBuilder();
        for (var equationIndex = 0; equationIndex < coeffCount; equationIndex++)
        {
            var equationCoefficients = inputCoefficients[equationIndex];
            
            for (var coeffIndex = 0; coeffIndex < coeffCount; coeffIndex++)
            {
                inputString.Append($"{equationCoefficients[coeffIndex]:R}a_{{{coeffIndex}}} + ");
            }

            inputString.Remove(inputString.Length - 3, 3);

            inputString.Append($" = {inputConstTerms[equationIndex]:R} \\\\\n");
        }
        
        File.WriteAllText("input.tex", inputString.ToString());
        inputString.Clear();
        
        Console.WriteLine(inputString);

        
        var solutions = Solve(coeffCount, inputCoefficients, inputConstTerms);
        
        var outputString = new StringBuilder();
        for (var i = 0; i < coeffCount; i++)
        {
            var solution = solutions[i];
            outputString.Append($"a_{{{i}}} = {solution:N4} \\\\\n");
        }
        
        File.WriteAllText("output.tex", outputString.ToString());
        outputString.Clear();
    }
    
    private static double[] Solve(uint coeffCount, double[][] inputCoefficients, double[] inputConstTerms)
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
                coefficients[row * totalColCount + col + 1] = inputCoefficients[row][col];
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
            // debugging
            // coeffDestination.Clear(_commandQueue, coeffLength * sizeof(double));
            // Console.WriteLine($"{i} / {coeffCount}");
            
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
        
        // coeffSource.Print(_commandQueue, (int)coeffLength, sizeof(double), (int)totalColCount, d => d.ToString("+0.000000000000;-0.000000000000"));
        // scaleBuffer.Print(_commandQueue, (int)scaleLength, sizeof(double), 1, d => d.ToString("+0.000000000000;-0.000000000000"));
        
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

    private static (double[][] inputCoefficients, double[] inputConstTerms) GetCoefficients(uint coeffCount, bool randomCoefficients)
    {
        
        double[][] inputCoefficients; // length = coeffCount, coeffCount
        double[] inputConstTerms; // length = coeffCount

        if (randomCoefficients)
        {
            var random = new Random();
            
            inputCoefficients = new double[coeffCount][];
            inputConstTerms = new double[coeffCount];

            for (var row = 0; row < coeffCount; row++)
            {
                inputConstTerms[row] = random.NextDouble() * 1/random.NextDouble();
                inputCoefficients[row] = new double[coeffCount];
                for (var col = 0; col < coeffCount; col++)
                {
                    inputCoefficients[row][col] = random.NextDouble() * 1/random.NextDouble();
                }
            }
            
        }
        else switch (coeffCount)
        {
            case 1:
                inputCoefficients = [[4]];
                inputConstTerms = [2];
                /*
                 * Solution:
                 *
                 * a = 0.5
                 */
                break;
            case 3:
                inputCoefficients = [
                    [1, 2, 3],
                    [7, 7, 4],
                    [5, 3, 7],
                ];
                inputConstTerms = [
                    4,
                    9,
                    0
                ];
                
                /*
                 * Solutions:
                 * 
                 * a = -1.63492063492064
                 * b = +2.98412698412698
                 * c = -0.111111111111111
                 */
                break;
            case 4:
                inputCoefficients = [
                    [1, 2, 3, 4],
                    [7, 7, 4, 2],
                    [5, 3, 7, 1],
                    [3, 5, 9, 8],
                ];
                inputConstTerms = [
                    4,
                    9,
                    0,
                    5,
                ];
                
                /*
                 * Solutions:
                 * 
                 * a = +2.02919708029197
                 * b = -0.386861313868613
                 * c = -1.54744525547445
                 * d = +1.84671532846715
                 */
                break;
            case 5:
                inputCoefficients = [
                    [1, 2, 3, 4, 7],
                    [7, 7, 4, 2, 5],
                    [5, 3, 7, 1, 6],
                    [3, 5, 9, 8, 2],
                    [7, 6, 8, 4, 3],
                ];
                inputConstTerms = [
                    4,
                    9,
                    0,
                    5,
                    5,
                ];
                
                /*
                 * Solutions:
                 *
                 * a = -0.00122324159021403
                 * b = +1.48746177370031
                 * c = -0.86177370030581
                 * d = +0.625688073394495
                 * e = +0.158409785932722
                 */
                break;
            case 6:
                inputCoefficients =
                [
                    [7, 4, 6, 3, 2, 9],
                    [4, 4, 5, 3, 4, 8],
                    [6, 5, 5, 5, 4, 7],
                    [3, 6, 3, 1, 7, 3],
                    [2, 5, 2, 2, 3, 9],
                    [3, 5, 4, 5, 7, 4],
                ];
                inputConstTerms =
                [
                    2,
                    8,
                    5,
                    3,
                    2,
                    7,
                ];
                /*
                 * Solutions:
                 *
                 * a = -33.375
                 * b = +40.8916666666667
                 * c = +42.65
                 * d = +4.18333333333333
                 * e = -32.9833333333333
                 * f = -14.4916666666667
                 */
                break;
            case 7:
                inputCoefficients =
                [
                    [7, 4, 6, 3, 2, 9, 1],
                    [4, 4, 5, 3, 4, 8, 4],
                    [6, 5, 5, 5, 4, 7, 5],
                    [3, 6, 3, 1, 7, 3, 1],
                    [2, 5, 2, 2, 3, 9, 5],
                    [3, 5, 4, 5, 7, 4, 4],
                    [2, 7, 5, 6, 5, 1, 2],
                ];
                inputConstTerms =
                [
                    2,
                    8,
                    5,
                    3,
                    2,
                    7,
                    3,
                ];
                /*
                 * Solutions:
                 *
                 * a = -0.768620519565932
                 * b = -1.31548010522854
                 * c = +2.43546530746465
                 * d = -0.888770141400855
                 * e = +0.891072015784282
                 * f = -0.271662282144031
                 * g = +1.35859914501809
                 */
                break;
            case 8:
                inputCoefficients = [
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [1, 3, 9, 6, 8, 5, 7, 5],
                    [9, 5, 1, 7, 8, 1, 5, 3],
                    [3, 2, 5, 4, 0, 8, 2, 8],
                    [8, 5, 3, 7, 7, 6, 3, 2],
                    [4, 1, 2, 3, 5, 4, 4, 1],
                    [6, 3, 2, 1, 0, 0, 1, 5],
                    [5, 7, 1, 6, 6, 5, 2, 3],
                ];
                inputConstTerms = [
                    1,
                    6,
                    7,
                    9,
                    2,
                    2,
                    9,
                    2,
                ];
                /*
                 * Solutions:
                 *
                 * a = +0.467568267594169
                 * b = +12.4901391252868
                 * c = +1.70086213276104
                 * d = +7.41027159032043
                 * e = -23.7865388884778
                 * f = +0.31747206393843
                 * g = +23.1808998741952
                 * h = -13.0537445422926
                 */
                break;
            default: throw new Exception($"Coefficient count {coeffCount} does not have a predefined set of inputs");
        }

        return (inputCoefficients, inputConstTerms);
    }
}
