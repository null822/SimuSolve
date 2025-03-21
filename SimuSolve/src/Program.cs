﻿using System.Numerics;
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
        const int coeffLength = totalColCount * totalRowCount; // 2n(n + 1) = 2n^2 + 2n
        
        // TODO: make non-power-of-2 coefficient counts work
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
         * a = +2.02919708029197
         * b = -0.386861313868613
         * c = -1.54744525547445
         * d = +1.84671532846715
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
        
        var blockRowCount = coeffCount * 2;

        for (var i = 0; i < coeffCount - 1; i++)
        {
            // fork on powers of 2
            if (BitOperations.IsPow2(rowCount))
            {
                splitterKernel.SetArg("buffer", coeffSource);
                splitterKernel.SetArg("rowCount", rowCount);
                
                splitterKernel.EnqueueNdRanged(_commandQueue, [blockCount, rowCount, colCount]);
                
                blockCount *= 2;
                blockRowCount /= 2;
            }
            
            // calculate scale value for each row
            scaleCalculatorKernel.SetArg("srcBuffer", coeffSource);
            scaleCalculatorKernel.SetArg("rowCount", blockRowCount);
            scaleCalculatorKernel.SetArg("targetValue", colCount - 1); // target the last term in each row
            scaleCalculatorKernel.EnqueueNdRanged(_commandQueue, [blockCount, rowCount]);
            
            // apply scale to all actively used coefficients in every row
            scalerKernel.SetArg("srcBuffer", coeffSource);
            scalerKernel.SetArg("rowCount", blockRowCount);
            scalerKernel.EnqueueNdRanged(_commandQueue, [blockCount, rowCount, colCount]);
            
            // shrink the block size by 1 in both dimensions. this is done here to make the eliminator not do extra work
            rowCount--;
            colCount--;
            
            // subtract rows (equations) to eliminate the last term in each row
            eliminatorKernel.SetArg("srcBuffer", coeffSource);
            eliminatorKernel.SetArg("desBuffer", coeffDestination);
            eliminatorKernel.SetArg("rowCount", blockRowCount);
            eliminatorKernel.EnqueueNdRanged(_commandQueue, [blockCount, rowCount, colCount]);
            
            (coeffSource, coeffDestination) = (coeffDestination, coeffSource);
            
        }
        
        coeffSource.Print(_commandQueue, coeffLength, sizeof(double), totalColCount, d => d.ToString("+0.000000000000000;-0.000000000000000"));
        coeffDestination.Print(_commandQueue, coeffLength, sizeof(double), totalColCount, d => d.ToString("+0.000000000000000;-0.000000000000000"));
        
        unknownSolverKernel.SetArg("valueBuffer", coeffSource);
        unknownSolverKernel.EnqueueNdRanged(_commandQueue, [blockCount]);
        
        var buf = coeffSource.Map<double>(_commandQueue, coeffLength * sizeof(double));
        for (var i = 0; i < coeffCount; i++)
        {
            var solution = buf[i * 2 * totalColCount];
            var coefficient = (char)('a' + i);
            Console.WriteLine($"{coefficient} = {solution:+0.000000000000000;-0.000000000000000}");
        }

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
                var coeff = coeffs[row * colCount + col + 1]; // +1 since const terms are at index 0
                
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
