﻿#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void Main(
    global double* srcBuffer,
    global double* desBuffer,
    uint rowCount,
    uint colCount)
{
    size_t blockIndex = get_global_id(0);
    size_t rowIndex = get_global_id(1);
    size_t colIndex = get_global_id(2);
    
    size_t blockLength = rowCount * colCount;

    size_t coeff1Index = (blockIndex * blockLength) + (rowIndex * colCount) + (colIndex);
    size_t coeff2Index = coeff1Index + colCount; // coeff2 is 1 row further
    
    double coeff1 = srcBuffer[coeff1Index];
    double coeff2 = srcBuffer[coeff2Index];
    
    desBuffer[coeff1Index] = coeff1 - coeff2;
}
