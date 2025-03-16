﻿#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void Main(
    global double* scaleBuffer,
    global double* srcBuffer,
    uint rowCount, // ammount of rows in a block
    uint totalColCount, // total amount of cols in buffer
    uint targetValue) // index of the target coefficient to calculate scales for
{
    size_t blockLength = rowCount * totalColCount;

    size_t blockIndex = get_global_id(0);
    size_t rowIndex = get_global_id(1);

    size_t valueIndex = (blockIndex * blockLength) + (rowIndex * totalColCount) + targetValue;
    size_t scaleIndex = (blockIndex * rowCount) + rowIndex; // a block in the scale buffer is 1 wide
    
    double scale = 1.0 / srcBuffer[valueIndex];
    scaleBuffer[scaleIndex] = scale;
}
