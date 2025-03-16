#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void Main(
    global double* scaleBuffer,
    global double* srcBuffer,
    uint rowCount, // ammount of rows in a block
    uint totalColCount) // total amount of cols in buffer
{
    size_t blockLength = rowCount * totalColCount;

    size_t blockIndex = get_global_id(0);
    size_t rowIndex = get_global_id(1);
    size_t colIndex = get_global_id(2);

    size_t valueIndex = (blockIndex * blockLength) + (rowIndex * totalColCount) + colIndex;
    size_t scaleIndex = (blockIndex * rowCount) + rowIndex; // a block in the scale buffer is 1 wide
    
    srcBuffer[valueIndex] *= scaleBuffer[scaleIndex];
    // srcBuffer[valueIndex] = rowCount;
}
