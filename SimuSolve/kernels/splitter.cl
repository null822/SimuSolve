#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void Main(
    global double* buffer,
    uint rowCount, // ammount of rows in a block
    uint colCount, // ammount of cols in a block
    uint totalColCount) // total amount of cols in buffer
{
    size_t blockLength = rowCount * totalColCount;

    size_t blockIndex = get_global_id(0);
    size_t rowIndex = get_global_id(1);
    size_t colIndex = get_global_id(2);


    size_t blockSrcIndex = blockIndex * 2 * blockLength;
    size_t blockDesIndex = blockSrcIndex + blockLength; // destination is 1 block further
    size_t elementSrcIndex = (rowIndex * totalColCount) + colIndex;
    size_t elementDesIndex = (rowIndex * totalColCount) + (colIndex == 0 ? 0 : (colCount - colIndex)); // flip cols (but not the const terms)

    size_t srcIndex = blockSrcIndex + elementSrcIndex;
    size_t desIndex = blockDesIndex + elementDesIndex;

    buffer[desIndex] = buffer[srcIndex];
}
