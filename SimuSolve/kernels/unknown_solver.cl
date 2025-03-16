#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void Main(
    global double* valueBuffer,
    uint totalColCount) // total amount of cols in buffer
{
    size_t blockLength = totalColCount * 2;

    size_t blockIndex = get_global_id(0);

    size_t absRowIndex = blockIndex * blockLength;
    size_t valueIndex = absRowIndex + 1;

    // result = const term / coefficient
    valueBuffer[absRowIndex] /= valueBuffer[valueIndex];
}
