#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void Main(
    global double* coeffBuffer,
    global double* outputBuffer,
    uint totalColCount) // total amount of cols in buffer
{
    size_t blockLength = totalColCount * 2;

    size_t index = get_global_id(0);

    outputBuffer[index] = coeffBuffer[index * 2 * totalColCount];
}
