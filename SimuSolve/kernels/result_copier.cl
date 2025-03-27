kernel void Main(
    global double* coeffBuffer,
    global double* outputBuffer,
    uint center, // half of the maxCoeffCount
    uint coeffCountDiff, // the difference between maxCoeffCount and coeffCount
    uint totalColCount) // total amount of cols in buffer
{
    size_t blockLength = totalColCount * 2;

    size_t index = get_global_id(0);

    // undo the output-flipping of the solving algorithm
    size_t x = index >= center ? index + coeffCountDiff : index; // skip the unused outputs
    size_t unsortedIndex = x ^ (x >> 1);

    outputBuffer[index] = coeffBuffer[unsortedIndex * 2 * totalColCount];
}
