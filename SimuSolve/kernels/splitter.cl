#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/*

16 cols
stage = 4
shift = 3

0000 => 0100
   0 =>    8
1111 => 1011
  15 =>   11
1001 => 1101
   9 =>   13
0011 => 0111
   3 =>    7

*/


kernel void Main(
    global double* buffer,
    uint rowCount, // ammount of rows in a block
    uint totalColCount) // total amount of cols in buffer
{
    size_t blockLength = rowCount * totalColCount;

    size_t blockIndex = get_global_id(0);
    size_t rowIndex = get_global_id(1);
    size_t colIndex = get_global_id(2);


    size_t blockSrcIndex = blockIndex * 2 * blockLength;
    size_t blockDesIndex = blockSrcIndex + blockLength; // destination is 1 block further
    size_t elementSrcIndex = (rowIndex * totalColCount) + colIndex;
    
    size_t center = rowCount / 2;
    size_t desColIndex;

    // flip left and right sides (flipping columns), but not the first (const term) column
    if (colIndex == 0) {
        desColIndex = 0;
    } else if (colIndex-1 < center) {
        desColIndex = colIndex + center;
    } else {
        desColIndex = colIndex - center;
    }
    
    size_t elementDesIndex = (rowIndex * totalColCount) + desColIndex;

    size_t srcIndex = blockSrcIndex + elementSrcIndex;
    size_t desIndex = blockDesIndex + elementDesIndex;
    
    buffer[desIndex] = buffer[srcIndex];
}
