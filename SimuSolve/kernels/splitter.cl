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


    // indexes of the source and destination blocks
    size_t blockSrcIndex = blockIndex * 2 * blockLength;
    size_t blockDesIndex = blockSrcIndex + blockLength; // destination is 1 block further

    // the source and destination row index (relative to the start of the block)
    size_t blockRowIndex = rowIndex * totalColCount;

    // indexes of the source and destination terms (relative to the start of the row)
    size_t elementSrcIndex = colIndex;
    size_t elementDesIndex = (colIndex == 0 ? 0 : (colCount - colIndex)); // flip cols (but not the const terms)

    // full source and destination indexes
    size_t srcIndex = blockSrcIndex + blockRowIndex +  + elementSrcIndex;
    size_t desIndex = blockDesIndex + blockRowIndex +  + elementDesIndex;

    buffer[desIndex] = buffer[srcIndex];
}

