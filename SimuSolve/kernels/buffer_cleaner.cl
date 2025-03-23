kernel void Main(
    global char* buffer)
{
    size_t index = get_global_id(0);

    buffer[index] = 0;
}
