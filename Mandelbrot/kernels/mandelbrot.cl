#pragma OPENCL EXTENSION cl_khr_fp64 : enable

static inline double mandel(double c_re, double c_im, int iterations) {
    double z_re = c_re;
    double z_im = c_im;
    
    int i;
    for (i = 0; i < iterations; ++i) {
        // if (z_re * z_re + z_im * z_im > 4.0)
        //     break;

        double new_re = z_re*z_re - z_im*z_im;
        double new_im = 2.0 * z_re * z_im;

        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return z_re * z_re + z_im * z_im;
}
kernel void Mandelbrot(
    double x0, double y0,
    double x1, double y1,
    int width, int height,
    int maxIterations,
    global double* output)
{
    double dx = (x1 - x0) / width;
    double dy = (y1 - y0) / height;

    double x = x0 + get_global_id(0) * dx;
    double y = y0 + get_global_id(1) * dy;

    int index = get_global_id(1) * width + get_global_id(0);
    output[index] = mandel(x, y, maxIterations);
}
