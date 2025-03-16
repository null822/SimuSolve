namespace Mandelbrot;

public static class Constants
{
    public const int PlatformIndex = 1;
    public const int DeviceIndex = 0;

    /*private const double Cx = -0.7059999995;
    private const double Cy = -0.296;

    private const double Dx = 0.00000000000001;
    private const double Dy = 0.00000000000001;*/
    
    // -0.706
    // -0.296
    
    private const double Cx = -0.706;
    private const double Cy = -0.296;

    private const double Dx = 0.01;
    private const double Dy = 0.01;
    
    // -, -
    public const double X0 = Cx - Dx;
    public const double Y0 = Cy - Dy;
    
    // +, +
    public const double X1 = Cx + Dx;
    public const double Y1 = Cy + Dy;
    
    public const uint Width = 2048;
    public const uint Height = 2048;
    
    public const uint MaxIterations = 1024;
}