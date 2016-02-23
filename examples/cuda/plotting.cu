#include <forge.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <CUDACopy.hpp>
#include <cstdio>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float    dx = 0.1;
static const float    FRANGE_START = 0.f;
static const float    FRANGE_END = 2 * 3.141592f;
static const size_t   DATA_SIZE = ( FRANGE_END - FRANGE_START ) / dx;

void kernel(float* dev_out, int functionCode);

int main(void)
{
    float *sin_out;
    float *cos_out;
    float *tan_out;
    float *log_out;

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "Plotting Demo");
    wnd.makeCurrent();
    wnd.grid(1,2);

    fg::Chart chart(fg::FG_2D);
    chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.1f, 1.1f);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    fg::Plot plt0 = chart.plot( DATA_SIZE, fg::f32);                                 //create a default plot
    fg::Plot plt1 = chart.plot( DATA_SIZE, fg::f32, fg::FG_LINE, fg::FG_NONE);       //or specify a specific plot type
    fg::Plot plt2 = chart.plot( DATA_SIZE, fg::f32, fg::FG_LINE, fg::FG_TRIANGLE);   //last parameter specifies marker shape
    fg::Plot plt3 = chart.plot( DATA_SIZE, fg::f32, fg::FG_SCATTER, fg::FG_CROSS);

    /*
     * Set plot colors
     */
    plt0.setColor(fg::FG_RED);
    plt1.setColor(fg::FG_BLUE);
    plt2.setColor(fg::FG_YELLOW);            //use a forge predefined color
    plt3.setColor((fg::Color) 0x257973FF);  //or any hex-valued color
    /*
     * Set plot legends
     */
    plt0.setLegend("Sine");
    plt1.setLegend("Cosine");
    plt2.setLegend("Tangent");
    plt3.setLegend("Log base 10");

    FORGE_CUDA_CHECK(cudaMalloc((void**)&sin_out, sizeof(float) * DATA_SIZE * 2));
    FORGE_CUDA_CHECK(cudaMalloc((void**)&cos_out, sizeof(float) * DATA_SIZE * 2));
    FORGE_CUDA_CHECK(cudaMalloc((void**)&tan_out, sizeof(float) * DATA_SIZE * 2));
    FORGE_CUDA_CHECK(cudaMalloc((void**)&log_out, sizeof(float) * DATA_SIZE * 2));

    kernel(sin_out, 0);
    kernel(cos_out, 1);
    kernel(tan_out, 2);
    kernel(log_out, 3);
    /* copy your data into the vertex buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    fg::copy(plt0.vertices(), sin_out);
    fg::copy(plt1.vertices(), cos_out);
    fg::copy(plt2.vertices(), tan_out);
    fg::copy(plt3.vertices(), log_out);

    do {
        wnd.draw(chart);
    } while(!wnd.close());

    FORGE_CUDA_CHECK(cudaFree(sin_out));
    FORGE_CUDA_CHECK(cudaFree(cos_out));
    FORGE_CUDA_CHECK(cudaFree(tan_out));
    FORGE_CUDA_CHECK(cudaFree(log_out));
    return 0;
}

__global__
void simple_sinf(float* out, const size_t DATA_SIZE, int fnCode)
{
    int x = blockIdx.x * blockDim.x  + threadIdx.x;

    if (x<DATA_SIZE) {
        out[ 2 * x ] = x * dx;
        switch(fnCode) {
            case 0:
                out[ 2 * x + 1 ] = sinf(x*dx);
                break;
            case 1:
                out[ 2 * x + 1 ] = cosf(x*dx);
                break;
            case 2:
                out[ 2 * x + 1 ] = tanf(x*dx);
                break;
            case 3:
                out[ 2 * x + 1 ] = log10f(x*dx);
                break;
        }
    }
}

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(float* dev_out, int functionCode)
{
    static const dim3 threads(1024);
    dim3 blocks(divup(DATA_SIZE, 1024));

    simple_sinf << < blocks, threads >> >(dev_out, DATA_SIZE, functionCode);
}
