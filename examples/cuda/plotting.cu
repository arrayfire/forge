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

void kernel(float* dev_out);

int main(void)
{
    float *dev_out;

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "Plotting Demo");
    wnd.makeCurrent();
    wnd.grid(1,2);
    /* create an font object and load necessary font
     * and later pass it on to window object so that
     * it can be used for rendering text */
    fg::Font fnt;
#ifdef OS_WIN
    fnt.loadSystemFont("Calibri", 32);
#else
    fnt.loadSystemFont("Vera", 32);
#endif
    wnd.setFont(&fnt);

    fg::Chart chart(fg::FG_2D);
    chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.1f, 1.1f);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    fg::Plot plt0 = chart.plot( DATA_SIZE, fg::f32);                                 //create a default plot
    fg::Plot plt1 = chart.plot( DATA_SIZE, fg::f32, fg::FG_LINE, fg::FG_NONE);       //or specify a specific plot type
    fg::Plot plt2 = chart.plot( DATA_SIZE, fg::f32, fg::FG_LINE, fg::FG_TRIANGLE);   //last parameter specifies marker shape
    fg::Plot plt3 = chart.plot( DATA_SIZE, fg::f32, fg::FG_SCATTER, fg::FG_POINT);

    /*
     * Set plot colors
     */
    plt0.setColor(fg::FG_YELLOW);
    plt1.setColor(fg::FG_BLUE);
    plt2.setColor(fg::FG_WHITE);                                                  //use a forge predefined color
    plt3.setColor((fg::Color) 0xABFF01FF);                                        //or any hex-valued color

    CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_out, sizeof(float) * DATA_SIZE * 2));
    kernel(dev_out);
    /* copy your data into the vertex buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    fg::copy(plt0, dev_out);
    fg::copy(plt1, dev_out);
    fg::copy(plt2, dev_out);
    fg::copy(plt3, dev_out);

    do {
        wnd.draw(chart);
        wnd.swapBuffers();
    } while(!wnd.close());

    CUDA_ERROR_CHECK(cudaFree(dev_out));
    return 0;
}


__global__
void simple_sinf(float* out, const size_t DATA_SIZE, const float dx)
{
    int x = blockIdx.x * blockDim.x  + threadIdx.x;

    if (x<DATA_SIZE) {
        out[ 2 * x ] = x * dx;
        out[ 2 * x + 1 ] = sin(x*dx);
    }
}

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(float* dev_out)
{
    static const dim3 threads(1024);
    dim3 blocks(divup(DATA_SIZE, 1024));

    simple_sinf << < blocks, threads >> >(dev_out, DATA_SIZE, dx);
}
