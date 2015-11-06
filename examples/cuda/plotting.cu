#include <forge.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <CUDACopy.hpp>
#include <cstdio>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;
const unsigned WIN_ROWS = 2;
const unsigned WIN_COLS = 2;

const float    dx = 0.1;
const float    FRANGE_START = 0.f;
const float    FRANGE_END = 2 * 3.141592f;
const size_t   DATA_SIZE = ( FRANGE_END - FRANGE_START ) / dx;

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

    /*
     * Split the window into grid regions
     */
    wnd.grid(WIN_ROWS, WIN_COLS);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    fg::Plot plt0( DATA_SIZE, fg::f32);                              //create a default plot
    fg::Plot plt1( DATA_SIZE, fg::f32, fg::FG_LINE, fg::FG_NONE);       //or specify a specific plot type
    fg::Plot plt2( DATA_SIZE, fg::f32, fg::FG_LINE, fg::FG_TRIANGLE);   //last parameter specifies marker shape
    fg::Plot plt3( DATA_SIZE, fg::f32, fg::FG_SCATTER, fg::FG_POINT);

    /*
     * Set plot colors
     */
    plt0.setColor(fg::FG_YELLOW);
    plt1.setColor(fg::FG_BLUE);
    plt2.setColor(fg::FG_WHITE);                                                  //use a forge predefined color
    plt3.setColor((fg::Color) 0xABFF01FF);                                        //or any hex-valued color

    /*
     * Set draw limits for plots
     */
    plt0.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);
    plt1.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);
    plt2.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);
    plt3.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);

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
        wnd.draw(0, 0, plt0,  NULL                );
        wnd.draw(0, 1, plt1, "sinf_line_blue"     );
        wnd.draw(1, 1, plt2, "sinf_line_triangle" );
        wnd.draw(1, 0, plt3, "sinf_scatter_point" );
        // draw window and poll for events last
        wnd.swapBuffers();
    } while(!wnd.close());

    CUDA_ERROR_CHECK(cudaFree(dev_out));
    return 0;
}


__global__
void simple_sinf(float* out)
{
    int x = blockIdx.x * blockDim.x  + threadIdx.x;

    if (x<DATA_SIZE) {
        out[ 2 * x ] = x * dx;
        out[ 2 * x + 1 ] = sin(x*dx);
    }
}

void kernel(float* dev_out)
{
    static const dim3 threads(DATA_SIZE);
    dim3 blocks(1);

    simple_sinf<<< blocks, threads >>>(dev_out);
}
