#include <forge.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <CUDACopy.hpp>
#include <cstdio>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float XMIN = -1.0f;
static const float XMAX = 2.f;
static const float YMIN = -1.0f;
static const float YMAX = 1.f;

const float DX = 0.01;
const size_t XSIZE = (XMAX-XMIN)/DX+1;
const size_t YSIZE = (YMAX-YMIN)/DX+1;

void kernel(float t, float dx, float* dev_out);

int main(void)
{
    float *dev_out;

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "3d Surface Demo");
    wnd.makeCurrent();
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

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    fg::Surface surf(XSIZE, YSIZE, fg::f32, fg::FG_SURFACE);

    /*
     * Set plot colors
     */
    surf.setColor(fg::FG_YELLOW);

    /*
     * Set draw limits for plots
     */
    surf.setAxesLimits(1.1f, -1.1f, 1.1f, -1.1f, 10.f, -5.f);

    /*
    * Set axis titles
    */
    surf.setZAxisTitle("z-axis");
    surf.setYAxisTitle("y-axis");
    surf.setXAxisTitle("x-axis");

    static float t=0;
    CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_out, sizeof(float) * XSIZE * YSIZE * 3));
    kernel(t, DX, dev_out);
    /* copy your data into the vertex buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    fg::copy(surf, dev_out);

    do {
        t+=0.07;
        kernel(t, DX, dev_out);
        //fg::copy(surf, dev_out);
        // draw window and poll for events last
        wnd.draw(surf);
    } while(!wnd.close());

    CUDA_ERROR_CHECK(cudaFree(dev_out));
    return 0;
}


__global__
void sincos_surf(float t, float dx, float* out)
{
    int i = blockIdx.x * blockDim.x  + threadIdx.x;
    int j = blockIdx.y * blockDim.y  + threadIdx.y;
    
    float x=XMIN+i*dx;
    float y=YMIN+j*dx;
    if (i<DIMX && j<DIMY) {
        out[ 3 * (j*DIMX+i)     ] = x;
        out[ 3 * (j*DIMX+i) + 1 ] = y;
        out[ 3 * (j*DIMX+i) + 2 ] = 10*x*-abs(y) * cos(x*x*(y+t))+sin(y*(x+t))-1.5;
    }
}

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(float t, float dx, float* dev_out)
{
    static const dim3 threads(8, 8);
    dim3 blocks(XSIZE, YSIZE);

    sincos_surf<<< blocks, threads >>>(t, dx, dev_out);
}
