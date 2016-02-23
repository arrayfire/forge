#include <forge.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <CUDACopy.hpp>
#include <cstdio>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

const float XMIN = -1.0f;
const float XMAX = 2.f;
const float YMIN = -1.0f;
const float YMAX = 1.f;

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
    fnt.loadSystemFont("Calibri");
#else
    fnt.loadSystemFont("Vera");
#endif
    wnd.setFont(&fnt);

    fg::Chart chart(fg::FG_3D);
    chart.setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, -5.f, 10.f);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    fg::Surface surf = chart.surface(XSIZE, YSIZE, fg::f32);
    surf.setColor(fg::FG_YELLOW);

    static float t=0;
    FORGE_CUDA_CHECK(cudaMalloc((void**)&dev_out, XSIZE * YSIZE * 3 * sizeof(float) ));
    kernel(t, DX, dev_out);
    /* copy your data into the vertex buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    fg::copy(surf.vertices(), dev_out);

    do {
        t+=0.07;
        kernel(t, DX, dev_out);
        fg::copy(surf.vertices(), dev_out);
        wnd.draw(chart);
    } while(!wnd.close());

    FORGE_CUDA_CHECK(cudaFree(dev_out));
    return 0;
}


__global__
void sincos_surf(float t, float dx, float* out,
				 const float XMIN, const float YMIN,
				 const size_t XSIZE, const size_t YSIZE)
{
    int i = blockIdx.x * blockDim.x  + threadIdx.x;
    int j = blockIdx.y * blockDim.y  + threadIdx.y;

    float x= ::XMIN + i*dx;
    float y= ::YMIN + j*dx;
    if (i<XSIZE && j<YSIZE) {
        int offset = j + i * YSIZE;
        out[ 3 * offset     ] = x;
        out[ 3 * offset + 1 ] = y;
        out[ 3 * offset + 2 ] = 10*x*-abs(y) * cos(x*x*(y+t))+sin(y*(x+t))-1.5;
    }
}

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(float t, float dx, float* dev_out)
{
    static const dim3 threads(8, 8);
    dim3 blocks(divup(XSIZE, threads.x),
                divup(YSIZE, threads.y));

    sincos_surf<<< blocks, threads >>>(t, dx, dev_out, XMIN, YMIN, XSIZE, YSIZE);
}
