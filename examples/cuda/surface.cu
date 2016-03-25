#include <forge.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <CUDACopy.hpp>
#include <cstdio>
#include <iostream>

const float XMIN = -8.0f;
const float XMAX = 8.f;
const float YMIN = -8.0f;
const float YMAX = 8.f;

const float DX = 0.5;
const size_t XSIZE = (XMAX-XMIN)/DX;
const size_t YSIZE = (YMAX-YMIN)/DX;

void kernel(float dx, float* dev_out);

int main(void)
{
    float *dev_out;

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(1024, 768, "3d Surface Demo");
    wnd.makeCurrent();

    fg::Chart chart(FG_CHART_3D);
    chart.setAxesLimits(-10.f, 10.f, -10.f, 10.f, -0.5f, 1.f);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    fg::Surface surf = chart.surface(XSIZE, YSIZE, f32);
    surf.setColor(FG_YELLOW);

    FORGE_CUDA_CHECK(cudaMalloc((void**)&dev_out, XSIZE * YSIZE * 3 * sizeof(float) ));
    kernel(DX, dev_out);
    /* copy your data into the vertex buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    fg::copy(surf.vertices(), dev_out);

    do {
        wnd.draw(chart);
    } while(!wnd.close());

    FORGE_CUDA_CHECK(cudaFree(dev_out));
    return 0;
}


__global__
void sincos_surf(float dx, float* out,
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
        float z = sqrt(x*x+y*y) + 2.2204e-16;
        out[ 3 * offset + 2 ] = sinf(z)/z;
    }
}

inline
int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(float dx, float* dev_out)
{
    static const dim3 threads(8, 8);
    dim3 blocks(divup(XSIZE, threads.x),
                divup(YSIZE, threads.y));

    sincos_surf<<< blocks, threads >>>(dx, dev_out, XMIN, YMIN, XSIZE, YSIZE);
}
