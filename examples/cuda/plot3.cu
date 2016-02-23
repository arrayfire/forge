/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <forge.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <CUDACopy.hpp>
#include <cstdio>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float ZMIN = 0.1f;
static const float ZMAX = 10.f;

const float DX = 0.005;
const size_t ZSIZE = (ZMAX-ZMIN)/DX+1;

void kernel(float t, float dx, float* dev_out);

int main(void)
{
    float *dev_out;

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "Plot 3d Demo");
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
    chart.setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, 0.f, 10.f);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    fg::Plot plot3 = chart.plot(ZSIZE, fg::f32);

    static float t=0;
    FORGE_CUDA_CHECK(cudaMalloc((void**)&dev_out, ZSIZE * 3 * sizeof(float) ));
    kernel(t, DX, dev_out);
    /* copy your data into the vertex buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    fg::copy(plot3.vertices(), dev_out);


    do {
        t+=0.01;
        kernel(t, DX, dev_out);
        fg::copy(plot3.vertices(), dev_out);
        wnd.draw(chart);
    } while(!wnd.close());

    FORGE_CUDA_CHECK(cudaFree(dev_out));
    return 0;
}


__global__
void gen_curve(float t, float dx, float* out, const float ZMIN, const size_t ZSIZE)
{
    int offset = blockIdx.x * blockDim.x  + threadIdx.x;

    float z = ZMIN + offset*dx;
    if(offset < ZSIZE){
        out[ 3 * offset     ] = cos(z*t+t)/z;
        out[ 3 * offset + 1 ] = sin(z*t+t)/z;
        out[ 3 * offset + 2 ] = z + 0.1*sin(t);
    }
}

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(float t, float dx, float* dev_out)
{
    static const dim3 threads(1024);
    dim3 blocks(divup(ZSIZE, 1024));

    gen_curve<<< blocks, threads >>>(t, dx, dev_out, ZMIN, ZSIZE);
}
