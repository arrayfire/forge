/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <forge.h>
#define USE_FORGE_CUDA_COPY_HELPERS
#include <fg/compute_copy.h>
#include <cstdio>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float dx           = 0.1f;
static const float FRANGE_START = 0.f;
static const float FRANGE_END   = 2 * 3.141592f;
static const size_t DATA_SIZE   = (size_t)((FRANGE_END - FRANGE_START) / dx);

void kernel(float* dev_out, int functionCode);

int main(void) {
    float* sin_out;
    float* cos_out;
    float* tan_out;
    float* log_out;

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other forge::* object to be created successfully
     */
    forge::Window wnd(DIMX, DIMY, "Plotting Demo");
    wnd.makeCurrent();

    forge::Chart chart(FG_CHART_2D);
    chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.0f, 1.0f);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    forge::Plot plt0 =
        chart.plot(DATA_SIZE, forge::f32);  // create a default plot
    forge::Plot plt1 =
        chart.plot(DATA_SIZE, forge::f32, FG_PLOT_LINE,
                   FG_MARKER_NONE);  // or specify a specific plot type
    forge::Plot plt2 = chart.plot(
        DATA_SIZE, forge::f32, FG_PLOT_LINE,
        FG_MARKER_TRIANGLE);  // last parameter specifies marker shape
    forge::Plot plt3 =
        chart.plot(DATA_SIZE, forge::f32, FG_PLOT_SCATTER, FG_MARKER_CROSS);

    /*
     * Set plot colors
     */
    plt0.setColor(FG_RED);
    plt1.setColor(FG_BLUE);
    plt2.setColor(FG_YELLOW);                 // use a forge predefined color
    plt3.setColor((forge::Color)0x257973FF);  // or any hex-valued color
    /*
     * Set plot legends
     */
    plt0.setLegend("Sine");
    plt1.setLegend("Cosine");
    plt2.setLegend("Tangent");
    plt3.setLegend("Log base 10");

    FORGE_CUDA_CHECK(
        cudaMalloc((void**)&sin_out, sizeof(float) * DATA_SIZE * 2));
    FORGE_CUDA_CHECK(
        cudaMalloc((void**)&cos_out, sizeof(float) * DATA_SIZE * 2));
    FORGE_CUDA_CHECK(
        cudaMalloc((void**)&tan_out, sizeof(float) * DATA_SIZE * 2));
    FORGE_CUDA_CHECK(
        cudaMalloc((void**)&log_out, sizeof(float) * DATA_SIZE * 2));

    kernel(sin_out, 0);
    kernel(cos_out, 1);
    kernel(tan_out, 2);
    kernel(log_out, 3);

    GfxHandle* handles[4];
    createGLBuffer(&handles[0], plt0.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[1], plt1.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[2], plt2.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[3], plt3.vertices(), FORGE_VERTEX_BUFFER);

    /* copy your data into the vertex buffer object exposed by
     * forge::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copyToGLBuffer(handles[0], (ComputeResourceHandle)sin_out,
                   plt0.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)cos_out,
                   plt1.verticesSize());
    copyToGLBuffer(handles[2], (ComputeResourceHandle)tan_out,
                   plt2.verticesSize());
    copyToGLBuffer(handles[3], (ComputeResourceHandle)log_out,
                   plt3.verticesSize());

    do { wnd.draw(chart); } while (!wnd.close());

    FORGE_CUDA_CHECK(cudaFree(sin_out));
    FORGE_CUDA_CHECK(cudaFree(cos_out));
    FORGE_CUDA_CHECK(cudaFree(tan_out));
    FORGE_CUDA_CHECK(cudaFree(log_out));
    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);
    releaseGLBuffer(handles[3]);

    return 0;
}

__global__ void simple_sinf(float* out, const size_t _data_size, int fnCode,
                            const float _dx, const float _frange_start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _data_size) {
        float x  = _frange_start + i * _dx;
        int idx  = 2 * i;
        out[idx] = x;

        switch (fnCode) {
            case 0: out[idx + 1] = sinf(x); break;
            case 1: out[idx + 1] = cosf(x); break;
            case 2: out[idx + 1] = tanf(x); break;
            case 3: out[idx + 1] = log10f(x); break;
        }
    }
}

inline int divup(int a, int b) { return (a + b - 1) / b; }

void kernel(float* dev_out, int functionCode) {
    static const dim3 threads(1024);
    dim3 blocks(divup(DATA_SIZE, 1024));

    // clang-format off
    simple_sinf<<<blocks, threads>>>(dev_out, DATA_SIZE, functionCode, dx,
                                     FRANGE_START);
    // clang-format on
}
