/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <forge.h>
#define USE_FORGE_CUDA_COPY_HELPERS
#include <fg/compute_copy.h>
#include <cstdio>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float DX           = 0.1f;
static const float FRANGE_START = 0.f;
static const float FRANGE_END   = 2 * 3.141592f;
static const size_t DATA_SIZE   = (size_t)((FRANGE_END - FRANGE_START) / DX);

curandState_t* state;

void kernel(float* dev_out, int functionCode, float* colors, float* alphas,
            float* radii);

inline int divup(int a, int b) { return (a + b - 1) / b; }

__global__ void setupRandomKernel(curandState* states,
                                  unsigned long long seed) {
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

int main(void) {
    FORGE_CUDA_CHECK(
        cudaMalloc((void**)&state, DATA_SIZE * sizeof(curandState_t)));
    // clang-format off
    setupRandomKernel<<<divup(DATA_SIZE, 32), 32>>>(state, 314567);
    // clang-format on

    float* cos_out;
    float* tan_out;
    float* colors_out;
    float* alphas_out;
    float* radii_out;

    FORGE_CUDA_CHECK(
        cudaMalloc((void**)&cos_out, sizeof(float) * DATA_SIZE * 2));
    FORGE_CUDA_CHECK(
        cudaMalloc((void**)&tan_out, sizeof(float) * DATA_SIZE * 2));
    FORGE_CUDA_CHECK(
        cudaMalloc((void**)&colors_out, sizeof(float) * DATA_SIZE * 3));
    FORGE_CUDA_CHECK(
        cudaMalloc((void**)&alphas_out, sizeof(float) * DATA_SIZE));
    FORGE_CUDA_CHECK(cudaMalloc((void**)&radii_out, sizeof(float) * DATA_SIZE));

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other forge::* object to be created successfully
     */
    forge::Window wnd(DIMX, DIMY, "Bubble chart with Transparency Demo");
    wnd.makeCurrent();

    forge::Chart chart(FG_CHART_2D);
    chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.0f, 1.0f);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    forge::Plot plt1 =
        chart.plot(DATA_SIZE, forge::f32, FG_PLOT_LINE, FG_MARKER_TRIANGLE);
    forge::Plot plt2 =
        chart.plot(DATA_SIZE, forge::f32, FG_PLOT_LINE, FG_MARKER_CIRCLE);

    /* Set plot colors */
    plt1.setColor(FG_RED);
    plt2.setColor(FG_GREEN);  // use a forge predefined color
    /* Set plot legends */
    plt1.setLegend("Cosine");
    plt2.setLegend("Tangent");
    /* set plot global marker size */
    plt1.setMarkerSize(20);
    /* copy your data into the opengl buffer object exposed by
     * forge::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */

    GfxHandle* handles[5];

    // create GL-CUDA interop buffers
    createGLBuffer(&handles[0], plt1.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[1], plt2.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[2], plt2.colors(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[3], plt2.alphas(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[4], plt2.radii(), FORGE_VERTEX_BUFFER);

    kernel(cos_out, 0, NULL, NULL, NULL);
    kernel(tan_out, 1, colors_out, alphas_out, radii_out);

    // copy the data from compute buffer to graphics buffer
    copyToGLBuffer(handles[0], (ComputeResourceHandle)cos_out,
                   plt1.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)tan_out,
                   plt2.verticesSize());

    /* update color value for tan graph */
    copyToGLBuffer(handles[2], (ComputeResourceHandle)colors_out,
                   plt2.colorsSize());
    /* update alpha values for tan graph */
    copyToGLBuffer(handles[3], (ComputeResourceHandle)alphas_out,
                   plt2.alphasSize());
    /* update marker sizes for tan graph markers */
    copyToGLBuffer(handles[4], (ComputeResourceHandle)radii_out,
                   plt2.radiiSize());

    do { wnd.draw(chart); } while (!wnd.close());

    // destroy GL-CUDA Interop buffer
    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);
    releaseGLBuffer(handles[3]);
    releaseGLBuffer(handles[4]);
    // destroy CUDA handles
    FORGE_CUDA_CHECK(cudaFree(cos_out));
    FORGE_CUDA_CHECK(cudaFree(tan_out));
    FORGE_CUDA_CHECK(cudaFree(colors_out));
    FORGE_CUDA_CHECK(cudaFree(alphas_out));
    FORGE_CUDA_CHECK(cudaFree(radii_out));

    return 0;
}

__global__ void mapKernel(float* out, int functionCode, float frange_start,
                          float dx) {
    int id  = blockIdx.x * blockDim.x + threadIdx.x;
    float x = frange_start + id * dx;
    float y;

    switch (functionCode) {
        case 0: y = cos(x); break;
        case 1: y = tan(x); break;
        default: y = sin(x); break;
    }

    out[2 * id + 0] = x;
    out[2 * id + 1] = y;
}

__global__ void colorsKernel(float* colors, curandState* states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    colors[3 * id + 0] = curand_uniform(&states[id]);
    colors[3 * id + 1] = curand_uniform(&states[id]);
    colors[3 * id + 2] = curand_uniform(&states[id]);
}

__global__ void randKernel(float* out, curandState* states, float min,
                           float scale) {
    int id  = blockIdx.x * blockDim.x + threadIdx.x;
    out[id] = curand_uniform(&states[id]) * scale + min;
}

void kernel(float* dev_out, int functionCode, float* colors, float* alphas,
            float* radii) {
    static const dim3 threads(32);
    dim3 blocks(divup(DATA_SIZE, 32));

    // clang-format off
    mapKernel<<<blocks, threads>>>(dev_out, functionCode, FRANGE_START, DX);

    if (colors) colorsKernel<<<blocks, threads>>>(colors, state);

    if (alphas) randKernel<<<blocks, threads>>>(alphas, state, 0, 1);

    if (radii) randKernel<<<blocks, threads>>>(radii, state, 20, 60);
    // clang-format on
}
