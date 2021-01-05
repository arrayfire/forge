/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cuda_runtime.h>
#include <forge.h>
#define USE_FORGE_CUDA_COPY_HELPERS
#include <fg/compute_copy.h>

const unsigned DIMX = 640;
const unsigned DIMY = 480;
const float MINIMUM = 1.0f;
const float MAXIMUM = 20.f;
const float STEP    = 2.0f;
const int NELEMS    = (int)((MAXIMUM - MINIMUM + 1) / STEP);

void generateColors(float* colors);

void generatePoints(float* points, float* dirs);

inline int divup(int a, int b) { return (a + b - 1) / b; }

int main(void) {
    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other forge::* object to be created successfully
     */
    forge::Window wnd(DIMX, DIMY, "3D Vector Field Demo");
    wnd.makeCurrent();

    forge::Chart chart(FG_CHART_3D);
    chart.setAxesLimits(MINIMUM - 1.0f, MAXIMUM, MINIMUM - 1.0f, MAXIMUM,
                        MINIMUM - 1.0f, MAXIMUM);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    int numElems             = NELEMS * NELEMS * NELEMS;
    forge::VectorField field = chart.vectorField(numElems, forge::f32);
    field.setColor(0.f, 1.f, 0.f, 1.f);

    float* points;
    float* colors;
    float* dirs;

    FORGE_CUDA_CHECK(cudaMalloc((void**)&points, 3 * numElems * sizeof(float)));
    FORGE_CUDA_CHECK(cudaMalloc((void**)&colors, 3 * numElems * sizeof(float)));
    FORGE_CUDA_CHECK(cudaMalloc((void**)&dirs, 3 * numElems * sizeof(float)));

    generatePoints(points, dirs);
    generateColors(colors);

    GfxHandle* handles[3];
    createGLBuffer(&handles[0], field.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[1], field.colors(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[2], field.directions(), FORGE_VERTEX_BUFFER);

    copyToGLBuffer(handles[0], (ComputeResourceHandle)points,
                   field.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)colors,
                   field.colorsSize());
    copyToGLBuffer(handles[2], (ComputeResourceHandle)dirs,
                   field.directionsSize());

    do { wnd.draw(chart); } while (!wnd.close());

    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);

    FORGE_CUDA_CHECK(cudaFree(points));
    FORGE_CUDA_CHECK(cudaFree(colors));
    FORGE_CUDA_CHECK(cudaFree(dirs));

    return 0;
}

__global__ void genColorsKernel(float* colors, int nelems) {
    const float AF_BLUE[4]   = {0.0588f, 0.1137f, 0.2745f, 1.0f};
    const float AF_ORANGE[4] = {0.8588f, 0.6137f, 0.0745f, 1.0f};

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < nelems) {
        if (i % 2 == 0) {
            colors[3 * i + 0] = AF_ORANGE[0];
            colors[3 * i + 1] = AF_ORANGE[1];
            colors[3 * i + 2] = AF_ORANGE[2];
        } else {
            colors[3 * i + 0] = AF_BLUE[0];
            colors[3 * i + 1] = AF_BLUE[1];
            colors[3 * i + 2] = AF_BLUE[2];
        }
    }
}

void generateColors(float* colors) {
    const int numElems = NELEMS * NELEMS * NELEMS;
    static const dim3 threads(512);
    dim3 blocks(divup(numElems, threads.x));

    // clang-format off
    genColorsKernel<<<blocks, threads>>>(colors, numElems);
    // clang-format on
}

__global__ void pointGenKernel(float* points, float* dirs, int nBBS0,
                               int nelems, float minimum, float step) {
    int k = blockIdx.x / nBBS0;
    int i = blockDim.x * (blockIdx.x - k * nBBS0) + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < nelems && j < nelems && k < nelems) {
        float x = minimum + i * step;
        float y = minimum + j * step;
        float z = minimum + k * step;

        int id = i + j * nelems + k * nelems * nelems;

        points[3 * id + 0] = x;
        points[3 * id + 1] = y;
        points[3 * id + 2] = z;

        dirs[3 * id + 0] = x - 10.f;
        dirs[3 * id + 1] = y - 10.f;
        dirs[3 * id + 2] = z - 10.f;
    }
}

void generatePoints(float* points, float* dirs) {
    static dim3 threads(8, 8);

    int blk_x = divup(NELEMS, threads.x);
    int blk_y = divup(NELEMS, threads.y);

    dim3 blocks(blk_x * NELEMS, blk_y);

    // clang-format off
    pointGenKernel<<<blocks, threads>>>(points, dirs, blk_x, NELEMS, MINIMUM,
                                        STEP);
    // clang-format on
}
