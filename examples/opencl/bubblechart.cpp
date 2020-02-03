/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <forge.h>

#include "cl_helpers.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <sstream>
#include <vector>

using namespace cl;
using namespace std;

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float DX           = 0.1f;
static const float FRANGE_START = 0.f;
static const float FRANGE_END   = 2 * 3.141592f;
static const int DATA_SIZE      = (int)((FRANGE_END - FRANGE_START) / DX);

#define USE_FORGE_OPENCL_COPY_HELPERS
#include <fg/compute_copy.h>

// clang-format off
static const std::string chartKernels =
R"EOK(
float rand(int x) {
    x = (x << 13) ^ x;
    return (1.0 - ((x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) /
                      1073741824.0);
}

kernel void randKernel(global float* out, unsigned seed, float min, float scale,
                       int DATA_SIZE) {
    int id = get_global_id(0);
    if (id < DATA_SIZE) out[id] = scale * (1 + rand(seed * id)) / 2.0f + min;
}

kernel void colorsKernel(global float* out, unsigned rseed, unsigned gseed,
                         unsigned bseed, int DATA_SIZE) {
    int id = get_global_id(0);
    if (id < DATA_SIZE) {
        out[3 * id + 0] = (1 + rand(rseed * id)) / 2.0f;
        out[3 * id + 1] = (1 + rand(gseed * id)) / 2.0f;
        out[3 * id + 2] = (1 + rand(bseed * id)) / 2.0f;
    }
}

kernel void mapKernel(global float* out, int functionCode, float FRANGE_START,
                      float DX, int DATA_SIZE) {
    int id  = get_global_id(0);
    float x = FRANGE_START + id * DX;
    float y;

    switch (functionCode) {
        case 0: y = cos(x); break;
        case 1: y = tan(x); break;
        default: y = sin(x); break;
    }

    if (id < DATA_SIZE) {
        out[2 * id + 0] = x;
        out[2 * id + 1] = y;
    }
}
)EOK";
// clang-format on

inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}

void kernel(cl::Buffer& devOut, int fnCode, int outFlags, cl::Buffer& colorsOut,
            cl::Buffer& alphasOut, cl::Buffer& radiiOut,
            cl::CommandQueue& queue, cl::Device& device) {
    static bool compileFlag = true;

    static cl::Program prog;
    static cl::Kernel randKernel, colorsKernel, mapKernel;

    std::srand((unsigned)(std::time(0)));

    if (compileFlag) {
        try {
            prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), chartKernels,
                               false);

            std::vector<cl::Device> devs;
            devs.push_back(device);
            prog.build(devs);

            randKernel   = cl::Kernel(prog, "randKernel");
            colorsKernel = cl::Kernel(prog, "colorsKernel");
            mapKernel    = cl::Kernel(prog, "mapKernel");
        } catch (cl::Error err) {
            std::cout << "Compile Errors: " << std::endl;
            std::cout << err.what() << err.err() << std::endl;
            std::cout << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                      << std::endl;
            exit(255);
        }
        std::cout << "Kernels compiled successfully" << std::endl;
        compileFlag = false;
    }

    static const NDRange local(32);
    NDRange global(local[0] * divup(DATA_SIZE, (int)(local[0])));

    mapKernel.setArg(0, devOut);
    mapKernel.setArg(1, fnCode);
    mapKernel.setArg(2, FRANGE_START);
    mapKernel.setArg(3, DX);
    mapKernel.setArg(4, DATA_SIZE);
    queue.enqueueNDRangeKernel(mapKernel, cl::NullRange, global, local);

    if (outFlags & 0x00000001) {
        colorsKernel.setArg(0, colorsOut);
        colorsKernel.setArg(1, std::rand());
        colorsKernel.setArg(2, std::rand());
        colorsKernel.setArg(3, std::rand());
        colorsKernel.setArg(4, DATA_SIZE);
        queue.enqueueNDRangeKernel(colorsKernel, cl::NullRange, global, local);
    }

    if (outFlags & 0x00000002) {
        randKernel.setArg(0, alphasOut);
        randKernel.setArg(1, std::rand());
        randKernel.setArg(2, 0.0f);
        randKernel.setArg(3, 1.0f);
        randKernel.setArg(4, DATA_SIZE);
        queue.enqueueNDRangeKernel(randKernel, cl::NullRange, global, local);
    }

    if (outFlags & 0x00000004) {
        randKernel.setArg(0, radiiOut);
        randKernel.setArg(1, std::rand());
        randKernel.setArg(2, 20.0f);
        randKernel.setArg(3, 60.0f);
        randKernel.setArg(4, DATA_SIZE);
        queue.enqueueNDRangeKernel(randKernel, cl::NullRange, global, local);
    }
}

int main(void) {
    try {
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

        /*
         * Helper function to create a CLGL interop context.
         * This function checks for if the extension is available
         * and creates the context on the appropriate device.
         * Note: context and queue are defined in cl_helpers.h
         */
        context       = createCLGLContext(wnd);
        Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        queue         = CommandQueue(context, device);

        GfxHandle* handles[5];

        // create GL-OpenCL interop buffers
        createGLBuffer(&handles[0], plt1.vertices(), FORGE_VERTEX_BUFFER);
        createGLBuffer(&handles[1], plt2.vertices(), FORGE_VERTEX_BUFFER);
        createGLBuffer(&handles[2], plt2.colors(), FORGE_VERTEX_BUFFER);
        createGLBuffer(&handles[3], plt2.alphas(), FORGE_VERTEX_BUFFER);
        createGLBuffer(&handles[4], plt2.radii(), FORGE_VERTEX_BUFFER);

        cl::Buffer cosOut(context, CL_MEM_READ_WRITE,
                          sizeof(float) * DATA_SIZE * 2);
        cl::Buffer tanOut(context, CL_MEM_READ_WRITE,
                          sizeof(float) * DATA_SIZE * 2);
        cl::Buffer colorsOut(context, CL_MEM_READ_WRITE,
                             sizeof(float) * DATA_SIZE * 3);
        cl::Buffer alphasOut(context, CL_MEM_READ_WRITE,
                             sizeof(float) * DATA_SIZE);
        cl::Buffer radiiOut(context, CL_MEM_READ_WRITE,
                            sizeof(float) * DATA_SIZE);
        cl::Buffer dummy;

        kernel(cosOut, 0, 0, dummy, dummy, dummy, queue, device);
        kernel(tanOut, 1, 0x00000007, colorsOut, alphasOut, radiiOut, queue,
               device);

        /* copy your data into the opengl buffer object exposed by
         * forge::Plot class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        copyToGLBuffer(handles[0], (ComputeResourceHandle)cosOut(),
                       plt1.verticesSize());
        copyToGLBuffer(handles[1], (ComputeResourceHandle)tanOut(),
                       plt2.verticesSize());

        /* update color value for tan graph */
        copyToGLBuffer(handles[2], (ComputeResourceHandle)colorsOut(),
                       plt2.colorsSize());
        /* update alpha values for tan graph */
        copyToGLBuffer(handles[3], (ComputeResourceHandle)alphasOut(),
                       plt2.alphasSize());
        /* update marker sizes for tan graph markers */
        copyToGLBuffer(handles[4], (ComputeResourceHandle)radiiOut(),
                       plt2.radiiSize());

        do { wnd.draw(chart); } while (!wnd.close());

        // destroy GL-OpenCL Interop buffer
        releaseGLBuffer(handles[0]);
        releaseGLBuffer(handles[1]);
        releaseGLBuffer(handles[2]);
        releaseGLBuffer(handles[3]);
        releaseGLBuffer(handles[4]);

    } catch (forge::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}
