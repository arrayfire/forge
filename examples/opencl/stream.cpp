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
#include <iterator>
#include <sstream>
#include <vector>

using namespace cl;
using namespace std;

const unsigned DIMX = 640;
const unsigned DIMY = 480;
const float MINIMUM = 1.0f;
const float MAXIMUM = 20.f;
const float STEP    = 2.0f;
const int NELEMS    = (int)((MAXIMUM - MINIMUM + 1) / STEP);

#define USE_FORGE_OPENCL_COPY_HELPERS
#include <fg/compute_copy.h>

// clang-format off
static const std::string streamKernel =
R"EOK(
constant float AF_BLUE[4]         = {0.0588f, 0.1137f, 0.2745f, 1.0f};
constant float AF_ORANGE[4]           = {0.8588f, 0.6137f, 0.0745f, 1.0f};

kernel void genColorsKernel(global float* colors, int NELEMS) {
    const size_t nelems = NELEMS * NELEMS * NELEMS;

    int i = get_global_id(0);

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

kernel void pointGenKernel(global float* points, global float* dirs, int nBBS0,
                           int NELEMS, float MINIMUM, float STEP) {
    int k = get_group_id(0) / nBBS0;
    int i = get_local_size(0) * (get_group_id(0) - k * nBBS0) + get_local_id(0);
    int j = get_global_id(1);

    if (i < NELEMS && j < NELEMS && k < NELEMS) {
        float x = MINIMUM + i * STEP;
        float y = MINIMUM + j * STEP;
        float z = MINIMUM + k * STEP;

        int id = i + j * NELEMS + k * NELEMS * NELEMS;

        points[3 * id + 0] = x;
        points[3 * id + 1] = y;
        points[3 * id + 2] = z;

        dirs[3 * id + 0] = x - 10.f;
        dirs[3 * id + 1] = y - 10.f;
        dirs[3 * id + 2] = z - 10.f;
    }
}
)EOK";
// clang-format on

inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}

void generatePoints(Buffer& points, Buffer& dirs, Buffer& colors,
                    CommandQueue& queue, Device& device) {
    static bool compileFlag = true;

    static cl::Program prog;
    static cl::Kernel pointGenKernel;
    static cl::Kernel colorsKernel;

    if (compileFlag) {
        try {
            prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), streamKernel,
                               false);

            std::vector<cl::Device> devs;
            devs.push_back(device);
            prog.build(devs);

            pointGenKernel = cl::Kernel(prog, "pointGenKernel");
            colorsKernel   = cl::Kernel(prog, "genColorsKernel");
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

    static const NDRange local(8, 8);
    int blk_x = divup(NELEMS, (int)(local[0]));
    int blk_y = divup(NELEMS, (int)(local[1]));

    NDRange global(NELEMS * local[0] * blk_x, local[1] * blk_y);

    pointGenKernel.setArg(0, points);
    pointGenKernel.setArg(1, dirs);
    pointGenKernel.setArg(2, blk_x);
    pointGenKernel.setArg(3, NELEMS);
    pointGenKernel.setArg(4, MINIMUM);
    pointGenKernel.setArg(5, STEP);
    queue.enqueueNDRangeKernel(pointGenKernel, cl::NullRange, global, local);
    const int numElems = NELEMS * NELEMS * NELEMS;
    static const NDRange thrds(64, 1);
    NDRange glob(thrds[0] * divup(numElems, (int)(thrds[0])), (int)(thrds[1]));

    colorsKernel.setArg(0, colors);
    colorsKernel.setArg(1, NELEMS);
    queue.enqueueNDRangeKernel(colorsKernel, cl::NullRange, glob, thrds);
}

int main(void) {
    try {
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

        /*
         * Helper function to create a CLGL interop context.
         * This function checks for if the extension is available
         * and creates the context on the appropriate device.
         * Note: context and queue are defined in cl_helpers.h
         */
        context       = createCLGLContext(wnd);
        Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        queue         = CommandQueue(context, device);

        cl::Buffer points(context, CL_MEM_READ_WRITE,
                          sizeof(float) * 3 * numElems);
        cl::Buffer colors(context, CL_MEM_READ_WRITE,
                          sizeof(float) * 3 * numElems);
        cl::Buffer dirs(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 3 * numElems);

        GfxHandle* handles[3];
        createGLBuffer(&handles[0], field.vertices(), FORGE_VERTEX_BUFFER);
        createGLBuffer(&handles[1], field.colors(), FORGE_VERTEX_BUFFER);
        createGLBuffer(&handles[2], field.directions(), FORGE_VERTEX_BUFFER);

        generatePoints(points, dirs, colors, queue, device);

        copyToGLBuffer(handles[0], (ComputeResourceHandle)points(),
                       field.verticesSize());
        copyToGLBuffer(handles[1], (ComputeResourceHandle)colors(),
                       field.colorsSize());
        copyToGLBuffer(handles[2], (ComputeResourceHandle)dirs(),
                       field.directionsSize());

        do { wnd.draw(chart); } while (!wnd.close());

        releaseGLBuffer(handles[0]);
        releaseGLBuffer(handles[1]);
        releaseGLBuffer(handles[2]);

    } catch (forge::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}
