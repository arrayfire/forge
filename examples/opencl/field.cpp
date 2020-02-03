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

const unsigned DIMX      = 640;
const unsigned DIMY      = 480;
const float MINIMUM      = 1.0f;
const float MAXIMUM      = 20.f;
const float STEP         = 2.0f;
const float NELEMS       = (MAXIMUM - MINIMUM + 1) / STEP;
const unsigned DPOINTS[] = {5, 5, 5, 15, 15, 5, 15, 15};

#define USE_FORGE_OPENCL_COPY_HELPERS
#include <fg/compute_copy.h>

// clang-format off
static const std::string fieldKernel =
R"EOK(
constant float PI = 3.14159265359;

kernel void pointGenKernel(global float* points, global float* dirs, int NELEMS,
                           float MINIMUM, float STEP) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < NELEMS && j < NELEMS) {
        int id = i + j * NELEMS;

        float x = MINIMUM + i * STEP;
        float y = MINIMUM + j * STEP;

        points[2 * id + 0] = x;
        points[2 * id + 1] = y;

        dirs[2 * id + 0] = sin(2.0 * PI * x / 10.0);
        dirs[2 * id + 1] = sin(2.0 * PI * y / 10.0);
    }
}
)EOK";
// clang-format on

inline int divup(int a, int b)
{
    return (a + b - 1) / b;
}

void generatePoints(cl::Buffer& points, cl::Buffer& dirs,
                    cl::CommandQueue& queue, cl::Device& device) {
    static bool compileFlag = true;

    static cl::Program prog;
    static cl::Kernel pointGenKernel;

    if (compileFlag) {
        try {
            prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), fieldKernel,
                               false);

            std::vector<cl::Device> devs;
            devs.push_back(device);
            prog.build(devs);

            pointGenKernel = cl::Kernel(prog, "pointGenKernel");
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
    NDRange global(local[0] * divup((int)(NELEMS), (int)(local[0])),
                   local[1] * divup((int)(NELEMS), (int)(local[1])));

    pointGenKernel.setArg(0, points);
    pointGenKernel.setArg(1, dirs);
    pointGenKernel.setArg(2, (int)NELEMS);
    pointGenKernel.setArg(3, MINIMUM);
    pointGenKernel.setArg(4, STEP);
    queue.enqueueNDRangeKernel(pointGenKernel, cl::NullRange, global, local);
}

int main(void) {
    try {
        /*
         * First Forge call should be a window creation call
         * so that necessary OpenGL context is created for any
         * other forge::* object to be created successfully
         */
        forge::Window wnd(DIMX, DIMY, "Vector Field Demo");
        wnd.makeCurrent();

        forge::Chart chart(FG_CHART_2D);
        chart.setAxesLimits(MINIMUM - 1.0f, MAXIMUM, MINIMUM - 1.0f, MAXIMUM);
        chart.setAxesTitles("x-axis", "y-axis");

        forge::Plot divPoints =
            chart.plot(4, forge::u32, FG_PLOT_SCATTER, FG_MARKER_CIRCLE);
        divPoints.setColor(0.9f, 0.9f, 0.0f, 1.f);
        divPoints.setLegend("Convergence Points");
        divPoints.setMarkerSize(24);

        size_t npoints = (size_t)(NELEMS * NELEMS);

        forge::VectorField field =
            chart.vectorField((unsigned)(npoints), forge::f32);
        field.setColor(0.f, 0.6f, 0.3f, 1.f);

        /*
         * Helper function to create a CLGL interop context.
         * This function checks for if the extension is available
         * and creates the context on the appropriate device.
         * Note: context and queue are defined in cl_helpers.h
         */
        context       = createCLGLContext(wnd);
        Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        queue         = CommandQueue(context, device);

        GfxHandle* handles[3];

        createGLBuffer(&handles[0], divPoints.vertices(), FORGE_VERTEX_BUFFER);
        createGLBuffer(&handles[1], field.vertices(), FORGE_VERTEX_BUFFER);
        createGLBuffer(&handles[2], field.directions(), FORGE_VERTEX_BUFFER);

        cl::Buffer dpoints(context, CL_MEM_READ_WRITE, sizeof(unsigned) * 8);
        cl::Buffer points(context, CL_MEM_READ_WRITE,
                          sizeof(float) * 2 * npoints);
        cl::Buffer dirs(context, CL_MEM_READ_WRITE,
                        sizeof(float) * 2 * npoints);

        queue.enqueueWriteBuffer(dpoints, CL_TRUE, 0, sizeof(unsigned) * 8,
                                 DPOINTS);
        generatePoints(points, dirs, queue, device);

        copyToGLBuffer(handles[0], (ComputeResourceHandle)dpoints(),
                       divPoints.verticesSize());

        copyToGLBuffer(handles[1], (ComputeResourceHandle)points(),
                       field.verticesSize());
        copyToGLBuffer(handles[2], (ComputeResourceHandle)dirs(),
                       field.directionsSize());

        do { wnd.draw(chart); } while (!wnd.close());

        // destroy GL-CUDA interop buffers
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
