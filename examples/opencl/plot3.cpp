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
#include <complex>
#include <iostream>
#include <iterator>
#include <mutex>
#include <vector>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float ZMIN = 0.1f;
static const float ZMAX = 10.f;

const float DX              = 0.005f;
static const unsigned ZSIZE = (unsigned)((ZMAX - ZMIN) / DX + 1);

using namespace std;

#define USE_FORGE_OPENCL_COPY_HELPERS
#include <fg/compute_copy.h>

// clang-format off
static const std::string sincos_surf_kernel =
R"EOK(
kernel
void generateCurve(global float* out, const float t,
                   const float dx, const float zmin,
                   const unsigned SIZE) {
    int offset = get_global_id(0);
    float z = zmin + offset * dx;
    if (offset < SIZE) {
       out[offset*3 + 0] = cos(z*t+t)/z;
       out[offset*3 + 1] = sin(z*t+t)/z;
       out[offset*3 + 2] = z + 0.1*sin(t);
    }
}
)EOK";
// clang-format on

inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

void kernel(cl::Buffer& devOut, cl::CommandQueue& queue, float t) {
    static std::once_flag compileFlag;
    static cl::Program prog;
    static cl::Kernel kern;

    std::call_once(compileFlag, [queue]() {
        prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(),
                           sincos_surf_kernel, true);
        kern = cl::Kernel(prog, "generateCurve");
    });

    NDRange global(ZSIZE);

    kern.setArg(0, devOut);
    kern.setArg(1, t);
    kern.setArg(2, DX);
    kern.setArg(3, ZMIN);
    kern.setArg(4, ZSIZE);
    queue.enqueueNDRangeKernel(kern, cl::NullRange, global);
}

int main(void) {
    try {
        /*
         * First Forge call should be a window creation call
         * so that necessary OpenGL context is created for any
         * other forge::* object to be created successfully
         */
        forge::Window wnd(DIMX, DIMY, "Three dimensional line plot demo");
        wnd.makeCurrent();

        forge::Chart chart(FG_CHART_3D);

        chart.setAxesLabelFormat("%3.1f", "%3.1f", "%.2e");

        chart.setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, 0.f, 10.f);

        chart.setAxesTitles("x-axis", "y-axis", "z-axis");

        forge::Plot plot3 = chart.plot(ZSIZE, forge::f32);

        /*
         * Helper function to create a CLGL interop context.
         * This function checks for if the extension is available
         * and creates the context on the appropriate device.
         * Note: context and queue are defined in cl_helpers.h
         */
        context       = createCLGLContext(wnd);
        Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        queue         = CommandQueue(context, device);

        cl::Buffer devOut(context, CL_MEM_READ_WRITE,
                          sizeof(float) * ZSIZE * 3);
        static float t = 0;
        kernel(devOut, queue, t);

        GfxHandle* handle;
        createGLBuffer(&handle, plot3.vertices(), FORGE_VERTEX_BUFFER);
        /* copy your data into the pixel buffer object exposed by
         * forge::Surface class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        copyToGLBuffer(handle, (ComputeResourceHandle)devOut(),
                       plot3.verticesSize());

        do {
            t += 0.01f;
            kernel(devOut, queue, t);
            copyToGLBuffer(handle, (ComputeResourceHandle)devOut(),
                           plot3.verticesSize());
            wnd.draw(chart);
        } while (!wnd.close());

        releaseGLBuffer(handle);

    } catch (forge::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }
    return 0;
}
