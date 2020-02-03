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

static const float XMIN = -8.0f;
static const float XMAX = 8.f;
static const float YMIN = -8.0f;
static const float YMAX = 8.f;

const float DX       = 0.5;
const unsigned XSIZE = (unsigned)((XMAX - XMIN) / DX);
const unsigned YSIZE = (unsigned)((YMAX - YMIN) / DX);

using namespace std;

#define USE_FORGE_OPENCL_COPY_HELPERS
#include <fg/compute_copy.h>

// clang-format off
static const std::string sin_surf_kernel =
R"EOK(
kernel void
surf(global float* out, const float dx, const float xmin, const float ymin,
     const unsigned w, const unsigned h) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float x = xmin + i * dx;
    float y = ymin + j * dx;

    if (i < w && j < h) {
        int offset          = j + i * h;
        out[3 * offset]     = x;
        out[3 * offset + 1] = y;
        float z             = sqrt(x * x + y * y) + 2.2204e-16;
        out[3 * offset + 2] = sin(z) / z;
    }
}
)EOK";
// clang-format on

inline
int divup(int a, int b)
{
    return (a + b - 1) / b;
}

void kernel(cl::Buffer& devOut, cl::CommandQueue& queue, cl::Device& device) {
    static bool compileFlag = true;
    static cl::Program prog;
    static cl::Kernel kern;

    if (compileFlag) {
        try {
            prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(),
                               sin_surf_kernel, false);

            std::vector<cl::Device> devs;
            devs.push_back(device);
            prog.build(devs);

            kern = cl::Kernel(prog, "surf");
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

    NDRange local(8, 8);
    NDRange global(local[0] * divup(XSIZE, (int)(local[0])),
                   local[1] * divup(YSIZE, (int)(local[1])));

    kern.setArg(0, devOut);
    kern.setArg(1, DX);
    kern.setArg(2, XMIN);
    kern.setArg(3, YMIN);
    kern.setArg(4, XSIZE);
    kern.setArg(5, YSIZE);
    queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local);
}

int main(void) {
    try {
        /*
         * First Forge call should be a window creation call
         * so that necessary OpenGL context is created for any
         * other forge::* object to be created successfully
         */
        forge::Window wnd(1024, 768, "3d Surface Demo");
        wnd.makeCurrent();

        forge::Chart chart(FG_CHART_3D);
        chart.setAxesLimits(-10.f, 10.f, -10.f, 10.f, -0.5f, 1.f);
        chart.setAxesTitles("x-axis", "y-axis", "z-axis");

        forge::Surface surf = chart.surface(XSIZE, YSIZE, forge::f32);
        surf.setColor(FG_YELLOW);

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
                          sizeof(float) * XSIZE * YSIZE * 3);

        kernel(devOut, queue, device);

        GfxHandle* handle;
        createGLBuffer(&handle, surf.vertices(), FORGE_VERTEX_BUFFER);
        /* copy your data into the pixel buffer object exposed by
         * forge::Surface class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        copyToGLBuffer(handle, (ComputeResourceHandle)devOut(),
                       surf.verticesSize());

        do { wnd.draw(chart); } while (!wnd.close());

        releaseGLBuffer(handle);
    } catch (forge::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}
