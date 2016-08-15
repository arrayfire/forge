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
#include <mutex>
#include <complex>
#include <cmath>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

static const float XMIN = -8.0f;
static const float XMAX = 8.f;
static const float YMIN = -8.0f;
static const float YMAX = 8.f;

const float DX = 0.5;
const unsigned XSIZE = (XMAX-XMIN)/DX;
const unsigned YSIZE = (YMAX-YMIN)/DX;

using namespace std;

cl::CommandQueue queue;
cl::Context context;

cl_context getContext()
{
    return context();
}

cl_command_queue getCommandQueue()
{
    return queue();
}

#define USE_FORGE_OPENCL_COPY_HELPERS
#include <ComputeCopy.h>

static const std::string sin_surf_kernel =
R"EOK(
kernel
void surf(global float* out, const float dx,
          const float xmin, const float ymin,
          const unsigned w, const unsigned h)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float x = xmin + i*dx;
    float y = ymin + j*dx;

    if (i<w && j<h) {
        int offset = j + i * h;
        out[ 3 * offset     ] = x;
        out[ 3 * offset + 1 ] = y;
        float z = sqrt(x*x+y*y) + 2.2204e-16;
        out[ 3 * offset + 2 ] = sin(z)/z;
    }
}
)EOK";

inline
int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(cl::Buffer& devOut, cl::CommandQueue& queue, cl::Device& device)
{
    static bool compileFlag = true;
    static cl::Program prog;
    static cl::Kernel  kern;

    if (compileFlag) {
        try {
            prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), sin_surf_kernel, false);

            std::vector<cl::Device> devs;
            devs.push_back(device);
            prog.build(devs);

            kern = cl::Kernel(prog, "surf");
        } catch (cl::Error err) {
            std::cout << "Compile Errors: " << std::endl;
            std::cout << err.what() << err.err() << std::endl;
            std::cout << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            exit(255);
        }
        std::cout<< "Kernels compiled successfully" << std::endl;
        compileFlag = false;
    }

    NDRange local(8, 8);
    NDRange global(local[0]*divup(XSIZE, local[0]),
                   local[1]*divup(YSIZE, local[1]));

    kern.setArg(0, devOut);
    kern.setArg(1, DX);
    kern.setArg(2, XMIN);
    kern.setArg(3, YMIN);
    kern.setArg(4, XSIZE);
    kern.setArg(5, YSIZE);
    queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local);
}

int main(void)
{
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

        Platform plat = getPlatform();
        // Select the default platform and create a context using this platform and the GPU
#if defined(OS_MAC)
        CGLContextObj cgl_current_ctx = CGLGetCurrentContext();
        CGLShareGroupObj cgl_share_group = CGLGetShareGroup(cgl_current_ctx);

        cl_context_properties cps[] = {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)cgl_share_group,
            0
        };
#elif defined(OS_LNX)
        cl_context_properties cps[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)wnd.context(),
            CL_GLX_DISPLAY_KHR, (cl_context_properties)wnd.display(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)plat(),
            0
        };
#else /* OS_WIN */
        cl_context_properties cps[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)wnd.context(),
            CL_WGL_HDC_KHR, (cl_context_properties)wnd.display(),
            CL_CONTEXT_PLATFORM, (cl_context_properties)plat(),
            0
        };
#endif
        std::vector<Device> devs;
        plat.getDevices(CL_DEVICE_TYPE_GPU, &devs);

        Device device;

        for (auto& d : devs) {
            if (checkExtnAvailability(d, CL_GL_SHARING_EXT)) {
                device = d;
                context = Context(device, cps);
                try {
                    queue = CommandQueue(context, device);
                    break;
                } catch (cl::Error err) {
                    continue;
                }
            }
        }

        cl::Buffer devOut(context, CL_MEM_READ_WRITE, sizeof(float) * XSIZE * YSIZE * 3);

        kernel(devOut, queue, device);

        GfxHandle* handle;
        createGLBuffer(&handle, surf.vertices(), FORGE_VBO);
        /* copy your data into the pixel buffer object exposed by
         * forge::Surface class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        copyToGLBuffer(handle, (ComputeResourceHandle)devOut(), surf.verticesSize());

        do {
            wnd.draw(chart);
        } while(!wnd.close());

        releaseGLBuffer(handle);
    }catch (forge::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}
