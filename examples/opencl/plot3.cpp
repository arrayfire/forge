/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <forge.h>
#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>
#include <OpenCLCopy.hpp>
#include <mutex>
#include <complex>
#include <cmath>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>
#include "cl_helpers.h"

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float ZMIN = 0.1f;
static const float ZMAX = 10.f;

const float DX = 0.005;
static const unsigned ZSIZE = (ZMAX-ZMIN)/DX+1;


using namespace std;

static const std::string sincos_surf_kernel =
"kernel void generateCurve(global float* out, const float t, const float dx, const float zmin, const unsigned SIZE)\n"
"{\n"
"    int offset = get_global_id(0);\n"
"\n"
"    float z = zmin + offset*dx;\n"
"    if (offset < SIZE) {\n"
"       out[offset*3 + 0] = cos(z*t+t)/z;\n"
"       out[offset*3 + 1] = sin(z*t+t)/z;\n"
"       out[offset*3 + 2] = z + 0.1*sin(t);\n"
"    }\n"
"}\n";

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(cl::Buffer& devOut, cl::CommandQueue& queue, float t)
{
    static std::once_flag   compileFlag;
    static cl::Program      prog;
    static cl::Kernel       kern;

    std::call_once(compileFlag,
        [queue]() {
        prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), sincos_surf_kernel, true);
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

int main(void)
{
    try {
        /*
         * First Forge call should be a window creation call
         * so that necessary OpenGL context is created for any
         * other fg::* object to be created successfully
         */
        fg::Window wnd(DIMX, DIMY, "Three dimensional line plot demo");
        wnd.makeCurrent();

        fg::Chart chart(FG_CHART_3D);
        chart.setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, 0.f, 10.f);
        chart.setAxesTitles("x-axis", "y-axis", "z-axis");

        fg::Plot plot3 = chart.plot(ZSIZE, fg::f32);

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
        CommandQueue queue;
        Context context;
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

        cl::Buffer devOut(context, CL_MEM_READ_WRITE, sizeof(float) * ZSIZE * 3);
        static float t=0;
        kernel(devOut, queue, t);
        /* copy your data into the pixel buffer object exposed by
         * fg::Surface class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        fg::copy(plot3.vertices(), plot3.verticesSize(), devOut, queue);

        do {
            t+=0.01;
            kernel(devOut, queue, t);
            fg::copy(plot3.vertices(), plot3.verticesSize(), devOut, queue);
            wnd.draw(chart);
        } while(!wnd.close());
    }catch (fg::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }
    return 0;
}

