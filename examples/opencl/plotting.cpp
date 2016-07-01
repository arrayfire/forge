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
#include <vector>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>

using namespace cl;
using namespace std;

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

const float    dx = 0.1;
const float    FRANGE_START = 0.f;
const float    FRANGE_END = 2 * 3.141592f;
const unsigned DATA_SIZE = ( FRANGE_END - FRANGE_START ) / dx;

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

static const std::string sinf_ocl_kernel = R"(
kernel void sinf(global float* out, const float dx, const unsigned DATA_SIZE, int fnCode)
{
    unsigned x = get_global_id(0);
    if(x < DATA_SIZE) {
        out[2 * x] = x * dx ;
        switch(fnCode) {
            case 0:
                out[ 2 * x + 1 ] = sin(x*dx);
                break;
            case 1:
                out[ 2 * x + 1 ] = cos(x*dx);
                break;
            case 2:
                out[ 2 * x + 1 ] = tan(x*dx);
                break;
            case 3:
                out[ 2 * x + 1 ] = log10(x*dx);
                break;
        }
    }
}
)";

void kernel(cl::Buffer& devOut, cl::CommandQueue& queue, int fnCode)
{
    static std::once_flag   compileFlag;
    static cl::Program      prog;
    static cl::Kernel       kern;

    std::call_once(compileFlag,
        [queue]() {
        prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), sinf_ocl_kernel, true);
            kern = cl::Kernel(prog, "sinf");
        });

    static const NDRange global(DATA_SIZE * 2);

    kern.setArg(0, devOut);
    kern.setArg(1, dx);
    kern.setArg(2, DATA_SIZE);
    kern.setArg(3, fnCode);
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
        fg::Window wnd(DIMX, DIMY, "Plotting Demo");
        wnd.makeCurrent();

        fg::Chart chart(FG_CHART_2D);
        chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.0f, 1.0f);

        /* Create several plot objects which creates the necessary
         * vertex buffer objects to hold the different plot types
         */
        fg::Plot plt0 = chart.plot(DATA_SIZE, fg::f32);                                 //create a default plot
        fg::Plot plt1 = chart.plot(DATA_SIZE, fg::f32, FG_PLOT_LINE, FG_MARKER_NONE);       //or specify a specific plot type
        fg::Plot plt2 = chart.plot(DATA_SIZE, fg::f32, FG_PLOT_LINE, FG_MARKER_TRIANGLE);   //last parameter specifies marker shape
        fg::Plot plt3 = chart.plot(DATA_SIZE, fg::f32, FG_PLOT_SCATTER, FG_MARKER_CROSS);

        /*
         * Set plot colors
         */
        plt0.setColor(FG_RED);
        plt1.setColor(FG_BLUE);
        plt2.setColor(FG_YELLOW);            //use a forge predefined color
        plt3.setColor((fg::Color) 0x257973FF);  //or any hex-valued color
        /*
         * Set plot legends
         */
        plt0.setLegend("Sine");
        plt1.setLegend("Cosine");
        plt2.setLegend("Tangent");
        plt3.setLegend("Log base 10");

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

        cl::Buffer sinOut(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE * 2);
        cl::Buffer cosOut(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE * 2);
        cl::Buffer tanOut(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE * 2);
        cl::Buffer logOut(context, CL_MEM_READ_WRITE, sizeof(float) * DATA_SIZE * 2);
        kernel(sinOut, queue, 0);
        kernel(cosOut, queue, 1);
        kernel(tanOut, queue, 2);
        kernel(logOut, queue, 3);

        GfxHandle* handles[4];
        createGLBuffer(&handles[0], plt0.vertices(), FORGE_VBO);
        createGLBuffer(&handles[1], plt1.vertices(), FORGE_VBO);
        createGLBuffer(&handles[2], plt2.vertices(), FORGE_VBO);
        createGLBuffer(&handles[3], plt3.vertices(), FORGE_VBO);
        /* copy your data into the vertex buffer object exposed by
         * fg::Plot class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        copyToGLBuffer(handles[0], (ComputeResourceHandle)sinOut(), plt0.verticesSize());
        copyToGLBuffer(handles[1], (ComputeResourceHandle)cosOut(), plt1.verticesSize());
        copyToGLBuffer(handles[2], (ComputeResourceHandle)tanOut(), plt2.verticesSize());
        copyToGLBuffer(handles[3], (ComputeResourceHandle)logOut(), plt3.verticesSize());

        do {
            wnd.draw(chart);
        } while(!wnd.close());

        releaseGLBuffer(handles[0]);
        releaseGLBuffer(handles[1]);
        releaseGLBuffer(handles[2]);
        releaseGLBuffer(handles[3]);

    }catch (fg::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }
    return 0;
}
