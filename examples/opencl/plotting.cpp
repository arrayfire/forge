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
#include <vector>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include "cl_helpers.h"

using namespace cl;
using namespace std;

const unsigned DIMX = 1000;
const unsigned DIMY = 800;
const unsigned WIN_ROWS = 2;
const unsigned WIN_COLS = 2;

const float    dx = 0.1;
const float    FRANGE_START = 0.f;
const float    FRANGE_END = 2 * 3.141592f;
const unsigned SIZE = ( FRANGE_END - FRANGE_START ) / dx;

static const std::string sinf_ocl_kernel =
"kernel void sinf(global float* out, const float dx, const unsigned SIZE)\n"
"{\n"
"    unsigned x = get_global_id(0);\n"
"    if(x < SIZE){\n"
"        out[2 * x] = x * dx ;\n"
"        out[2 * x + 1] = sin(x*dx);\n"
"    }\n"
"}\n";

void kernel(cl::Buffer& devOut, cl::CommandQueue& queue)
{
    static std::once_flag   compileFlag;
    static cl::Program      prog;
    static cl::Kernel       kern;

    std::call_once(compileFlag,
        [queue]() {
        prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), sinf_ocl_kernel, true);
            kern = cl::Kernel(prog, "sinf");
        });

    static const NDRange global(SIZE * 2);

    kern.setArg(0, devOut);
    kern.setArg(1, dx);
    kern.setArg(2, SIZE);
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
        fg::Window wnd(DIMX, DIMY, "Fractal Demo");
        wnd.makeCurrent();
        /* create an font object and load necessary font
        * and later pass it on to window object so that
        * it can be used for rendering text */
        fg::Font fnt;
#ifdef OS_WIN
        fnt.loadSystemFont("Calibri", 32);
#else
        fnt.loadSystemFont("Vera", 32);
#endif
        wnd.setFont(&fnt);

        /*
         * Split the window into grid regions
         */
        wnd.grid(WIN_ROWS, WIN_COLS);

        /* Create several plot objects which creates the necessary
         * vertex buffer objects to hold the different plot types
         */
        fg::Plot plt0(SIZE, fg::f32);                                 //create a default plot
        fg::Plot plt1(SIZE, fg::f32, fg::FG_LINE, fg::FG_NONE);       //or specify a specific plot type
        fg::Plot plt2(SIZE, fg::f32, fg::FG_LINE, fg::FG_TRIANGLE);   //last parameter specifies marker shape
        fg::Plot plt3(SIZE, fg::f32, fg::FG_SCATTER, fg::FG_POINT);

        /*
         * Set plot colors
         */
        plt0.setColor(fg::FG_YELLOW);
        plt1.setColor(fg::FG_BLUE);
        plt2.setColor(fg::FG_WHITE);                                        //use a forge predefined color
        plt3.setColor((fg::Color) 0xABFF01FF);                              //or any hex-valued color

        /*
         * Set draw limits for plots
         */
        plt0.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);
        plt1.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);
        plt2.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);
        plt3.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);

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

        cl::Buffer devOut(context, CL_MEM_READ_WRITE, sizeof(float) * SIZE * 2);
        kernel(devOut, queue);

        /* copy your data into the vertex buffer object exposed by
         * fg::Plot class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        fg::copy(plt0, devOut, queue);
        fg::copy(plt1, devOut, queue);
        fg::copy(plt2, devOut, queue);
        fg::copy(plt3, devOut, queue);

        do {
            wnd.draw(0, 0, plt0,  NULL                );
            wnd.draw(0, 1, plt1, "sinf_line_blue"     );
            wnd.draw(1, 1, plt2, "sinf_line_triangle" );
            wnd.draw(1, 0, plt3, "sinf_scatter_point" );
            // draw window and poll for events last
            wnd.swapBuffers();
        } while(!wnd.close());
    }catch (fg::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }
    return 0;
}
