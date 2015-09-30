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

static const float XMIN = -1.0f;
static const float XMAX = 2.f;
static const float YMIN = -1.0f;
static const float YMAX = 1.f;

const float DX = 0.01;
const unsigned XSIZE = (XMAX-XMIN)/DX+1;
const unsigned YSIZE = (YMAX-YMIN)/DX+1;


using namespace std;

static const std::string sincos_surf_kernel =
"kernel void sincos_surf(global float* out, const float dx, const float t, const float xmin, const float ymin, const unsigned w, const unsigned h)\n"
"{\n"
"    int offset = get_global_id(0);\n"
"    unsigned i = offset%w;\n"
"    unsigned j = offset/w;\n"
"\n"
"      float x = xmin + i*dx;\n"
"      float y = ymin + j*dx;\n"
"      out[offset*3 + 0] = x;\n"
"      out[offset*3 + 1] = y;\n"
"      out[offset*3 + 2] = 10*x*-fabs(y) * cos(x*x*(y+t))+sin(y*(x+t))-1.5;\n"
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
            kern = cl::Kernel(prog, "sincos_surf");
        });

    NDRange global(XSIZE*YSIZE);

    kern.setArg(0, devOut);
    kern.setArg(1, DX);
    kern.setArg(2, t);
    kern.setArg(3, XMIN);
    kern.setArg(4, YMIN);
    kern.setArg(5, XSIZE);
    kern.setArg(6, YSIZE);
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
        fg::Window wnd(DIMX, DIMY, "3d Surface Demo");
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

        /* Create several plot objects which creates the necessary
         * vertex buffer objects to hold the different plot types
         */
        fg::Surface surf(XSIZE, YSIZE, fg::f32, fg::FG_SURFACE);

        /*
         * Set plot colors
         */
        surf.setColor(fg::FG_YELLOW);

        /*
         * Set draw limits for plots
         */
        surf.setAxesLimits(1.1f, -1.1f, 1.1f, -1.1f, 10.f, -5.f);

        /*
        * Set axis titles
        */
        surf.setZAxisTitle("z-axis");
        surf.setYAxisTitle("y-axis");
        surf.setXAxisTitle("x-axis");


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
                break;
            }
        }

        Context context(device, cps);
        CommandQueue queue(context, device);

        cl::Buffer devOut(context, CL_MEM_READ_WRITE, sizeof(float) * XSIZE * YSIZE * 3);
        static float t=0;
        kernel(devOut, queue, t);
        /* copy your data into the pixel buffer object exposed by
         * fg::Surface class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        /*
        float buffcopy[XSIZE*YSIZE*3];
        queue.enqueueReadBuffer(
        devOut,
        true,
        0,
        XSIZE * YSIZE * 3 * sizeof(float),
        buffcopy);
        for(int i=0;i<XSIZE*YSIZE*3; i+=3)
            cout<<"("<<buffcopy[i]<<","<<buffcopy[i+1]<<","<<buffcopy[i+2]<<')'<<endl;
        */
        fg::copy(surf, devOut, queue);

        do {
            t+=0.07;
            kernel(devOut, queue, t);
            //fg::copy(surf, devOut, queue);
            // draw window and poll for events last
            wnd.draw(surf);
        } while(!wnd.close());
    }catch (fg::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }
    return 0;
}

