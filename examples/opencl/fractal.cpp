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

const unsigned DIMX = 512;
const unsigned DIMY = 512;
const unsigned IMG_SIZE = DIMX * DIMY * 4;

#define USE_FORGE_OPENCL_COPY_HELPERS
#include <ComputeCopy.h>

static const std::string fractal_ocl_kernel =
"float magnitude(float2 a)\n"
"{\n"
"    return sqrt(a.s0*a.s0+a.s1*a.s1);\n"
"}\n"
"\n"
"float2 mul(float2 a, float2 b)\n"
"{\n"
"    return (float2)(a.s0*b.s0-a.s1*b.s1, a.s1*b.s0+a.s0*b.s1);\n"
"}\n"
"\n"
"float2 add(float2 a, float2 b)\n"
"{\n"
"    return (float2)(a.s0+b.s0, a.s1+b.s1);\n"
"}\n"
"\n"
"int pixel(int x, int y, int width, int height)\n"
"{\n"
"\n"
"    const float scale = 1.5;\n"
"    float jx = scale * (float)(width/2.0f - x)/(width/2.0f);\n"
"    float jy = scale * (float)(height/2.0f - y)/(height/2.0f);\n"
"\n"
"    float2 c = (float2)(-0.8f, 0.156f);\n"
"    float2 a = (float2)(jx, jy);\n"
"\n"
"    for (int i=0; i<200; i++) {\n"
"        a = add(mul(a, a), c);\n"
"        if (magnitude(a) > 1000.0f)\n"
"            return 0;\n"
"    }\n"
"\n"
"    return 1;\n"
"}\n"
"\n"
"kernel\n"
"void julia(global unsigned char* out, const unsigned w, const unsigned h)\n"
"{\n"
"    int x = get_group_id(0) * get_local_size(0) + get_local_id(0);\n"
"    int y = get_group_id(1) * get_local_size(1) + get_local_id(1);\n"
"\n"
"    if (x<w && y<h) {\n"
"        int offset        = x + y * w;\n"
"        int juliaValue    = pixel(x, y, w, h);\n"
"        out[offset*4 + 1] = 255 * juliaValue;\n"
"        out[offset*4 + 0] = 0;\n"
"        out[offset*4 + 2] = 0;\n"
"        out[offset*4 + 3] = 255;\n"
"    }\n"
"}\n";

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(cl::Buffer& devOut, cl::CommandQueue& queue)
{
    static std::once_flag   compileFlag;
    static cl::Program      prog;
    static cl::Kernel       kern;

    std::call_once(compileFlag,
        [queue]() {
        prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), fractal_ocl_kernel, true);
            kern = cl::Kernel(prog, "julia");
        });

    auto juliaOp = cl::KernelFunctor<Buffer, unsigned, unsigned>(kern);

    static const NDRange local(8, 8);
    NDRange global(local[0] * divup(DIMX, local[0]),
                   local[1] * divup(DIMY, local[1]));

    juliaOp(EnqueueArgs(queue, global, local), devOut, DIMX, DIMY);
}

int main(void)
{
    try {

        /*
        * First Forge call should be a window creation call
        * so that necessary OpenGL context is created for any
        * other forge::* object to be created successfully
        */
        forge::Window wnd(DIMX, DIMY, "Fractal Demo");
        wnd.makeCurrent();

        /* Create an image object which creates the necessary
        * textures and pixel buffer objects to hold the image
        * */
        forge::Image img(DIMX, DIMY, FG_RGBA, forge::u8);

        /*
         * Helper function to create a CLGL interop context.
         * This function checks for if the extension is available
         * and creates the context on the appropriate device.
         * Note: context and queue are defined in cl_helpers.h
         */
        context = createCLGLContext(wnd);
        Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        queue = CommandQueue(context, device);

        /* copy your data into the pixel buffer object exposed by
         * forge::Image class and then proceed to rendering.
         * To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        cl::Buffer devOut(context, CL_MEM_READ_WRITE, IMG_SIZE);

        kernel(devOut, queue);

        GfxHandle* handle = 0;

        // create GL-CPU interop buffer
        createGLBuffer(&handle, img.pbo(), FORGE_PBO);

        // copy the data from compute buffer to graphics buffer
        copyToGLBuffer(handle, (ComputeResourceHandle)devOut(), img.size());

        do {
            wnd.draw(img);
        } while(!wnd.close());

        // destroy GL-CPU Interop buffer
        releaseGLBuffer(handle);

    }catch (forge::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}
