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
#include <iostream>
#include <mutex>
#include <sstream>

using namespace cl;
using namespace std;

const unsigned DIMX     = 512;
const unsigned DIMY     = 512;
const unsigned IMG_SIZE = DIMX * DIMY * 4;

#define USE_FORGE_OPENCL_COPY_HELPERS
#include <fg/compute_copy.h>

// clang-format off
static const std::string fractal_ocl_kernel =
R"EOK(
float magnitude(float2 a) {
    return sqrt(a.s0*a.s0+a.s1*a.s1);
}
float2 mul(float2 a, float2 b) {
    return (float2)(a.s0*b.s0-a.s1*b.s1, a.s1*b.s0+a.s0*b.s1);
}
float2 add(float2 a, float2 b) {
    return (float2)(a.s0+b.s0, a.s1+b.s1);
}
int pixel(int x, int y, int width, int height) {
    const float scale = 1.5;
    float jx = scale * (float)(width/2.0f - x)/(width/2.0f);
    float jy = scale * (float)(height/2.0f - y)/(height/2.0f);
    float2 c = (float2)(-0.8f, 0.156f);
    float2 a = (float2)(jx, jy);

    for (int i=0; i<200; i++) {
        a = add(mul(a, a), c);
        if (magnitude(a) > 1000.0f)
            return 0;
    }
    return 1;
}

kernel
void julia(global unsigned char* out, const unsigned w, const unsigned h) {
    int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
    if (x<w && y<h) {
        int offset        = x + y * w;
        int juliaValue    = pixel(x, y, w, h);
        out[offset*4 + 1] = 255 * juliaValue;
        out[offset*4 + 0] = 0;
        out[offset*4 + 2] = 0;
        out[offset*4 + 3] = 255;
    }
}
)EOK";
// clang-format on

inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

void kernel(cl::Buffer& devOut, cl::CommandQueue& queue) {
    static std::once_flag compileFlag;
    static cl::Program prog;
    static cl::Kernel kern;

    std::call_once(compileFlag, [queue]() {
        prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(),
                           fractal_ocl_kernel, true);
        kern = cl::Kernel(prog, "julia");
    });

    auto juliaOp = cl::KernelFunctor<Buffer, unsigned, unsigned>(kern);

    static const NDRange local(8, 8);
    NDRange global(local[0] * divup(DIMX, (int)(local[0])),
                   local[1] * divup(DIMY, (int)(local[1])));

    juliaOp(EnqueueArgs(queue, global, local), devOut, DIMX, DIMY);
}

int main(void) {
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
        context       = createCLGLContext(wnd);
        Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        queue         = CommandQueue(context, device);

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
        createGLBuffer(&handle, img.pixels(), FORGE_IMAGE_BUFFER);

        // copy the data from compute buffer to graphics buffer
        copyToGLBuffer(handle, (ComputeResourceHandle)devOut(), img.size());

        do { wnd.draw(img); } while (!wnd.close());

        // destroy GL-CPU Interop buffer
        releaseGLBuffer(handle);

    } catch (forge::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}
