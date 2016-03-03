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
#include <cmath>
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
const unsigned IMG_SIZE = DIMX * DIMY * 4;

const unsigned WIN_ROWS = 1;
const unsigned WIN_COLS = 2;

static float persistance = 0.1;
const unsigned NBINS = 5;

static const std::string fractal_ocl_kernel =
"__constant int perm[] = { 26, 58, 229, 82, 132, 72, 144, 251, 196, 192, 127, 16,\n"
"    68, 118, 104, 213, 91, 105, 203, 61, 59, 93, 136, 249, 27, 137, 141, 223, 119,\n"
"    193, 155, 43, 71, 244, 170, 115, 201, 150, 165, 78, 208, 53, 90, 232, 209, 83,\n"
"    45, 174, 140, 178, 220, 184, 70, 6, 202, 17, 128, 212, 117, 200, 254, 57, 248,\n"
"    62, 164, 172, 19, 177, 241, 103, 48, 38, 210, 129, 23, 211, 8, 112, 107,  126,\n"
"    252,  198, 32, 123, 111,  176,  206, 15, 219, 221, 147, 245, 67, 92, 108, 143,\n"
"    54, 102, 169, 22, 74, 124, 181, 186, 138, 18, 7, 34, 81, 46, 120, 236, 89,228,\n"
"    197, 205, 13, 63, 134,  242, 157, 135, 237, 35, 234, 49, 85, 76, 148, 188, 98,\n"
"    87, 173, 84, 226, 11, 125, 122, 2, 94, 191, 179, 175, 187, 133, 231, 154,  44,\n"
"    28, 110, 247, 121, 146, 240, 97, 88, 130,195, 30, 25, 56, 171, 80, 69, 139, 9,\n"
"    238, 160, 227, 204, 31, 40, 66, 77, 21, 159,  162, 207,  167, 214, 10, 3, 149,\n"
"    194, 239, 166,  145, 235, 20, 50, 113, 189, 99, 37, 86, 42, 168, 114, 96, 246,\n"
"    183, 250, 233, 156, 52,  65, 131, 47,  255, 5, 33, 217, 73, 4, 60, 64, 109, 0,\n"
"    215, 100, 180, 12, 24, 190, 222, 106, 41, 216, 230, 161, 55, 152, 79, 75, 142,\n"
"    36, 101, 1, 253, 225, 51, 224, 182, 116, 218, 95, 39, 158,  14, 243, 151, 163,\n"
"    29, 153, 199, 185\n"
"};\n"
"\n"
"__constant float2 default_gradients[4] = {(float2) (1,1), (float2)(-1,1), (float2)(1,-1), (float2)(-1,-1) };\n"
"\n"
"float interp(float t){\n"
"    return ((6 * t - 15) * t + 10) * t * t * t;\n"
"}\n"
"\n"
"float lerp (float x0, float x1, float t) {\n"
"        return x0 + (x1 - x0) * t;\n"
"}\n"
"\n"
"float perlinNoise(float x, float y, int tileSize)\n"
"{\n"
"    int x_grid = x/tileSize;\n"
"    int y_grid = y/tileSize;\n"
"    unsigned rand_id0 = perm[(x_grid+2*y_grid) % 256 ] % 4;\n"
"    unsigned rand_id1 = perm[(x_grid+1+2*y_grid) % 256 ] % 4;\n"
"    unsigned rand_id2 = perm[(x_grid+2*(y_grid+1)) % 256 ] % 4;\n"
"    unsigned rand_id3 = perm[(x_grid+1+2*(y_grid+1)) % 256 ] % 4;\n"
"\n"
"    x=fmod(x,tileSize)/tileSize;\n"
"    y=fmod(y,tileSize)/tileSize;\n"
"    float u = interp(x);\n"
"    float v = interp(y);\n"
"\n"
"    float influence_vecs[4];\n"
"    influence_vecs[0] = dot((float2)(x,y) - (float2)(0,0), default_gradients[rand_id0]);\n"
"    influence_vecs[1] = dot((float2)(x,y) - (float2)(1,0), default_gradients[rand_id1]);\n"
"    influence_vecs[2] = dot((float2)(x,y) - (float2)(0,1), default_gradients[rand_id2]);\n"
"    influence_vecs[3] = dot((float2)(x,y) - (float2)(1,1),             default_gradients[rand_id3]);\n"
"\n"
"    return lerp(lerp(influence_vecs[0], influence_vecs[1], u), lerp(influence_vecs[2], influence_vecs[3], u), v);\n"
"}\n"
"float octavesPerlin(float x, float y, int octaves, float persistence, int tileSize)\n"
"{\n"
"    float total = 0, max_value = 0;\n"
"    float amplitude = 1, frequency = 1;\n"
"    for(int i=0; i<octaves; ++i){\n"
"        total += perlinNoise( x*frequency, y*frequency, tileSize) * amplitude;\n"
"        max_value += amplitude;\n"
" \n"
"        amplitude *= persistence;\n"
"        frequency *= 2;\n"
"    }\n"
"    return total/max_value;\n"
"}\n"
"\n"
"kernel\n"
"void image_gen(global unsigned char* out, const unsigned w, const unsigned h, float persistance, int tileSize)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"\n"
"    if (x<w && y<h) {\n"
"        int offset  = y * w + x;\n"
"        int octaves = 4;\n"
"        int noiseValue    = 255 * octavesPerlin(x, y, octaves, persistance, tileSize);\n"
"        out[offset*4 + 0] = noiseValue;\n"
"        out[offset*4 + 1] = noiseValue;\n"
"        out[offset*4 + 2] = noiseValue;\n"
"        out[offset*4 + 3] = 255;\n"
"    }\n"
"}\n"
"\n"
"kernel\n"
"void hist_freq(const global unsigned char* out, global int* hist_array, const unsigned w, const unsigned h, const unsigned nbins)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"\n"
"    if (x<w && y<h) {\n"
"        int offset  = y * w + x;\n"
"        unsigned char noiseVal = out[offset*4 + 0];\n"
"        atomic_add(hist_array + convert_int(nbins * convert_float(noiseVal)/255.f) , 1);\n"
"    }\n"
"}\n"
"kernel\n"
"void zero_buffer(global unsigned int* out, const unsigned size)\n"
"{\n"
"    if(get_global_id(0) < size) {\n"
"        out[get_global_id(0)] = 0;\n"
"    }\n"
"}\n"
"float rand(int x)\n"
"{\n"
"    x = (x << 13) ^ x;\n"
"    return ( 1.0 - ( (x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0);\n"
"}\n"
"kernel\n"
"void set_colors(global float* out, uint const seed)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    out[3*i+0] = (1+rand(seed * i))/2.0f;\n"
"    out[3*i+1] = (1+rand(seed ^ i))/2.0f;\n"
"    out[3*i+2] = (1+rand(seed / i))/2.0f;\n"
"}\n";

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(cl::Buffer& devOut, cl::Buffer& histOut, cl::Buffer& colors, cl::CommandQueue& queue)
{
    static std::once_flag   compileFlag;
    static cl::Program      prog;
    static cl::Kernel       kern_img, kern_hist, kern_zero, kern_colors;

    std::call_once(compileFlag,
        [queue]() {
        prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), fractal_ocl_kernel, true);
            kern_img    = cl::Kernel(prog, "image_gen");
            kern_hist   = cl::Kernel(prog, "hist_freq");
            kern_zero   = cl::Kernel(prog, "zero_buffer");
            kern_colors = cl::Kernel(prog, "set_colors");
        });

    static const NDRange local(16, 16);
    NDRange global(local[0] * divup(DIMX, local[0]),
                   local[1] * divup(DIMY, local[1]));

    static int tileSize = 32; tileSize++;
    persistance += 0.01;
    kern_img.setArg(0, devOut);
    kern_img.setArg(1, DIMX);
    kern_img.setArg(2, DIMY);
    kern_img.setArg(3, persistance);
    kern_img.setArg(4, tileSize);
    queue.enqueueNDRangeKernel(kern_img, cl::NullRange, global, local);

    static const NDRange global_hist(NBINS);
    kern_zero.setArg(0, histOut);
    kern_zero.setArg(1, NBINS);
    queue.enqueueNDRangeKernel(kern_zero, cl::NullRange, global_hist);

    kern_hist.setArg(0, devOut);
    kern_hist.setArg(1, histOut);
    kern_hist.setArg(2, DIMX);
    kern_hist.setArg(3, DIMY);
    kern_hist.setArg(4, NBINS);
    queue.enqueueNDRangeKernel(kern_hist, cl::NullRange, global, local);

    kern_colors.setArg(0, colors);
    kern_colors.setArg(1, std::rand());
    queue.enqueueNDRangeKernel(kern_colors, cl::NullRange, global_hist);
}

int main(void)
{
    try {
        /*
        * First Forge call should be a window creation call
        * so that necessary OpenGL context is created for any
        * other fg::* object to be created successfully
        */
        fg::Window wnd(DIMX, DIMY, "Histogram Demo");
        wnd.makeCurrent();

        /*
         * Split the window into grid regions
         */
        wnd.grid(WIN_ROWS, WIN_COLS);

        fg::Image img(DIMX, DIMY, FG_RGBA, u8);

        fg::Chart chart(FG_2D);
        /* set x axis limits to maximum and minimum values of data
         * and y axis limits to range [0, nBins]*/
        chart.setAxesLimits(0, 1, 0, 1000);

        /*
         * Create histogram object specifying number of bins
         */
        fg::Histogram hist = chart.histogram(NBINS, u8);
        /*
         * Set histogram colors
         */
        hist.setColor(FG_YELLOW);

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

        cl::Buffer devOut(context, CL_MEM_READ_WRITE, IMG_SIZE);
        cl::Buffer histOut(context, CL_MEM_READ_WRITE, NBINS * sizeof(int));
        cl::Buffer colors(context, CL_MEM_READ_WRITE, 3 * NBINS * sizeof(float));

        /*
         * generate image, and prepare data to pass into
         * Histogram's underlying vertex buffer object
         */
        kernel(devOut, histOut, colors, queue);
         /* To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        fg::copy(img, devOut, queue);
        fg::copy(hist.vertices(), hist.verticesSize(), histOut, queue);
        fg::copy(hist.vertices(), hist.verticesSize(), colors, queue);

        do {
            wnd.draw(0, 0, img,  "Dynamic Perlin Noise" );
            wnd.draw(1, 0, chart, "Histogram of Noisy Image");

            wnd.swapBuffers();

            kernel(devOut, histOut, colors, queue);
            fg::copy(img, devOut, queue);

            // limit histogram update frequency
            if (fmod(persistance, 0.4f) < 0.02f) {
                fg::copy(hist.vertices(), hist.verticesSize(), histOut, queue);
                fg::copy(hist.vertices(), hist.verticesSize(), colors, queue);
            }

        } while(!wnd.close());
    }catch (fg::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}
