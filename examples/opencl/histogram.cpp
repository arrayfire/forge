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

const unsigned IMGW = 256;
const unsigned IMGH = 256;
const unsigned DIMX = 1000;
const unsigned DIMY = 800;
const unsigned IMG_SIZE = IMGW * IMGH * 4;
const unsigned WIN_ROWS = 1;
const unsigned WIN_COLS = 2;
const unsigned NBINS = 256;
const float PERSISTENCE = 0.5f;

static const std::string fractal_ocl_kernel =
"float rand(int x)\n"
"{\n"
"    x = (x << 13) ^ x;\n"
"    return ( 1.0 - ( (x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0);\n"
"}\n"
"kernel\n"
"void baseNoiseKernel(global float* baseNoise, unsigned IMGW, unsigned IMGH, long long seed)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"\n"
"    if (x<IMGW && y<IMGH) {\n"
"        baseNoise[y*IMGW+x] = curand_uniform(&state[0]);\n"
"    }\n"
"}\n"
"float interp(float x0, float x1, float t) {\n"
"        return x0 + (x1 - x0) * t;\n"
"}\n"
"kernel\n"
"void perlinKernel(global unsigned char* perlinNoise, global float* baseNoise,\n"
"                  unsigned IMGW, unsigned IMGH)\n"
"{\n"
"    int i = get_global_id(0);\n"
"    int j = get_global_id(1);\n"
"\n"
"    if (i<IMGW && j<IMGH) {\n"
"        float pnoise = 0.0f;\n"
"        float amp    = 1.0f;\n"
"        float tamp   = 0.0f;\n"
"\n"
"        for (int octave=6; octave>=0; --octave)\n"
"        {\n"
"            int period = 1 << octave;\n"
"            float freq = 1.0f / period;\n"
"\n"
"            int si0 = (i/period) * period;\n"
"            int si1 = (si0 + period) % IMGW;\n"
"            float hblend = (i - si0) * freq;\n"
"\n"
"            int sj0 = (j/period) * period;\n"
"            int sj1 = (sj0 + period) % IMGH;\n"
"            float vblend = (j - sj0) * freq;\n"
"\n"
"            float top = interp(baseNoise[si0+IMGW*sj0], baseNoise[si1+IMGW*sj0], hblend);\n"
"            float bot = interp(baseNoise[si0+IMGW*sj1], baseNoise[si1+IMGW*sj1], hblend);\n"
"\n"
"            pnoise += (amp * interp(top, bot, vblend));\n"
"\n"
"            tamp += amp;\n"
"            amp *= PERSISTENCE;\n"
"        }\n"
"\n"
"        uint offset = i+j*IMGW;\n"
"        perlinNoise[4*offset+0] = pnoise/tamp;\n"
"        perlinNoise[4*offset+1] = pnoise/tamp;\n"
"        perlinNoise[4*offset+2] = pnoise/tamp;\n"
"        perlinNoise[4*offset+3] = 255;\n"
"    }\n"
"}\n"
"kernel\n"
"void histogramKernel(const global unsigned char* perlinNoise, global int* histOut, const unsigned w, const unsigned h, const unsigned nbins)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"\n"
"    if (x<w && y<h) {\n"
"        int offset  = y * w + x;\n"
"        unsigned char noiseVal = perlinNoise[offset*4 + 0];\n"
"        atomic_add(histOut + convert_int(nbins * convert_float(noiseVal)/255.f) , 1);\n"
"    }\n"
"}\n"
"kernel\n"
"void setColorsKernel(global float* out, uint const seed)\n"
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

void kernel(cl::Buffer& baseNoise, cl::Buffer& perlinNoise, cl::Buffer& histOut, cl::Buffer& colors,
            cl::CommandQueue& queue)
{
    static std::once_flag   compileFlag;
    static cl::Program      prog;
    static cl::Kernel       genBaseNoise, genPerlinNoise, genHistogram, getHistColors;

    std::srand(std::time(0));

    std::call_once(compileFlag,
        [queue]() {
        prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), fractal_ocl_kernel, true);
            genBaseNoise   = cl::Kernel(prog, "baseNoiseKernel");
            genPerlinNoise = cl::Kernel(prog, "perlinKernel");
            genHistogram   = cl::Kernel(prog, "histogramKernel");
            getHistColors  = cl::Kernel(prog, "setColorsKernel");
        });

    static const NDRange local(16, 16);
    NDRange global(local[0] * divup(IMGW, local[0]),
                   local[1] * divup(IMGH, local[1]));

    genBaseNoise.setArg(0, baseNoise);
    genBaseNoise.setArg(1, IMGW);
    genBaseNoise.setArg(2, IMGH);
    genBaseNoise.setArg(3, std::rand());
    queue.enqueueNDRangeKernel(genBaseNoise, cl::NullRange, global, local);

    genPerlinNoise.setArg(0, perlinNoise);
    genPerlinNoise.setArg(1, baseNoise);
    genPerlinNoise.setArg(2, IMGW);
    genPerlinNoise.setArg(3, IMGH);
    queue.enqueueNDRangeKernel(genPerlinNoise, cl::NullRange, global, local);

    genHistogram.setArg(0, histOut);
    genHistogram.setArg(1, NBINS);
    queue.enqueueNDRangeKernel(genHistogram, cl::NullRange, global, local);

    static const NDRange global_hist(NBINS);
    getHistColors.setArg(0, colors);
    getHistColors.setArg(1, std::rand());
    queue.enqueueNDRangeKernel(getHistColors, cl::NullRange, global_hist);
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

        fg::Image img(IMGW, IMGH, FG_RGBA, u8);

        fg::Chart chart(FG_2D);
        /* set x axis limits to maximum and minimum values of data
         * and y axis limits to range [0, number of pixels ideally]
         * but practically total number of pixels as y range will skew
         * the histogram graph vertically. Therefore setting it to
         * 25% of total number of pixels */
        chart.setAxesLimits(0, 1, 0, IMGW*IMGH/(float)(NBINS/4.0));

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

        cl::Buffer baseNoise(context, CL_MEM_READ_WRITE, IMG_SIZE);
        cl::Buffer perlinNoise(context, CL_MEM_READ_WRITE, IMG_SIZE);
        cl::Buffer histOut(context, CL_MEM_READ_WRITE, NBINS * sizeof(int));
        cl::Buffer colors(context, CL_MEM_READ_WRITE, 3 * NBINS * sizeof(float));

        /*
         * generate image, and prepare data to pass into
         * Histogram's underlying vertex buffer object
         */
        kernel(baseNoise, perlinNoise, histOut, colors, queue);
         /* To help the users with copying the data from compute
         * memory to display memory, Forge provides copy headers
         * along with the library to help with this task
         */
        fg::copy(img, perlinNoise, queue);
        fg::copy(hist.vertices(), hist.verticesSize(), histOut, queue);
        fg::copy(hist.vertices(), hist.verticesSize(), colors, queue);

        do {
            wnd.draw(0, 0, img,  "Dynamic Perlin Noise" );
            wnd.draw(1, 0, chart, "Histogram of Noisy Image");
            wnd.swapBuffers();

            // limit histogram update frequency
            //if (fmod(persistance, 0.4f) < 0.02f) {
                kernel(baseNoise, perlinNoise, histOut, colors, queue);

                fg::copy(img, perlinNoise, queue);
                fg::copy(hist.vertices(), hist.verticesSize(), histOut, queue);
                fg::copy(hist.vertices(), hist.verticesSize(), colors, queue);
            //}

        } while(!wnd.close());
    }catch (fg::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}
