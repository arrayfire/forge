/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <forge.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>
#include "cl_helpers.h"

using namespace cl;
using namespace std;

const unsigned IMGW     = 256;
const unsigned IMGH     = 256;
const unsigned DIMX     = 1000;
const unsigned DIMY     = 800;
const unsigned IMG_SIZE = IMGW * IMGH * 4;
const unsigned NBINS    = 256;
const float PERSISTENCE = 0.5f;

#define USE_FORGE_OPENCL_COPY_HELPERS
#include <fg/compute_copy.h>

// clang-format off
static const std::string perlinKernels =
R"EOK(
float rand(int x) {
    x = (x << 13) ^ x;
    return (1.0 - ((x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) /
                      1073741824.0);
}

float interp(float x0, float x1, float t) { return x0 + (x1 - x0) * t; }

kernel void init(global float* base, global float* perlin, int IMGW, int IMGH,
                 int randSeed) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < IMGW && y < IMGH) {
        int i     = x + y * IMGW;
        base[i]   = (1 + rand(randSeed * i)) / 2.0f;
        perlin[i] = 0.0f;
    }
}

kernel void compute(global float* perlin, global float* base, unsigned IMGW,
                    unsigned IMGH, float amp, int period) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < IMGW && y < IMGH) {
        int index = y * IMGW + x;

        float freq = 1.0f / period;

        int si0      = (x / period) * period;
        int si1      = (si0 + period) % IMGW;
        float hblend = (x - si0) * freq;

        int sj0      = (y / period) * period;
        int sj1      = (sj0 + period) % IMGH;
        float vblend = (y - sj0) * freq;

        float top =
            interp(base[si0 + IMGW * sj0], base[si1 + IMGW * sj0], hblend);
        float bot =
            interp(base[si0 + IMGW * sj1], base[si1 + IMGW * sj1], hblend);

        perlin[index] += (amp * interp(top, bot, vblend));
    }
}

kernel void normalizeNoise(global float* perlin, unsigned IMGW, unsigned IMGH,
                           float tamp) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < IMGW && y < IMGH) {
        int index     = y * IMGW + x;
        perlin[index] = perlin[index] / tamp;
    }
}

kernel void fillImage(global unsigned char* ptr, unsigned width,
                      unsigned height, global float* perlin, unsigned IMGW,
                      unsigned IMGH) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int offset = x + y * width;

        unsigned u = (unsigned)(IMGW * x / (float)(width));
        unsigned v = (unsigned)(IMGH * y / (float)(height));
        int idx    = u + v * IMGW;

        unsigned char val   = 255 * perlin[idx];
        ptr[offset * 4 + 0] = val;
        ptr[offset * 4 + 1] = val;
        ptr[offset * 4 + 2] = val;
        ptr[offset * 4 + 3] = 255;
    }
}

kernel void memSet(global int* out, unsigned len) {
    if (get_global_id(0) < len) out[get_global_id(0)] = 0;
}

kernel void histogram(const global unsigned char* perlinNoise,
                      global int* histOut, const unsigned w, const unsigned h,
                      const unsigned nbins) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < w && y < h) {
        int offset             = y * w + x;
        unsigned char noiseVal = perlinNoise[offset * 4 + 0];
        offset                 = (int)(nbins * (noiseVal / 255.f));
        atomic_add(histOut + offset, 1);
    }
}

kernel void setColors(global float* out, unsigned rseed, unsigned gseed,
                      unsigned bseed) {
    int i          = get_global_id(0);
    out[3 * i + 0] = (1 + rand(rseed * i)) / 2.0f;
    out[3 * i + 1] = (1 + rand(gseed * i)) / 2.0f;
    out[3 * i + 2] = (1 + rand(bseed * i)) / 2.0f;
};
)EOK";
// clang-format on

inline
int divup(int a, int b)
{
    return (a + b - 1) / b;
}

void kernel(cl::Buffer& image, cl::Buffer& base, cl::Buffer& perlin,
            cl::Buffer& histOut, cl::Buffer& colors, cl::CommandQueue& queue,
            cl::Device& device) {
    static bool compileFlag = true;
    static cl::Program prog;
    static cl::Kernel initKernel, computeKernel, normKernel, fillKernel;
    static cl::Kernel memSetKernel, genHistogram, genHistColors;

    std::srand((unsigned)(std::time(0)));

    if (compileFlag) {
        try {
            prog = cl::Program(queue.getInfo<CL_QUEUE_CONTEXT>(), perlinKernels,
                               false);

            std::vector<cl::Device> devs;
            devs.push_back(device);
            prog.build(devs);

            initKernel    = cl::Kernel(prog, "init");
            computeKernel = cl::Kernel(prog, "compute");
            normKernel    = cl::Kernel(prog, "normalizeNoise");
            fillKernel    = cl::Kernel(prog, "fillImage");
            memSetKernel  = cl::Kernel(prog, "memSet");
            genHistogram  = cl::Kernel(prog, "histogram");
            genHistColors = cl::Kernel(prog, "setColors");
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

    static const NDRange local(16, 16);
    NDRange global(local[0] * divup(IMGW, (int)(local[0])),
                   local[1] * divup(IMGH, (int)(local[1])));

    float persistence = 0.5f;
    float amp         = 1.0f;
    float tamp        = 0.0f;

    initKernel.setArg(0, base);
    initKernel.setArg(1, perlin);
    initKernel.setArg(2, IMGW);
    initKernel.setArg(3, IMGH);
    initKernel.setArg(4, std::rand());
    queue.enqueueNDRangeKernel(initKernel, cl::NullRange, global, local);

    for (int octave = 6; octave >= 0; --octave) {
        int period = 1 << octave;
        computeKernel.setArg(0, perlin);
        computeKernel.setArg(1, base);
        computeKernel.setArg(2, IMGW);
        computeKernel.setArg(3, IMGH);
        computeKernel.setArg(4, amp);
        computeKernel.setArg(5, period);
        queue.enqueueNDRangeKernel(computeKernel, cl::NullRange, global, local);
        tamp += amp;
        amp *= persistence;
    }

    normKernel.setArg(0, perlin);
    normKernel.setArg(1, IMGW);
    normKernel.setArg(2, IMGH);
    normKernel.setArg(3, tamp);
    queue.enqueueNDRangeKernel(normKernel, cl::NullRange, global, local);

    fillKernel.setArg(0, image);
    fillKernel.setArg(1, IMGW);
    fillKernel.setArg(2, IMGH);
    fillKernel.setArg(3, perlin);
    fillKernel.setArg(4, IMGW);
    fillKernel.setArg(5, IMGH);
    queue.enqueueNDRangeKernel(fillKernel, cl::NullRange, global, local);

    static const NDRange global_hist(NBINS);

    memSetKernel.setArg(0, histOut);
    memSetKernel.setArg(1, NBINS);
    queue.enqueueNDRangeKernel(memSetKernel, cl::NullRange, global_hist);

    genHistogram.setArg(0, image);
    genHistogram.setArg(1, histOut);
    genHistogram.setArg(2, IMGW);
    genHistogram.setArg(3, IMGH);
    genHistogram.setArg(4, NBINS);
    queue.enqueueNDRangeKernel(genHistogram, cl::NullRange, global, local);

    genHistColors.setArg(0, colors);
    genHistColors.setArg(1, std::rand());
    genHistColors.setArg(2, std::rand());
    genHistColors.setArg(3, std::rand());
    queue.enqueueNDRangeKernel(genHistColors, cl::NullRange, global_hist);
}

int main(void) {
    try {
        /*
         * First Forge call should be a window creation call
         * so that necessary OpenGL context is created for any
         * other forge::* object to be created successfully
         */
        forge::Window wnd(DIMX, DIMY, "Histogram Demo");
        wnd.makeCurrent();

        forge::Image img(IMGW, IMGH, FG_RGBA, forge::u8);

        forge::Chart chart(FG_CHART_2D);

        chart.setAxesLabelFormat("%3.1f", "%.2e");

        /* set x axis limits to maximum and minimum values of data
         * and y axis limits to range [0, number of pixels ideally]
         * but practically total number of pixels as y range will skew
         * the histogram graph vertically. Therefore setting it to
         * 25% of total number of pixels */
        chart.setAxesLimits(0, 1, 0, IMGW * IMGH / (float)(NBINS / 4.0));

        /*
         * Create histogram object specifying number of bins
         */
        forge::Histogram hist = chart.histogram(NBINS, forge::s32);
        /*
         * Set histogram colors
         */
        hist.setColor(FG_YELLOW);

        /*
         * Helper function to create a CLGL interop context.
         * This function checks for if the extension is available
         * and creates the context on the appropriate device.
         * Note: context and queue are defined in cl_helpers.h
         */
        context       = createCLGLContext(wnd);
        Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
        queue         = CommandQueue(context, device);

        cl::Buffer image(context, CL_MEM_READ_WRITE, IMG_SIZE);
        cl::Buffer baseNoise(context, CL_MEM_READ_WRITE, IMG_SIZE);
        cl::Buffer perlinNoise(context, CL_MEM_READ_WRITE, IMG_SIZE);
        cl::Buffer histOut(context, CL_MEM_READ_WRITE, NBINS * sizeof(int));
        cl::Buffer colors(context, CL_MEM_READ_WRITE,
                          3 * NBINS * sizeof(float));

        GfxHandle* handles[3];

        createGLBuffer(&handles[0], img.pixels(), FORGE_IMAGE_BUFFER);
        createGLBuffer(&handles[1], hist.vertices(), FORGE_VERTEX_BUFFER);
        createGLBuffer(&handles[2], hist.colors(), FORGE_VERTEX_BUFFER);

        unsigned frame = 0;
        do {
            if (frame % 8 == 0) {
                kernel(image, baseNoise, perlinNoise, histOut, colors, queue,
                       device);

                copyToGLBuffer(handles[0], (ComputeResourceHandle)image(),
                               img.size());
                copyToGLBuffer(handles[1], (ComputeResourceHandle)histOut(),
                               hist.verticesSize());
                copyToGLBuffer(handles[2], (ComputeResourceHandle)colors(),
                               hist.colorsSize());

                frame = 0;
            }

            /*
             * Split the window into grid regions
             */
            wnd.draw(1, 2, 0, img, "Dynamic Perlin Noise");
            wnd.draw(1, 2, 1, chart, "Histogram of Noisy Image");

            wnd.swapBuffers();
            frame++;
        } while (!wnd.close());

        releaseGLBuffer(handles[0]);
        releaseGLBuffer(handles[1]);
        releaseGLBuffer(handles[2]);

    } catch (forge::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    } catch (cl::Error err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}
