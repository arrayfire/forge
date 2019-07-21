/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <forge.h>
#define USE_FORGE_CPU_COPY_HELPERS
#include <ComputeCopy.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

const unsigned DIMX     = 1000;
const unsigned DIMY     = 800;
const unsigned NSECTORS = 10;

std::vector<float> generateSectors(unsigned count = NSECTORS) {
    std::vector<float> result;
    float prefixSum = 0;
    for (; count > 0; --count) {
        result.push_back(prefixSum);
        result.push_back(std::rand() & 0xffff);
        prefixSum += result.back();
    }
    return result;
}

std::vector<float> generateColors(unsigned count = NSECTORS) {
    std::vector<float> result;
    for (; count > 0; --count) {
        for (int channel = 0; channel < 3; ++channel)
            result.push_back(std::rand() / (float)RAND_MAX);
    }
    return result;
}

int main(int argc, char* argv[]) {
    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other forge::* object to be created successfully
     */
    forge::Window wnd(DIMX, DIMY, "Pie Demo");
    wnd.makeCurrent();

    forge::Chart chart(FG_CHART_2D);

    /*
     * Create pie object specifying number of bins
     */
    forge::Pie pie = chart.pie(NSECTORS, forge::f32);

    GfxHandle* handles[2];
    createGLBuffer(&handles[0], pie.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[1], pie.colors(), FORGE_VERTEX_BUFFER);

    std::vector<float> pieArray = generateSectors();
    std::vector<float> colArray = generateColors();

    /* set the axes limits to minimum and maximum values of data */
    float valueRange =
        pieArray[pieArray.size() - 2] + pieArray[pieArray.size() - 1];
    chart.setAxesLimits(0, valueRange, 0, valueRange);

    copyToGLBuffer(handles[0], (ComputeResourceHandle)pieArray.data(),
                   pie.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)colArray.data(),
                   pie.colorsSize());

    do { wnd.draw(chart); } while (!wnd.close());

    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);

    return 0;
}
