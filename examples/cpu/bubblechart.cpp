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
#include <complex>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <functional>
#include <iterator>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

const float FRANGE_START = 0.f;
const float FRANGE_END = 2.f * 3.1415926f;

using namespace std;
void map_range_to_vec_vbo(float range_start, float range_end, float dx,
                          std::vector<float> &vec,
                          float (*map) (float))
{
    if(range_start > range_end && dx > 0) return;
    for(float i=range_start; i < range_end; i+=dx){
        vec.push_back(i);
        vec.push_back((*map)(i));
    }
}

int main(void)
{
    std::vector<float> cosData;
    std::vector<float> tanData;

    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, cosData, &cosf);
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, tanData, &tanf);

    std::random_device r;

    std::default_random_engine e1(r());
    std::mt19937_64 gen(r());

    std::uniform_int_distribution<int> uDist(20, 80);
    std::uniform_real_distribution<float> cDist(0.2, 0.6);
    std::uniform_real_distribution<float> fDist(0.4, 0.6);

    auto clr = std::bind(cDist, gen);
    auto rnd = std::bind(uDist, e1);
    auto alp = std::bind(fDist, gen);

    std::vector<float> colors(3*tanData.size());
    std::vector<float> alphas(tanData.size());
    std::vector<float> radii(tanData.size());

    std::generate(colors.begin(), colors.end(), clr);
    std::generate(radii.begin(), radii.end(), rnd);
    std::generate(alphas.begin(), alphas.end(), alp);

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other forge::* object to be created successfully
     */
    forge::Window wnd(DIMX, DIMY, "Bubble chart with Transparency Demo");
    wnd.makeCurrent();

    forge::Chart chart(FG_CHART_2D);
    chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.0f, 1.0f);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    forge::Plot plt1 = chart.plot(cosData.size()/2, forge::f32,
                               FG_PLOT_LINE, FG_MARKER_TRIANGLE); //or specify a specific plot type
    forge::Plot plt2 = chart.plot(tanData.size()/2, forge::f32,
                               FG_PLOT_LINE, FG_MARKER_CIRCLE); //last parameter specifies marker shape

    /* Set plot colors */
    plt1.setColor(FG_RED);
    plt2.setColor(FG_GREEN);            //use a forge predefined color
    /* Set plot legends */
    plt1.setLegend("Cosine");
    plt2.setLegend("Tangent");
    /* set plot global marker size */
    plt1.setMarkerSize(20);
    /* copy your data into the opengl buffer object exposed by
     * forge::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */

    GfxHandle* handles[5];

    // create GL-CPU interop buffers
    createGLBuffer(&handles[0], plt1.vertices(), FORGE_VBO);
    createGLBuffer(&handles[1], plt2.vertices(), FORGE_VBO);
    createGLBuffer(&handles[2], plt2.colors(), FORGE_VBO);
    createGLBuffer(&handles[3], plt2.alphas(), FORGE_VBO);
    createGLBuffer(&handles[4], plt2.markers(), FORGE_VBO);

    // copy the data from compute buffer to graphics buffer
    copyToGLBuffer(handles[0], (ComputeResourceHandle)cosData.data(), plt1.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)tanData.data(), plt2.verticesSize());

    /* update color value for tan graph */
    copyToGLBuffer(handles[2], (ComputeResourceHandle)colors.data(), plt2.colorsSize());
    /* update alpha values for tan graph */
    copyToGLBuffer(handles[3], (ComputeResourceHandle)alphas.data(), plt2.alphasSize());
    /* update marker sizes for tan graph markers */
    copyToGLBuffer(handles[4], (ComputeResourceHandle)radii.data(), plt2.markersSize());

    do {
        wnd.draw(chart);
    } while(!wnd.close());

    // destroy GL-CPU Interop buffer
    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);
    releaseGLBuffer(handles[3]);
    releaseGLBuffer(handles[4]);
    return 0;
}
