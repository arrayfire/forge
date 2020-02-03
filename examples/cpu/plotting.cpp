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
#include <fg/compute_copy.h>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

const float FRANGE_START = 0.f;
const float FRANGE_END   = 2.f * 3.1415926f;

using namespace std;
void map_range_to_vec_vbo(float range_start, float range_end, float dx,
                          std::vector<float>& vec, float (*map)(float)) {
    if (range_start > range_end && dx > 0) return;
    for (float i = range_start; i < range_end; i += dx) {
        vec.push_back(i);
        vec.push_back((*map)(i));
    }
}

int main(void) {
    std::vector<float> sinData;
    std::vector<float> cosData;
    std::vector<float> tanData;
    std::vector<float> logData;
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, sinData, &sinf);
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, cosData, &cosf);
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, tanData, &tanf);
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, logData, &log10f);

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other forge::* object to be created successfully
     */
    forge::Window wnd(DIMX, DIMY, "Plotting Demo");
    wnd.makeCurrent();

    forge::Chart chart(FG_CHART_2D);
    chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.0f, 1.0f);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    forge::Plot plt0 = chart.plot((unsigned)(sinData.size() / 2),
                                  forge::f32);  // create a default plot
    forge::Plot plt1 =
        chart.plot((unsigned)(cosData.size() / 2), forge::f32, FG_PLOT_LINE,
                   FG_MARKER_NONE);  // or specify a specific plot type
    forge::Plot plt2 = chart.plot(
        (unsigned)(tanData.size() / 2), forge::f32, FG_PLOT_LINE,
        FG_MARKER_TRIANGLE);  // last parameter specifies marker shape
    forge::Plot plt3 = chart.plot((unsigned)(logData.size() / 2), forge::f32,
                                  FG_PLOT_SCATTER, FG_MARKER_CROSS);

    /*
     * Set plot colors
     */
    plt0.setColor(FG_RED);
    plt1.setColor(FG_BLUE);
    plt2.setColor(FG_YELLOW);                 // use a forge predefined color
    plt3.setColor((forge::Color)0x257973FF);  // or any hex-valued color
    /*
     * Set plot legends
     */
    plt0.setLegend("Sine");
    plt1.setLegend("Cosine");
    plt2.setLegend("Tangent");
    plt3.setLegend("Log base 10");

    GfxHandle* handles[4];
    createGLBuffer(&handles[0], plt0.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[1], plt1.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[2], plt2.vertices(), FORGE_VERTEX_BUFFER);
    createGLBuffer(&handles[3], plt3.vertices(), FORGE_VERTEX_BUFFER);

    /* copy your data into the pixel buffer object exposed by
     * forge::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copyToGLBuffer(handles[0], (ComputeResourceHandle)sinData.data(),
                   plt0.verticesSize());
    copyToGLBuffer(handles[1], (ComputeResourceHandle)cosData.data(),
                   plt1.verticesSize());
    copyToGLBuffer(handles[2], (ComputeResourceHandle)tanData.data(),
                   plt2.verticesSize());
    copyToGLBuffer(handles[3], (ComputeResourceHandle)logData.data(),
                   plt3.verticesSize());

    do { wnd.draw(chart); } while (!wnd.close());

    releaseGLBuffer(handles[0]);
    releaseGLBuffer(handles[1]);
    releaseGLBuffer(handles[2]);
    releaseGLBuffer(handles[3]);

    return 0;
}
