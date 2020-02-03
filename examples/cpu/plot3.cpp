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

static const float ZMIN = 0.1f;
static const float ZMAX = 10.f;

const float DX     = 0.005f;
const size_t ZSIZE = (size_t)((ZMAX - ZMIN) / DX + 1);

using namespace std;

void generateCurve(float t, float dx, std::vector<float>& vec) {
    vec.clear();
    for (int i = 0; i < (int)ZSIZE; ++i) {
        float z = ZMIN + i * dx;
        vec.push_back((float)(cos(z * t + t) / z));
        vec.push_back((float)(sin(z * t + t) / z));
        vec.push_back((float)(z + 0.1 * sin(t)));
    }
}

int main(void) {
    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other forge::* object to be created successfully
     */
    forge::Window wnd(DIMX, DIMY, "Three dimensional line plot demo");
    wnd.makeCurrent();

    forge::Chart chart(FG_CHART_3D);

    chart.setAxesLabelFormat("%3.1f", "%3.1f", "%.2e");

    chart.setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, 0.f, 10.f);

    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    forge::Plot plot3 = chart.plot(ZSIZE, forge::f32);

    // generate a surface
    std::vector<float> function;
    static float t = 0;
    generateCurve(t, DX, function);

    GfxHandle* handle;
    createGLBuffer(&handle, plot3.vertices(), FORGE_VERTEX_BUFFER);

    /* copy your data into the pixel buffer object exposed by
     * forge::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copyToGLBuffer(handle, (ComputeResourceHandle)function.data(),
                   plot3.verticesSize());

    do {
        t += 0.01f;
        generateCurve(t, DX, function);
        copyToGLBuffer(handle, (ComputeResourceHandle)function.data(),
                       plot3.verticesSize());
        wnd.draw(chart);
    } while (!wnd.close());

    releaseGLBuffer(handle);

    return 0;
}
