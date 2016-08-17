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

using namespace std;

static const float XMIN = -8.0f;
static const float XMAX = 8.f;
static const float YMIN = -8.0f;
static const float YMAX = 8.f;

const float DX = 0.5;
const size_t XSIZE = (XMAX-XMIN)/DX;
const size_t YSIZE = (YMAX-YMIN)/DX;

void genSurface(float dx, std::vector<float> &vec )
{
    vec.clear();
    for(float x=XMIN; x < XMAX; x+=dx){
        for(float y=YMIN; y < YMAX; y+=dx){
            vec.push_back(x);
            vec.push_back(y);
            float z = sqrt(x*x+y*y) + 2.2204e-16;
            vec.push_back(sin(z)/z);
        }
    }
}

int main(void)
{
    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other forge::* object to be created successfully
     */
    forge::Window wnd(1024, 768, "3d Surface Demo");
    wnd.makeCurrent();

    forge::Chart chart(FG_CHART_3D);
    chart.setAxesLimits(-10.f, 10.f, -10.f, 10.f, -0.5f, 1.f);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    forge::Surface surf = chart.surface(XSIZE, YSIZE, forge::f32);
    surf.setColor(FG_YELLOW);

    //generate a surface
    std::vector<float> function;

    genSurface(DX, function);

    GfxHandle* handle;
    createGLBuffer(&handle, surf.vertices(), FORGE_VBO);

    /* copy your data into the pixel buffer object exposed by
     * forge::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copyToGLBuffer(handle, (ComputeResourceHandle)function.data(), surf.verticesSize());

    do {
        wnd.draw(chart);
    } while(!wnd.close());

    releaseGLBuffer(handle);

    return 0;
}
