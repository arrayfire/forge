/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <forge.h>
#include <CPUCopy.hpp>
#include <complex>
#include <cmath>
#include <vector>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;

static const float ZMIN = 0.1f;
static const float ZMAX = 10.f;

const float DX = 0.005;
const size_t ZSIZE = (ZMAX-ZMIN)/DX+1;

using namespace std;

void generateCurve(float t, float dx, std::vector<float> &vec )
{
    vec.clear();
    for (int i=0; i < (int)ZSIZE; ++i) {
        float z = ZMIN + i*dx;
        vec.push_back(cos(z*t+t)/z);
        vec.push_back(sin(z*t+t)/z);
        vec.push_back(z+0.1*sin(t));
    }
}

int main(void)
{
    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "Three dimensional line plot demo");
    wnd.makeCurrent();

    fg::Chart chart(FG_CHART_3D);
    chart.setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, 0.f, 10.f);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    fg::Plot plot3 = chart.plot(ZSIZE, fg::f32);

    //generate a surface
    std::vector<float> function;
    static float t=0;
    generateCurve(t, DX, function);
    /* copy your data into the pixel buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    fg::copy(plot3.vertices(), plot3.verticesSize(), (const void*)function.data());

    do {
        t+=0.01;
        generateCurve(t, DX, function);
        fg::copy(plot3.vertices(), plot3.verticesSize(), (const void*)function.data());
        wnd.draw(chart);
    } while(!wnd.close());

    return 0;
}
