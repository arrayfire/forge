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

const unsigned DIMX = 800;
const unsigned DIMY = 800;

static const float XMIN = -1.0f;
static const float XMAX = 2.f;
static const float YMIN = -1.0f;
static const float YMAX = 1.f;

const float DX = 0.01;
const size_t XSIZE = (XMAX-XMIN)/DX+1;
const size_t YSIZE = (YMAX-YMIN)/DX+1;

bool set=false;
using namespace std;
void gen_surface(float t, float dx, std::vector<float> &vec ){
    vec.clear();
    for(float x=XMIN; x < XMAX; x+=dx){
        for(float y=YMIN; y < YMAX; y+=dx){
            vec.push_back(x);
            vec.push_back(y);
            vec.push_back(10*x*-abs(y) * cos(x*x*(y+t))+sin(y*(x+t))-1.5);
        }
    }
}

int main(void)
{
    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "3d Surface Demo");
    wnd.makeCurrent();

    fg::Chart chart(fg::FG_3D);
    chart.setAxesLimits(-1.1f, 1.1f, -1.1f, 1.1f, -5.f, 10.f);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    fg::Surface surf = chart.surface(XSIZE, YSIZE, fg::f32);
    surf.setColor(fg::FG_YELLOW);

    //generate a surface
    std::vector<float> function;
    static float t=0;
    gen_surface(t, DX, function);
    /* copy your data into the pixel buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    fg::copy(surf.vertices(), surf.verticesSize(), (const void*)function.data());

    do {
        t+=0.07;
        gen_surface(t, DX, function);
        fg::copy(surf.vertices(), surf.verticesSize(), (const void*)function.data());
        wnd.draw(chart);
    } while(!wnd.close());

    return 0;
}

