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

static const float ZMIN = 0.1f;
static const float ZMAX = 10.f;

const float DX = 0.005;
const size_t ZSIZE = (ZMAX-ZMIN)/DX+1;

using namespace std;
void gen_curve(float t, float dx, std::vector<float> &vec ) {
    vec.clear();
    for(float z=ZMIN; z < ZMAX; z+=dx){
        vec.push_back(cos(z*t+t)/z);
        vec.push_back(sin(z*t+t)/z);
        vec.push_back(z+0.1*sin(t));
    }
}

int main(void){
    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "Plot3d Demo");
    wnd.makeCurrent();
    /* create an font object and load necessary font
     * and later pass it on to window object so that
     * it can be used for rendering text */
    fg::Font fnt;
#ifdef OS_WIN
    fnt.loadSystemFont("Calibri", 32);
#else
    fnt.loadSystemFont("Vera", 32);
#endif
    wnd.setFont(&fnt);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    fg::Plot3 plot3(ZSIZE, fg::f32);

    /*
     * Set draw limits for plots
     */
    plot3.setAxesLimits(1.1f, -1.1f, 1.1f, -1.1f, 10.f, 0.f);

    /*
    * Set axis titles
    */
    plot3.setAxesTitles("x-axis", "y-axis", "z-axis");

    //generate a surface
    std::vector<float> function;
    static float t=0;
    gen_curve(t, DX, function);
    /* copy your data into the pixel buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copy(plot3, &function[0]);

    do {
        t+=0.01;
        gen_curve(t, DX, function);
        copy(plot3, &function[0]);
        // draw window and poll for events last
        wnd.draw(plot3);
    } while(!wnd.close());

    return 0;
}
