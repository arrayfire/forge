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

int main(void){
    std::vector<float> function;
    static float t=0;
    gen_surface(t, DX, function);
    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "3d Surface Demo");
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
    int xsize=(XMAX-XMIN)/DX+1;
    int ysize=(YMAX-YMIN)/DX+1;
    fg::Surface surf(xsize, ysize, fg::FG_FLOAT, fg::FG_SURFACE);

    /*
     * Set plot colors
     */
    surf.setColor(fg::FG_YELLOW);

    /*
     * Set draw limits for plots
     */
    surf.setAxesLimits(1.1f, -1.1f, 1.1f, -1.1f, 10.f, -5.f);
    surf.setZAxisTitle("z-axis");
    surf.setYAxisTitle("y-axis");
    surf.setXAxisTitle("x-axis");

    /* copy your data into the pixel buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copy(surf, &function[0]);

    do {
        t+=0.07;
        gen_surface(t, DX, function);
        copy(surf, &function[0]);
        // draw window and poll for events last
        wnd.draw(surf);
    } while(!wnd.close());

    return 0;
}

