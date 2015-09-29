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

static int xsize=0;
static int ysize=0;
bool set=false;
using namespace std;
void map_range_to_vec_vbo(float dx, std::vector<float> &vec, float (*map) (float), float t){
    xsize=0;ysize=0;
    vec.clear();
    for(float i=XMIN; i < XMAX; i+=dx){
        xsize++;
        for(float j=YMIN; j < YMAX; j+=dx){
            if(!set)ysize++;
            vec.push_back(i);
            vec.push_back(j);
            vec.push_back(10*i*-abs(j) * cos(i*i*(j+t))+sin(j*(i+t))-1.5);
        }
        set=true;
    }
}

int main(void){
    std::vector<float> function;
    static float t=0;
    map_range_to_vec_vbo(0.01, function, &sinf, t);
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
    fg::Surface surf(xsize, ysize, fg::FG_FLOAT, fg::FG_SURFACE);
    fg::Plot plt(xsize*ysize, fg::FG_FLOAT, fg::FG_LINE_3D, fg::FG_NONE);       //or specify a specific plot type

    /*
     * Set plot colors
     */
    surf.setColor(fg::FG_YELLOW);
    plt.setColor (fg::FG_WHITE );

    /*
     * Set draw limits for plots
     */
    surf.setAxesLimits(1.1f, -1.1f, 1.1f, -1.1f, 10.f, -5.f);
    plt.setAxesLimits (1.1f, -1.1f, 1.1f, -1.1f, 10.f, -5.f);

    /* copy your data into the pixel buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copy(surf, &function[0]);
    copy(plt,  &function[0]);

    do {
        t+=0.01;
        map_range_to_vec_vbo(0.01f, function, &sinf, t);
        copy(surf, &function[0]);
        copy(plt,  &function[0]);
        // draw window and poll for events last
        wnd.draw(surf);
        //wnd.draw(plt);
    } while(!wnd.close());

    return 0;
}

