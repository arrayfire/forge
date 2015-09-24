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

const float FRANGE_START = 0.f;
const float FRANGE_END = 2.f * 3.1415926f;

using namespace std;
void map_range_to_vec_vbo(float range_start, float range_end, float dx, std::vector<float> &vec, float (*map) (float)){
    if(range_start > range_end && dx > 0) return;
    for(float i=range_start; i < range_end; i+=dx){
        vec.push_back(i);
        vec.push_back((*map)(i));
    }
}

int main(void){
    std::vector<float> function;
    map_range_to_vec_vbo(FRANGE_START, FRANGE_END, 0.1f, function, &sinf);

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "Plotting Demo");
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
    fg::Plot3 plt(10, 10, fg::FG_FLOAT);                              //create a default plot

    /*
     * Set plot colors
     */
    plt.setColor(fg::FG_YELLOW);

    /*
     * Set draw limits for plots
     */
    plt.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f, 1.1f, -1.1f);

    /* copy your data into the pixel buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copy(plt, &function[0]);

    do {
        // draw window and poll for events last
        wnd.draw(plt);
    } while(!wnd.close());

    return 0;
}

