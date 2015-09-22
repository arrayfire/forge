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
const unsigned WIN_ROWS = 2;
const unsigned WIN_COLS = 2;

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

    /*
     * Split the window into grid regions
     */
    wnd.grid(WIN_ROWS, WIN_COLS);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    fg::Plot plt0(function.size()/2, fg::FG_FLOAT);                              //create a default plot
    fg::Plot plt1(function.size()/2, fg::FG_FLOAT, fg::FG_LINE, fg::FG_NONE);       //or specify a specific plot type
    fg::Plot plt2(function.size()/2, fg::FG_FLOAT, fg::FG_LINE, fg::FG_TRIANGLE);   //last parameter specifies marker shape
    fg::Plot plt3(function.size()/2, fg::FG_FLOAT, fg::FG_SCATTER, fg::FG_POINT);

    /*
     * Set plot colors
     */
    plt0.setColor(fg::FG_YELLOW);
    plt1.setColor(fg::FG_BLUE);
    plt2.setColor(fg::FG_WHITE);                                                  //use a forge predefined color
    plt3.setColor((fg::Color) 0xABFF01FF);                                        //or any hex-valued color

    /*
     * Set draw limits for plots
     */
    plt0.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);
    plt1.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);
    plt2.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);
    plt3.setAxesLimits(FRANGE_END, FRANGE_START, 1.1f, -1.1f);

    /* copy your data into the pixel buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copy(plt0, &function[0]);
    copy(plt1, &function[0]);
    copy(plt2, &function[0]);
    copy(plt3, &function[0]);

    do {
        wnd.draw(0, 0, plt0,  NULL                );
        wnd.draw(0, 1, plt1, "sinf_line_blue"     );
        wnd.draw(1, 1, plt2, "sinf_line_triangle" );
        wnd.draw(1, 0, plt3, "sinf_scatter_point" );
        // draw window and poll for events last
        wnd.draw();
    } while(!wnd.close());

    return 0;
}

