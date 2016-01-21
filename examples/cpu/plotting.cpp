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

const float FRANGE_START = 0.f;
const float FRANGE_END = 2.f * 3.1415926f;

using namespace std;
void map_range_to_vec_vbo(float range_start, float range_end, float dx,
                          std::vector<float> &vec,
                          float (*map) (float))
{
    if(range_start > range_end && dx > 0) return;
    for(float i=range_start; i < range_end; i+=dx){
        vec.push_back(i);
        vec.push_back((*map)(i));
    }
}

int main(void)
{
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

    fg::Chart chart(fg::FG_2D);
    chart.setAxesLimits(FRANGE_START, FRANGE_END, -1.1f, 1.1f);

    /* Create several plot objects which creates the necessary
     * vertex buffer objects to hold the different plot types
     */
    fg::Plot plt0 = chart.plot(sinData.size()/2, fg::f32); //create a default plot
    fg::Plot plt1 = chart.plot(cosData.size()/2, fg::f32, fg::FG_LINE, fg::FG_NONE); //or specify a specific plot type
    fg::Plot plt2 = chart.plot(tanData.size()/2, fg::f32, fg::FG_LINE, fg::FG_TRIANGLE); //last parameter specifies marker shape
    fg::Plot plt3 = chart.plot(logData.size()/2, fg::f32, fg::FG_SCATTER, fg::FG_POINT);

    /*
     * Set plot colors
     */
    plt0.setColor(fg::FG_YELLOW);
    plt1.setColor(fg::FG_BLUE);
    plt2.setColor(fg::FG_WHITE);            //use a forge predefined color
    plt3.setColor((fg::Color) 0xABFF01FF);  //or any hex-valued color
    /*
     * Set plot legends
     */
    plt0.setLegend("Sine");
    plt1.setLegend("Cosine");
    plt2.setLegend("Tangent");
    plt3.setLegend("Log base 10");


    /* copy your data into the pixel buffer object exposed by
     * fg::Plot class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    copy(plt0, &sinData[0]);
    copy(plt1, &cosData[0]);
    copy(plt2, &tanData[0]);
    copy(plt3, &logData[0]);

    do {
        wnd.draw(chart);
    } while(!wnd.close());

    return 0;
}

