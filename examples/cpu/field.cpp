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

const unsigned DIMX = 640;
const unsigned DIMY = 480;
const float PI = 3.14159265359;
const float MINIMUM = 1.0f;
const float MAXIMUM = 20.f;
const float STEP    = 2.0f;

using namespace std;

inline float randn()
{
   return ((float) rand() / (RAND_MAX));
}

void generateColors(std::vector<float>& colors)
{
    int elems = (MAXIMUM-MINIMUM)/STEP;
    int numElems = elems*elems;
    colors.clear();
    for (int i=0; i<numElems; ++i) {
        colors.push_back(randn());
        colors.push_back(randn());
        colors.push_back(randn());
    }
}

void generatePoints(std::vector<float> &points, std::vector<float> &dirs)
{
    int numElems = (MAXIMUM-MINIMUM)/STEP;
    points.clear();

    for (int j=0; j<numElems; ++j) {
        float y = MINIMUM + j*STEP;
        for (int i=0; i<numElems; ++i) {
            float x = MINIMUM + i*STEP;
            points.push_back(x);
            points.push_back(y);
            dirs.push_back(sin(2*PI*x/10.f));
            dirs.push_back(sin(2*PI*y/10.f));
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
    fg::Window wnd(DIMX, DIMY, "Vector Field Demo");
    wnd.makeCurrent();

    fg::Chart chart(FG_CHART_2D);
    chart.setAxesLimits(1.f, 20.f, 1.f, 20.f);
    chart.setAxesTitles("x-axis", "y-axis");

    int elems = (MAXIMUM-MINIMUM)/STEP;
    int numElems = elems*elems;
    fg::VectorField field = chart.vectorField(numElems, fg::f32);
    field.setColor(0.f, 1.f, 0.f, 1.f);

    std::vector<float> points;
    std::vector<float> colors;
    std::vector<float> dirs;
    generatePoints(points, dirs);
    generateColors(colors);

    fg::copy(field.vertices(), field.verticesSize(), (const void*)points.data());
    fg::copy(field.colors(), field.colorsSize(), (const void*)colors.data());
    fg::copy(field.directions(), field.directionsSize(), (const void*)dirs.data());

    do {
        generateColors(colors);
        fg::copy(field.colors(), field.colorsSize(), (const void*)colors.data());
        wnd.draw(chart);
    } while(!wnd.close());

    return 0;
}
