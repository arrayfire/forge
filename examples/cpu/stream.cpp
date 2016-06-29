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
const int NELEMS    = (MAXIMUM-MINIMUM+1)/STEP;

using namespace std;

void generateColors(std::vector<float>& colors)
{
    static const float AF_BLUE[] = {0.0588f , 0.1137f , 0.2745f , 1.0f};
    static const float AF_ORANGE[] = {0.8588f , 0.6137f , 0.0745f , 1.0f};

    int numElems = NELEMS*NELEMS*NELEMS;
    colors.clear();
    for (int i=0; i<numElems; ++i) {
        if ((i%2) == 0) {
            colors.push_back(AF_ORANGE[0]);
            colors.push_back(AF_ORANGE[1]);
            colors.push_back(AF_ORANGE[2]);
        } else {
            colors.push_back(AF_BLUE[0]);
            colors.push_back(AF_BLUE[1]);
            colors.push_back(AF_BLUE[2]);
        }
    }
}

void moveColors(std::vector<float>& colors)
{
    int end = colors.size()-3;
    float r = colors[end+2];
    float g = colors[end+1];
    float b = colors[end];

    for (int i=0; i<end; i+=3) {
        colors[i] = r;
        colors[i+1] = g;
        colors[i+2] = b;

        r = colors[i+3];
        g = colors[i+4];
        b = colors[i+5];
    }
}

void generatePoints(std::vector<float> &points, std::vector<float> &dirs)
{
    points.clear();

    for (int k=0; k<NELEMS; ++k) {
        float z = MINIMUM + k*STEP;
        for (int j=0; j<NELEMS; ++j) {
            float y = MINIMUM + j*STEP;
            for (int i=0; i<NELEMS; ++i) {
                float x = MINIMUM + i*STEP;
                points.push_back(x);
                points.push_back(y);
                points.push_back(z);
                dirs.push_back(10.0f-x);
                dirs.push_back(10.0f-y);
                dirs.push_back(10.0f-z);
            }
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
    fg::Window wnd(DIMX, DIMY, "3D Vector Field Demo");
    wnd.makeCurrent();

    fg::Chart chart(FG_CHART_3D);
    chart.setAxesLimits(MINIMUM-1.0f, MAXIMUM,
                        MINIMUM-1.0f, MAXIMUM,
                        MINIMUM-1.0f, MAXIMUM);
    chart.setAxesTitles("x-axis", "y-axis", "z-axis");

    int numElems = NELEMS*NELEMS*NELEMS;
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
        moveColors(colors);
        fg::copy(field.colors(), field.colorsSize(), (const void*)colors.data());
        wnd.draw(chart);
    } while(!wnd.close());

    return 0;
}
