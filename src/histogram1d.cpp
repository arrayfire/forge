/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/histogram1d.h>
#include <fg/window.h>
#include <common.hpp>
#include <err_common.hpp>
#include <cstdio>

#include <math.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

// Struct for ticks and borders
struct point {
    GLfloat x;
    GLfloat y;
};

const char *gHistogramVertexShaderSrc =
"attribute vec2 coord2d;            "
"uniform mat4 transform;            "
"void main(void) {                  "
"   gl_Position = transform * vec4(coord2d.xy, 0, 1);"
"}";

const char *gHistogramFragmentShaderSrc =
"uniform vec4 color;          "
"void main(void) {            "
"   gl_FragColor = color;     "
"}";

static const point border[4] = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };

glm::mat4 viewport_transform_hist(int pVPW, int pVPH, float x, float y, float width, float height, float *pixel_x = 0, float *pixel_y = 0)
{
    float window_width  = pVPW;
    float window_height = pVPH;

    float offset_x = (2.0 * x + (width - window_width))   / window_width;
    float offset_y = (2.0 * y + (height - window_height)) / window_height;

    float scale_x = width  / window_width;
    float scale_y = height / window_height;

    if (pixel_x)
        *pixel_x = 2.0 / width;
    if (pixel_y)
        *pixel_y = 2.0 / height;

    return glm::scale(glm::translate(glm::mat4(1), glm::vec3(offset_x, offset_y, 0)), glm::vec3(scale_x, scale_y, 1));
}

namespace fg
{

Histogram::Histogram(GLenum pDataType)
          : mDataType(pDataType)
{
    MakeContextCurrent();
    // set window here
    mTickSize = 10;
    mMargin = 10;

    glGenBuffers(3, mVBO);
    mVBOsize = 0;

    mProgram = initShaders(gHistogramVertexShaderSrc, gHistogramFragmentShaderSrc);

    mAttrCoord2d = glGetAttribLocation (mProgram, "coord2d");
    mUnfmColor   = glGetUniformLocation(mProgram, "color");
    mUnfmTrans   = glGetUniformLocation(mProgram, "transform");

    // VBO for border
    glBindBuffer(GL_ARRAY_BUFFER, mVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof (border), border, GL_STATIC_DRAW);
}

Histogram::~Histogram()
{
    MakeContextCurrent();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(3, mVBO);
    glDeleteProgram(mProgram);
}

GLuint Histogram::vbo() const { return mVBO[0]; }

size_t Histogram::size() const { return mVBOsize; }

void Histogram::setAxesLimits(double pXmax, double pXmin, double pYmax, double pYmin)
{
    mXMax = pXmax;
    mXMin = pXmin;
    mYMax = pYmax;
    mYMin = pYmin;
}

void Histogram::setVBOSize(size_t pSize)
{
    mVBOsize = pSize;
}

void Histogram::render(int pViewPortWidth, int pViewPortHeight, int size) const
{
    glUseProgram(mProgram);

    // Set viewport. This will clip geometry
    glViewport(mMargin + mTickSize, mMargin + mTickSize, pViewPortWidth - mMargin * 2 - mTickSize, pViewPortHeight - mMargin * 2 - mTickSize);

    // Set scissor rectangle to clip fragments outside of viewport
    glScissor(mMargin + mTickSize, mMargin + mTickSize, pViewPortWidth - mMargin * 2 - mTickSize, pViewPortHeight - mMargin * 2 - mTickSize);

    glEnable(GL_SCISSOR_TEST);

    float graph_scale_x = 1/(mXMax - mXMin);
    float graph_scale_y = 1/(mYMax - mYMin);

    glm::mat4 transform = glm::translate(glm::scale(glm::mat4(1.0f),
                                         glm::vec3(graph_scale_x, graph_scale_y, 1)),
                                         glm::vec3(0, 0, 0));
    glUniformMatrix4fv(mUnfmTrans, 1, GL_FALSE, glm::value_ptr(transform));

    // Set the color to red
    GLfloat red[4] = {1, 0, 0, 1};
    glUniform4fv(mUnfmColor, 1, red);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO[0]);
    ForceCheckGL("Before enable vertex attribute array");

    glEnableVertexAttribArray(mAttrCoord2d);
    glVertexAttribPointer(mAttrCoord2d, 2, mDataType, GL_FALSE, 0, 0);
    ForceCheckGL("Before setting elements");

    printf("number of elemetns=%lu\n", mVBOsize/(2*sizeof(float)));

    glDrawArrays(GL_LINE_STRIP, 0, size);

    // Stop clipping
    glViewport(0, 0, pViewPortWidth, pViewPortHeight);
    glDisable(GL_SCISSOR_TEST);
/*
    // Draw borders
    float pixel_x, pixel_y;
    float margin        = mMargin;
    float ticksize      = mTickSize;
    float window_width  = pViewPortWidth;
    float window_height = pViewPortHeight;

    transform = viewport_transform_hist(window_width, window_height, margin + ticksize, margin + ticksize, window_width - 2 * margin - ticksize, window_height - 2 * margin - ticksize, &pixel_x, &pixel_y);
    glUniformMatrix4fv(mUnfmTrans, 1, GL_FALSE, glm::value_ptr(transform));

    // Set the color to black
    GLfloat black[4] = { 0, 0, 0, 1 };
    glUniform4fv(mUnfmColor, 1, black);

    glBindBuffer(GL_ARRAY_BUFFER, mVBO[1]);
    glVertexAttribPointer(mAttrCoord2d, 2, mDataType, GL_FALSE, 0, 0);

    glDrawArrays(GL_LINE_LOOP, 0, 4);
*/
    glUseProgram(0);
}


void drawHistogram(Window* pWindow, const Histogram& pHandle, const unsigned int nbins, const double minval, const double maxval)
{
    CheckGL("Begin histogram1d");
    MakeContextCurrent(pWindow);

    int wind_width, wind_height;
    glfwGetWindowSize(pWindow->window(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);

    int size = nbins*4;
    printf("size = %d",size);

    glClearColor(0.2, 0.2, 0.2, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    pHandle.render(wind_width, wind_height, size);

    glfwSwapBuffers(pWindow->window());
    glfwPollEvents();
    CheckGL("End plot2d");
}

}
