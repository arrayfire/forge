/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot2d.h>
#include <fg/window.h>
#include <common.hpp>
#include <err_common.hpp>

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

float offset_x = 0;
float scale_x = 1;
float offset_y = 0;
float scale_y = 1;

const char *gPlotVertexShaderSrc =
"attribute vec2 coord2d;            "
"uniform mat4 transform;            "
"void main(void) {                  "
"   gl_Position = transform * vec4(coord2d.xy, 0, 1);"
"}";

const char *gPlotFragmentShaderSrc =
"uniform vec4 color;          "
"void main(void) {            "
"   gl_FragColor = color;     "
"}";

static const point border[4] = { {-1, -1}, {1, -1}, {1, 1}, {-1, 1} };

glm::mat4 viewport_transform(int pVPW, int pVPH, float x, float y, float width, float height, float *pixel_x = 0, float *pixel_y = 0)
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

Plot::Plot(GLenum pDataType)
    : mDataType(pDataType)
{
    MakeContextCurrent();
    // set window here
    mTickSize = 10;
    mMargin = 10;

    glGenBuffers(3, mVBO);
    mVBOSize = 0;

    mProgram = initShaders(gPlotVertexShaderSrc, gPlotFragmentShaderSrc);

    mAttrCoord2d = glGetAttribLocation (mProgram, "coord2d");
    mUnfmColor   = glGetUniformLocation(mProgram, "color");
    mUnfmTrans   = glGetUniformLocation(mProgram, "transform");

    // VBO for border
    glBindBuffer(GL_ARRAY_BUFFER, mVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof (border), border, GL_STATIC_DRAW);
}

Plot::~Plot()
{
    MakeContextCurrent();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(3, mVBO);
    glDeleteProgram(mProgram);
}

GLuint Plot::vbo() const { return mVBO[0]; }

size_t Plot::vboSize() const { return mVBOSize; }

void Plot::setVBOSize(size_t pSize)
{
    mVBOSize = pSize;
}

void Plot::render(int pViewPortWidth, int pViewPortHeight,
                  double pXmax, double pXmin,
                  double pYmax, double pYmin) const
{
    glUseProgram(mProgram);

    // Set viewport. This will clip geometry
    glViewport(mMargin + mTickSize, mMargin + mTickSize, pViewPortWidth - mMargin * 2 - mTickSize, pViewPortHeight - mMargin * 2 - mTickSize);

    // Set scissor rectangle to clip fragments outside of viewport
    glScissor(mMargin + mTickSize, mMargin + mTickSize, pViewPortWidth - mMargin * 2 - mTickSize, pViewPortHeight - mMargin * 2 - mTickSize);

    glEnable(GL_SCISSOR_TEST);

    float graph_scale_x = 1/(pXmax - pXmin);
    float graph_scale_y = 1/(pYmax - pYmin);

    glm::mat4 transform = glm::translate(glm::scale(glm::mat4(1.0f),
                                         glm::vec3(graph_scale_x, graph_scale_y, 1)),
                                         glm::vec3(offset_x, 0, 0));
    glUniformMatrix4fv(mUnfmTrans, 1, GL_FALSE, glm::value_ptr(transform));

    // Set the color to red
    GLfloat red[4] = {1, 0, 0, 1};
    glUniform4fv(mUnfmColor, 1, red);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO[0]);
    ForceCheckGL("Before enable vertex attribute array");

    glEnableVertexAttribArray(mAttrCoord2d);
    glVertexAttribPointer(mAttrCoord2d, 2, mDataType, GL_FALSE, 0, 0);
    ForceCheckGL("Before setting elements");

    size_t elements = 0;
    switch(mDataType) {
        case GL_FLOAT:          elements = mVBOSize / (2 * sizeof(float));     break;
        case GL_INT:            elements = mVBOSize / (2 * sizeof(int  ));     break;
        case GL_UNSIGNED_INT:   elements = mVBOSize / (2 * sizeof(uint ));     break;
        case GL_BYTE:           elements = mVBOSize / (2 * sizeof(char ));     break;
        case GL_UNSIGNED_BYTE:  elements = mVBOSize / (2 * sizeof(uchar));     break;
    }
    glDrawArrays(GL_LINE_STRIP, 0, elements);

    // Stop clipping
    glViewport(0, 0, pViewPortWidth, pViewPortHeight);
    glDisable(GL_SCISSOR_TEST);

    // Draw borders
    float pixel_x, pixel_y;
    float margin        = mMargin;
    float ticksize      = mTickSize;
    float window_width  = pViewPortWidth;
    float window_height = pViewPortHeight;

    transform = viewport_transform(window_width, window_height, margin + ticksize, margin + ticksize, window_width - 2 * margin - ticksize, window_height - 2 * margin - ticksize, &pixel_x, &pixel_y);
    glUniformMatrix4fv(mUnfmTrans, 1, GL_FALSE, glm::value_ptr(transform));

    // Set the color to black
    GLfloat black[4] = { 0, 0, 0, 1 };
    glUniform4fv(mUnfmColor, 1, black);

    glBindBuffer(GL_ARRAY_BUFFER, mVBO[1]);
    glVertexAttribPointer(mAttrCoord2d, 2, mDataType, GL_FALSE, 0, 0);

    glDrawArrays(GL_LINE_LOOP, 0, 4);

    // Draw y tick marks
    point ticks[42];
    float ytickspacing = 0.1 * powf(10, -floor(log10(scale_y)));
    float top = -1.0 / scale_y - offset_y;       // top edge, in graph coordinates
    float bottom = 1.0 / scale_y - offset_y;     // right edge, in graph coordinates
    int top_i = ceil(top / ytickspacing);        // index of top tick, counted from the origin
    int bottom_i = floor(bottom / ytickspacing); // index of bottom tick, counted from the origin
    float y_rem = top_i * ytickspacing - top;    // space between top edge of graph and the first tick

    float y_firsttick = -1.0 + y_rem * scale_y;  // first tick in device coordinates

    int y_nticks = bottom_i - top_i + 1;         // number of y ticks to show
    if (y_nticks > 21)
        y_nticks = 21;    // should not happen

    for (int i = 0; i <= y_nticks; i++) {
        float y = y_firsttick + i * ytickspacing * scale_y;
        float ytickscale = ((i + top_i) % 10) ? 0.5 : 1;

        ticks[i * 2].x = -1;
        ticks[i * 2].y = y;
        ticks[i * 2 + 1].x = -1 - ticksize * ytickscale * pixel_x;
        ticks[i * 2 + 1].y = y;
    }

    glBindBuffer(GL_ARRAY_BUFFER, mVBO[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(ticks), ticks, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(mAttrCoord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_LINES, 0, y_nticks * 2);

    // Draw x tick marks
    ForceCheckGL("Before x ticks");
    float xtickspacing = 0.1 * powf(10, -floor(log10(scale_x)));
    float left = -1.0 / scale_x - offset_x;     // left edge, in graph coordinates
    float right = 1.0 / scale_x - offset_x;     // right edge, in graph coordinates
    int left_i = ceil(left / xtickspacing);     // index of left tick, counted from the origin
    int right_i = floor(right / xtickspacing);  // index of right tick, counted from the origin
    float x_rem = left_i * xtickspacing - left; // space between left edge of graph and the first tick

    float x_firsttick = -1.0 + x_rem * scale_x; // first tick in device coordinates

    int x_nticks = right_i - left_i + 1;        // number of x ticks to show

    if (x_nticks > 21)
        x_nticks = 21;    // should not happen

    for (int i = 0; i < x_nticks; i++) {
        float x = x_firsttick + i * xtickspacing * scale_x;
        float xtickscale = ((i + left_i) % 10) ? 0.5 : 1;

        ticks[i * 2].x = x;
        ticks[i * 2].y = -1;
        ticks[i * 2 + 1].x = x;
        ticks[i * 2 + 1].y = -1 - ticksize * xtickscale * pixel_y;
    }

    glBufferData(GL_ARRAY_BUFFER, sizeof(ticks), ticks, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(mAttrCoord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_LINES, 0, x_nticks * 2);

    glUseProgram(0);
}


void drawPlot(Window* pWindow, const Plot& pHandle,
            const double pXmax, const double pXmin,
            const double pYmax, const double pYmin)
{
    CheckGL("Begin plot2d");
    MakeContextCurrent(pWindow);

    int wind_width, wind_height;
    glfwGetWindowSize(pWindow->window(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);

    glClearColor(0.2, 0.2, 0.2, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    pHandle.render(wind_width, wind_height, pXmax, pXmin, pYmax, pYmin);

    glfwSwapBuffers(pWindow->window());
    glfwPollEvents();
    CheckGL("End plot2d");
}

}
