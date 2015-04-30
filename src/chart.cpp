/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/chart.h>
#include <fg/exception.h>
#include <common.hpp>
#include <err_common.hpp>

#include <math.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

const char *gChartVertexShaderSrc =
"#version 330\n"
"in vec2 point;\n"
"uniform mat4 transform;\n"
"void main(void) {\n"
"   gl_Position = transform * vec4(point.xy, 0, 1);\n"
"}";

const char *gChartFragmentShaderSrc =
"#version 330\n"
"uniform vec4 color;\n"
"out vec4 outputColor;\n"
"void main(void) {\n"
"   outputColor = color;\n"
"}";

const char *gChartSpriteFragmentShaderSrc =
"#version 330\n"
"uniform bool isYAxis;\n"
"uniform vec4 tick_color;\n"
"out vec4 outputColor;\n"
"void main(void) {\n"
"   bool y_axis = isYAxis && abs(gl_PointCoord.y)>0.2;\n"
"   bool x_axis = !isYAxis && abs(gl_PointCoord.x)>0.2;\n"
"   if(y_axis || x_axis)\n"
"       discard;\n"
"   else\n"
"       outputColor = tick_color;\n"
"}";

namespace fg
{

void Chart::bindBorderProgram() const { glUseProgram(mBorderProgram); }
void Chart::unbindBorderProgram() const { glUseProgram(0); }
GLuint Chart::rectangleVBO() const { return mDecorVBO; }
GLuint Chart::borderProgramPointIndex() const { return mBorderPointIndex; }
GLuint Chart::borderColorIndex() const { return mBorderColorIndex; }
GLuint Chart::borderMatIndex() const { return mBorderMatIndex; }
int Chart::tickSize() const { return mTickSize; }
int Chart::margin() const { return mMargin; }

Chart::Chart()
    : mTickCount(21), mTickSize(10), mMargin(10),
      mDecorVAO(0), mDecorVBO(0), mBorderProgram(0), mSpriteProgram(0)
{
    MakeContextCurrent();

    mBorderProgram = initShaders(gChartVertexShaderSrc, gChartFragmentShaderSrc);
    mSpriteProgram = initShaders(gChartVertexShaderSrc, gChartSpriteFragmentShaderSrc);

    mBorderPointIndex     = glGetAttribLocation (mBorderProgram, "point");
    mBorderColorIndex     = glGetUniformLocation(mBorderProgram, "color");
    mBorderMatIndex       = glGetUniformLocation(mBorderProgram, "transform");
    mSpriteTickcolorIndex = glGetUniformLocation(mSpriteProgram, "tick_color");
    mSpriteMatIndex       = glGetUniformLocation(mSpriteProgram, "transform");
    mSpriteTickaxisIndex  = glGetUniformLocation(mSpriteProgram, "isYAxis");

    static const float border[8] = { -1, -1, 1, -1, 1, 1, -1, 1 };
    static const int nValues = sizeof(border)/sizeof(float);

    std::vector<float> decorData;
    std::copy(border, border+nValues, std::back_inserter(decorData));

    float step = 2.0f/(mTickCount+1);
    /* push tick points for y axis */
    for (int i = 1; i <= mTickCount; i++) {
        /* (-1,-1) to (-1, 1)*/
        decorData.push_back(-1.0f);
        decorData.push_back(-1.0f+i*step);
    }
    /* push tick points for x axis */
    for (int i = 1; i <= mTickCount; i++) {
        /* (-1,-1) to (1, -1)*/
        decorData.push_back(-1.0f+i*step);
        decorData.push_back(-1);
    }
    /* create vbo that has the border and axis data */
    mDecorVBO = createBuffer<float>(decorData.size(), &(decorData.front()), GL_STATIC_DRAW);

    glGenVertexArrays(1, &mDecorVAO);
    glBindVertexArray(mDecorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mDecorVBO);
    glVertexAttribPointer(mBorderPointIndex, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(mBorderPointIndex);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Chart::~Chart()
{
    MakeContextCurrent();
    glDeleteBuffers(1, &mDecorVBO);
    glDeleteVertexArrays(1, &mDecorVAO);
    glDeleteProgram(mBorderProgram);
    glDeleteProgram(mSpriteProgram);
}

double Chart::xmax() const { return mXMax; }
double Chart::xmin() const { return mXMin; }
double Chart::ymax() const { return mYMax; }
double Chart::ymin() const { return mYMin; }

void Chart::setAxesLimits(double pXmax, double pXmin, double pYmax, double pYmin)
{
    mXMax = pXmax;
    mXMin = pXmin;
    mYMax = pYmax;
    mYMin = pYmin;
}

void Chart::renderChart(int pVPW, int pVPH) const
{
    static const GLfloat BLACK[4] = { 0, 0, 0, 1 };
    static const GLfloat BLUE[4] = { 0.0588f, 0.1137f, 0.2745f, 1 };
    int mar_tick = mMargin + mTickSize;
    int mar2_tick = mMargin + mar_tick;
    float w = pVPW - mar2_tick;
    float h = pVPH - mar2_tick;
    float offset_x = (2.0f * mar_tick + (w - pVPW)) / pVPW;
    float offset_y = (2.0f * mar_tick + (h - pVPH)) / pVPH;
    float scale_x = w / pVPW;
    float scale_y = h / pVPH;

    CheckGL("Begin Chart::render");
    /* bind the plotting shader program  */
    glUseProgram(mBorderProgram);

    /* set uniform attributes of shader
     * for drawing the plot borders */
    glm::mat4 trans = glm::scale(glm::translate(glm::mat4(1),
                                                glm::vec3(offset_x, offset_y, 0)),
                                 glm::vec3(scale_x, scale_y, 1));

    glUniformMatrix4fv(mBorderMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    glUniform4fv(mBorderColorIndex, 1, BLUE);

    /* Draw borders */
    glBindVertexArray(mDecorVAO);
    glDrawArrays(GL_LINE_LOOP, 0, 4);
    glBindVertexArray(0);

    /* reset shader program binding */
    glUseProgram(0);

    /* bind the sprite shader program to
     * draw ticks on x and y axes */
    glUseProgram(mSpriteProgram);
    glPointSize(mTickSize);
    glUniform4fv(mSpriteTickcolorIndex, 1, BLACK);
    glUniformMatrix4fv(mSpriteMatIndex, 1, GL_FALSE, glm::value_ptr(trans));

    /* Draw tick marks on y axis */
    glUniform1i(mSpriteTickaxisIndex, 1);
    glBindVertexArray(mDecorVAO);
    glDrawArrays(GL_POINTS, 4, mTickCount);
    glBindVertexArray(0);

    /* Draw tick marks on x axis */
    glUniform1i(mSpriteTickaxisIndex, 0);
    glBindVertexArray(mDecorVAO);
    glDrawArrays(GL_POINTS, 4+mTickCount, mTickCount);
    glBindVertexArray(0);

    /* restoring point size to default */
    glPointSize(1);
    /* reset shader program binding */
    glUseProgram(0);

    CheckGL("End Chart::render");
}

}
