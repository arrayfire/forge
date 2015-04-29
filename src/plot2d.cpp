/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot2d.h>
#include <fg/window.h>
#include <fg/exception.h>
#include <common.hpp>
#include <err_common.hpp>

#include <math.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

const char *gPlotVertexShaderSrc =
"#version 330\n"
"in vec2 coord2d;\n"
"uniform mat4 transform;\n"
"void main(void) {\n"
"   gl_Position = transform * vec4(coord2d.xy, 0, 1);\n"
"}";

const char *gPlotFragmentShaderSrc =
"#version 330\n"
"uniform vec4 color;\n"
"out vec4 outputColor;\n"
"void main(void) {\n"
"   outputColor = color;\n"
"}";

const char *gPlotSpriteFragmentShaderSrc =
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

void Plot::createDecorVAO()
{
    static const float border[8] = { -1, -1, 1, -1, 1, 1, -1, 1 };
    static const int nValues = sizeof(border)/sizeof(float);

    std::vector<float> decorData;
    std::copy(border, border+nValues, std::back_inserter(decorData));

    float step = 2.0f/(mTickCount+1);
    // push tick points
    for (int i = 1; i <= mTickCount; i++) {
        /* (-1,-1) to (-1, 1)*/
        decorData.push_back(-1.0f);
        decorData.push_back(-1.0f+i*step);
    }
    for (int i = 1; i <= mTickCount; i++) {
        /* (-1,-1) to (1, -1)*/
        decorData.push_back(-1.0f+i*step);
        decorData.push_back(-1);
    }

    GLuint mDecorVBO = createBuffer<float>(decorData.size(), &(decorData.front()), GL_STATIC_DRAW);

    glGenVertexArrays(1, &mDecorVAO);
    glBindVertexArray(mDecorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mDecorVBO);
    glVertexAttribPointer(mAttrCoord2d, 2, mDataType, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(mAttrCoord2d);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Plot::Plot(GLuint pNumPoints, GLenum pDataType)
    : mTickCount(21), mTickSize(10), mMargin(10),
      mNumPoints(pNumPoints), mDataType(pDataType), mMainVAO(0), mMainVBOsize(0)
{
    MakeContextCurrent();

    mProgram = initShaders(gPlotVertexShaderSrc, gPlotFragmentShaderSrc);
    mSpriteProgram = initShaders(gPlotVertexShaderSrc, gPlotSpriteFragmentShaderSrc);

    mAttrCoord2d = glGetAttribLocation (mProgram, "coord2d");
    mUnfmColor   = glGetUniformLocation(mProgram, "color");
    mUnfmTrans   = glGetUniformLocation(mProgram, "transform");
    mUnfmTickColor = glGetUniformLocation(mSpriteProgram, "tick_color");
    mUnfmTickTrans = glGetUniformLocation(mSpriteProgram, "transform");
    mUnfmTickAxis = glGetUniformLocation(mSpriteProgram, "isYAxis");

    //  axes and ticks are taken care of by
    //  createDecorVAO member function
    createDecorVAO();

    unsigned total_points = 2*mNumPoints;
    // buffersubdata calls on mMainVBO
    // will only update the points data
    switch(mDataType) {
        case GL_FLOAT:
            mMainVBO = createBuffer<float>(total_points, NULL, GL_DYNAMIC_DRAW);
            mMainVBOsize = total_points*sizeof(float);
            break;
        case GL_INT:
            mMainVBO = createBuffer<int>(total_points, NULL, GL_DYNAMIC_DRAW);
            mMainVBOsize = total_points*sizeof(int);
            break;
        case GL_UNSIGNED_INT:
            mMainVBO = createBuffer<unsigned>(total_points, NULL, GL_DYNAMIC_DRAW);
            mMainVBOsize = total_points*sizeof(unsigned);
            break;
        case GL_UNSIGNED_BYTE:
            mMainVBO = createBuffer<unsigned char>(total_points, NULL, GL_DYNAMIC_DRAW);
            mMainVBOsize = total_points*sizeof(unsigned char);
            break;
        default: fg::TypeError("Plot::Plot", __LINE__, 1, mDataType);
    }

    //create vao
    glGenVertexArrays(1, &mMainVAO);
    glBindVertexArray(mMainVAO);
    // attach plot vertices
    glBindBuffer(GL_ARRAY_BUFFER, mMainVBO);
    glVertexAttribPointer(mAttrCoord2d, 2, mDataType, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(mAttrCoord2d);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Plot::~Plot()
{
    MakeContextCurrent();
    glDeleteBuffers(1, &mMainVBO);
    glDeleteBuffers(1, &mDecorVBO);
    glDeleteVertexArrays(1, &mMainVAO);
    glDeleteVertexArrays(1, &mDecorVAO);
    glDeleteProgram(mProgram);
}

GLuint Plot::vbo() const { return mMainVBO; }

size_t Plot::size() const { return mMainVBOsize; }

double Plot::xmax() const { return mXMax; }
double Plot::xmin() const { return mXMin; }
double Plot::ymax() const { return mYMax; }
double Plot::ymin() const { return mYMin; }

void Plot::setAxesLimits(double pXmax, double pXmin, double pYmax, double pYmin)
{
    mXMax = pXmax;
    mXMin = pXmin;
    mYMax = pYmax;
    mYMin = pYmin;
}

void Plot::setColor(float r, float g, float b)
{
    mLineColor[0] = clampTo01(r);
    mLineColor[1] = clampTo01(g);
    mLineColor[2] = clampTo01(b);
    mLineColor[3] = 1.0f;
}

void Plot::render(int pVPW, int pVPH) const
{
    static const GLfloat black[4] = { 0, 0, 0, 1 };
    int mar_tick = mMargin + mTickSize;
    int mar2_tick = mMargin + mar_tick;
    float w = pVPW - mar2_tick;
    float h = pVPH - mar2_tick;
    float offset_x = (2.0f * mar_tick + (w - pVPW)) / pVPW;
    float offset_y = (2.0f * mar_tick + (h - pVPH)) / pVPH;
    float scale_x = w / pVPW;
    float scale_y = h / pVPH;
    float graph_scale_x = 1/(mXMax - mXMin);
    float graph_scale_y = 1/(mYMax - mYMin);

    CheckGL("Begin Plot::render");
    /* bind the plotting shader program  */
    glUseProgram(mProgram);

    /* Set viewport to plotting area */
    glViewport(mar_tick, mar_tick, pVPW - mar2_tick, pVPH - mar2_tick);

    /* Enavle scissor test to discard anything drawn beyond viewport.
     * Set scissor rectangle to clip fragments outside of viewport */
    glScissor(mar_tick, mar_tick, pVPW - mar2_tick, pVPH - mar2_tick);
    glEnable(GL_SCISSOR_TEST);

    glm::mat4 transform = glm::scale(glm::mat4(1.0f), glm::vec3(graph_scale_x, graph_scale_y, 1));
    glUniformMatrix4fv(mUnfmTrans, 1, GL_FALSE, glm::value_ptr(transform));
    glUniform4fv(mUnfmColor, 1, mLineColor);

    /* render the plot data */
    glBindVertexArray(mMainVAO);
    glDrawArrays(GL_LINE_STRIP, 0, mNumPoints);
    glBindVertexArray(0);

    /* Stop clipping and reset viewport to window dimensions */
    glDisable(GL_SCISSOR_TEST);
    glViewport(0, 0, pVPW, pVPH);

    /* set uniform attributes of shader
     * for drawing the plot borders */
    glm::mat4 trans = glm::scale(glm::translate(glm::mat4(1),
                                                glm::vec3(offset_x, offset_y, 0)),
                                 glm::vec3(scale_x, scale_y, 1));

    glUniformMatrix4fv(mUnfmTrans, 1, GL_FALSE, glm::value_ptr(trans));
    glUniform4fv(mUnfmColor, 1, black);

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
    glUniform4fv(mUnfmTickColor, 1, black);
    glUniformMatrix4fv(mUnfmTickTrans, 1, GL_FALSE, glm::value_ptr(trans));

    /* Draw tick marks on y axis */
    glUniform1i(mUnfmTickAxis, 1);
    glBindVertexArray(mDecorVAO);
    glDrawArrays(GL_POINTS, 4, mTickCount);
    glBindVertexArray(0);

    /* Draw tick marks on x axis */
    glUniform1i(mUnfmTickAxis, 0);
    glBindVertexArray(mDecorVAO);
    glDrawArrays(GL_POINTS, 4+mTickCount, mTickCount);
    glBindVertexArray(0);

    /* restoring point size to default */
    glPointSize(1);
    /* reset shader program binding */
    glUseProgram(0);

    CheckGL("End Plot::render");
}

}
