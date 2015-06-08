/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <chart.hpp>
#include <font.hpp>

#include <cmath>
#include <sstream>
#include <iomanip>
#include <mutex>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
typedef std::vector<std::string>::const_iterator StringIter;

std::string toString(float pVal, const int n = 2)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(n) << pVal;
    return out.str();
}

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

const std::shared_ptr<internal::font_impl>& getChartFont()
{
    static internal::_Font mChartFont;
    static std::once_flag flag;

    std::call_once(flag, []() {
#if defined(OS_WIN)
        mChartFont.loadSystemFont("Calibri", 32);
#else
        mChartFont.loadSystemFont("Vera", 32);
#endif
    });

    return mChartFont.impl();
}

namespace internal
{

void AbstractChart2D::bindResources(const void* pWnd)
{
    if (mVAOMap.find(pWnd) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(mBorderPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mDecorVBO);
        glVertexAttribPointer(mBorderPointIndex, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWnd] = vao;
    }

    glBindVertexArray(mVAOMap[pWnd]);
}

void AbstractChart2D::unbindResources() const
{
    glBindVertexArray(0);
}

void AbstractChart2D::bindBorderProgram() const
{
    glUseProgram(mBorderProgram);
}

void AbstractChart2D::unbindBorderProgram() const
{
    glUseProgram(0);
}

GLuint AbstractChart2D::rectangleVBO() const
{
    return mDecorVBO;
}

GLuint AbstractChart2D::borderProgramPointIndex() const
{
    return mBorderPointIndex;
}

GLuint AbstractChart2D::borderColorIndex() const
{
    return mBorderColorIndex;
}

GLuint AbstractChart2D::borderMatIndex() const
{
    return mBorderMatIndex;
}

int AbstractChart2D::tickSize() const
{
    return mTickSize;
}

int AbstractChart2D::leftMargin() const
{
    return mLeftMargin;
}

int AbstractChart2D::rightMargin() const
{
    return mRightMargin;
}

int AbstractChart2D::bottomMargin() const
{
    return mBottomMargin;
}

int AbstractChart2D::topMargin() const
{
    return mTopMargin;
}

void AbstractChart2D::setTickCount(int pTickCount)
{
    static const float border[8] = { -1, -1, 1, -1, 1, 1, -1, 1 };
    static const int nValues = sizeof(border)/sizeof(float);

    mTickCount = pTickCount;

    std::vector<float> decorData;
    std::copy(border, border+nValues, std::back_inserter(decorData));

    float step = 2.0f/(mTickCount+1);
    float yRange = mYMax-mYMin;
    float xRange = mXMax-mXMin;
    /* push tick points for y axis */
    for (int i = 1; i <= mTickCount; i++) {
        float temp = -1.0f+i*step;
        /* (-1,-1) to (-1, 1)*/
        decorData.push_back(-1.0f);
        decorData.push_back(temp);
        /* push tick text marker coordinates and the display text */
        mTickTextX.push_back(-1.0f);
        mTickTextY.push_back(temp);
        mYText.push_back(toString(mYMin+i*step*yRange/2));
    }
    /* push tick points for x axis */
    for (int i = 1; i <= mTickCount; i++) {
        float temp = -1.0f+i*step;
        /* (-1,-1) to (1, -1)*/
        decorData.push_back(-1.0f+i*step);
        decorData.push_back(-1);
        /* push tick text marker coordinates and the display text */
        mTickTextX.push_back(temp);
        mTickTextY.push_back(-1.0f);
        mXText.push_back(toString(mXMin+i*step*xRange/2));
    }

    /* check if decoration VBO has been already used(case where
     * tick marks are being changed from default(21) */
    if (mDecorVBO != 0)
        glDeleteBuffers(1, &mDecorVBO);

    /* create vbo that has the border and axis data */
    mDecorVBO = createBuffer<float>(GL_ARRAY_BUFFER, decorData.size(), &(decorData.front()), GL_STATIC_DRAW);
}

AbstractChart2D::AbstractChart2D()
    : mTickCount(8), mTickSize(10),
      mLeftMargin(64), mRightMargin(10), mTopMargin(10), mBottomMargin(32),
      mXMax(1), mXMin(0), mYMax(1), mYMin(0),
      mDecorVBO(0), mBorderProgram(0), mSpriteProgram(0)
{
    /* load font Vera font for chart text
     * renderings, below function actually returns a constant
     * reference to font object used by Chart objects, we are
     * calling it here just to make sure required font glyphs
     * are loaded into the shared Font object */
    getChartFont();

    mBorderProgram = initShaders(gChartVertexShaderSrc, gChartFragmentShaderSrc);
    mSpriteProgram = initShaders(gChartVertexShaderSrc, gChartSpriteFragmentShaderSrc);

    mBorderPointIndex     = glGetAttribLocation (mBorderProgram, "point");
    mBorderColorIndex     = glGetUniformLocation(mBorderProgram, "color");
    mBorderMatIndex       = glGetUniformLocation(mBorderProgram, "transform");
    mSpriteTickcolorIndex = glGetUniformLocation(mSpriteProgram, "tick_color");
    mSpriteMatIndex       = glGetUniformLocation(mSpriteProgram, "transform");
    mSpriteTickaxisIndex  = glGetUniformLocation(mSpriteProgram, "isYAxis");

    /* the following function sets the member variable
     * mTickCount and creates VBO to hold tick marks and the corresponding
     * text markers for those ticks on the axes */
    setTickCount(mTickCount);
}

AbstractChart2D::~AbstractChart2D()
{
    CheckGL("Begin Chart::~Chart");
    glDeleteBuffers(1, &mDecorVBO);
    glDeleteProgram(mBorderProgram);
    glDeleteProgram(mSpriteProgram);
    CheckGL("End Chart::~Chart");
}

void AbstractChart2D::setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin)
{
    mXMax = pXmax;
    mXMin = pXmin;
    mYMax = pYmax;
    mYMin = pYmin;

    /* remove all the tick text markers that were generated
     * by default during the base class(chart) creation and
     * update the text markers based on the new axes limits*/
    mXText.clear();
    mYText.clear();

    float step = 2.0f/(mTickCount+1);
    float yRange = mYMax-mYMin;
    float xRange = mXMax-mXMin;
    /* push tick points for y axis */
    for (int i = 1; i <= mTickCount; i++) {
        float temp = (i*step)/2;
        mYText.push_back(toString(mYMin+temp*yRange));
    }
    /* push tick points for x axis */
    for (int i = 1; i <= mTickCount; i++) {
        float temp = (i*step)/2;
        mXText.push_back(toString(mXMin+temp*xRange));
    }
}

void AbstractChart2D::setXAxisTitle(const char* pTitle)
{
    mXTitle = std::string(pTitle);
}

void AbstractChart2D::setYAxisTitle(const char* pTitle)
{
    mYTitle = std::string(pTitle);
}

float AbstractChart2D::xmax() const { return mXMax; }
float AbstractChart2D::xmin() const { return mXMin; }
float AbstractChart2D::ymax() const { return mYMax; }
float AbstractChart2D::ymin() const { return mYMin; }

void AbstractChart2D::renderChart(const void* pWnd, int pX, int pY, int pVPW, int pVPH)
{
    float w = float(pVPW - (mLeftMargin + mRightMargin + mTickSize));
    float h = float(pVPH - (mTopMargin + mBottomMargin + mTickSize));
    float offset_x = (2.0f * (mLeftMargin+mTickSize) + (w - pVPW)) / pVPW;
    float offset_y = (2.0f * (mBottomMargin+mTickSize) + (h - pVPH)) / pVPH;
    float scale_x = w / pVPW;
    float scale_y = h / pVPH;

    CheckGL("Begin Chart::render");

    bindResources(pWnd);

    /* bind the plotting shader program  */
    glUseProgram(mBorderProgram);

    /* set uniform attributes of shader
     * for drawing the plot borders */
    glm::mat4 trans = glm::scale(glm::translate(glm::mat4(1),
                                                glm::vec3(offset_x, offset_y, 0)),
                                 glm::vec3(scale_x, scale_y, 1));
    glUniformMatrix4fv(mBorderMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    glUniform4fv(mBorderColorIndex, 1, WHITE);

    /* Draw borders */
    glDrawArrays(GL_LINE_LOOP, 0, 4);

    /* reset shader program binding */
    glUseProgram(0);

    /* bind the sprite shader program to
     * draw ticks on x and y axes */
    glPointSize((GLfloat)mTickSize);

    glUseProgram(mSpriteProgram);
    glUniform4fv(mSpriteTickcolorIndex, 1, WHITE);
    glUniformMatrix4fv(mSpriteMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    /* Draw tick marks on y axis */
    glUniform1i(mSpriteTickaxisIndex, 1);
    glDrawArrays(GL_POINTS, 4, mTickCount);
    /* Draw tick marks on x axis */
    glUniform1i(mSpriteTickaxisIndex, 0);
    glDrawArrays(GL_POINTS, 4+mTickCount, mTickCount);

    glUseProgram(0);
    glPointSize(1);
    unbindResources();

    auto &fonter = getChartFont();
    fonter->setOthro2D(int(w), int(h));

    float pos[2];
    /* render tick marker texts for y axis */
    for (StringIter it = mYText.begin(); it!=mYText.end(); ++it) {
        int idx = int(it - mYText.begin());
        glm::vec4 res = trans * glm::vec4(mTickTextX[idx], mTickTextY[idx], 0, 1);
        pos[0] = w*(res.x+1.0f)/2.0f;
        pos[1] = h*(res.y+1.0f)/2.0f;
        pos[0] -= (pVPW-w)*0.50f;
        fonter->render(pWnd, pos, WHITE, it->c_str(), 15);
    }
    /* render tick marker texts for x axis */
    for (StringIter it = mXText.begin(); it!=mXText.end(); ++it) {
        int idx = int(it - mXText.begin());
        /* mTickCount offset is needed while reading point coordinates for
         * x axis tick marks */
        glm::vec4 res = trans * glm::vec4(mTickTextX[idx+mTickCount], mTickTextY[idx+mTickCount], 0, 1);
        pos[0] = w*(res.x+1.0f)/2.0f;
        pos[1] = h*(res.y+1.0f)/2.0f;
        pos[1] -= (pVPH-h)*0.32f;
        fonter->render(pWnd, pos, WHITE, it->c_str(), 15);
    }
    /* render chart axes titles */
    if (!mYTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f);
        pos[0] = w*(res.x+1.0f)/2.0f;
        pos[1] = h*(res.y+1.0f)/2.0f;
        pos[0] -= (pVPW-w)*0.70f;
        fonter->render(pWnd, pos, WHITE, mYTitle.c_str(), 15, true);
    }
    if (!mXTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(0.0f, -1.0f, 0.0f, 1.0f);
        pos[0] = w*(res.x+1.0f)/2.0f;
        pos[1] = h*(res.y+1.0f)/2.0f;
        pos[1] -= (pVPH-h)*0.70f;
        fonter->render(pWnd, pos, WHITE, mXTitle.c_str(), 15);
    }

    CheckGL("End Chart::render");
}

}
