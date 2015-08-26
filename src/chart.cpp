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
#include <mutex>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
typedef std::vector<std::string>::const_iterator StringIter;

static const int CHART2D_FONT_SIZE = 15;

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

void AbstractChart2D::bindResources(int pWindowId)
{
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
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
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
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

    float step = 2.0f/(mTickCount);
    /* push tick points for y axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    int ticksLeft = mTickCount/2;
    decorData.push_back(-1.0f);
    decorData.push_back(0.0f);
    mTickTextX.push_back(-1.0f);
    mTickTextY.push_back(0.0f);
    mYText.push_back(toString(0));

    for(int i=1; i<=ticksLeft; ++i) {
        /* [-1, 0) to [-1, -1] */
        float neg = i*-step;
        decorData.push_back(-1.0f);
        decorData.push_back(neg);
        /* puch tick marks */
        mTickTextX.push_back(-1.0f);
        mTickTextY.push_back(neg);
        /* push tick text label */
        mYText.push_back(toString(neg));

        /* [-1, 0) to [-1, 1] */
        float pos = i*step;
        decorData.push_back(-1.0f);
        decorData.push_back(pos);
        /* puch tick marks */
        mTickTextX.push_back(-1.0f);
        mTickTextY.push_back(pos);
        /* push tick text label */
        mYText.push_back(toString(pos));
    }

    /* push tick points for x axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    decorData.push_back(0.0f);
    decorData.push_back(-1.0f);
    mTickTextX.push_back(0.0f);
    mTickTextY.push_back(-1.0f);
    mXText.push_back(toString(0));

    for(int i=1; i<=ticksLeft; ++i) {
        /* (0, -1] to [-1, -1] */
        float neg = i*-step;
        decorData.push_back(neg);
        decorData.push_back(-1.0f);
        /* puch tick marks */
        mTickTextX.push_back(neg);
        mTickTextY.push_back(-1.0f);
        /* push tick text label */
        mXText.push_back(toString(neg));

        /* (0, -1] to [1, -1] */
        float pos = i*step;
        decorData.push_back(pos);
        decorData.push_back(-1.0f);
        /* puch tick marks */
        mTickTextX.push_back(pos);
        mTickTextY.push_back(-1.0f);
        /* push tick text label */
        mXText.push_back(toString(pos));
    }

    /* check if decoration VBO has been already used(case where
     * tick marks are being changed from default(21) */
    if (mDecorVBO != 0)
        glDeleteBuffers(1, &mDecorVBO);

    /* create vbo that has the border and axis data */
    mDecorVBO = createBuffer<float>(GL_ARRAY_BUFFER, decorData.size(), &(decorData.front()), GL_STATIC_DRAW);
}

AbstractChart2D::AbstractChart2D()
    : mTickCount(9), mTickSize(10),
      mLeftMargin(68), mRightMargin(8), mTopMargin(8), mBottomMargin(32),
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

    float step = 2.0f/(mTickCount);
    float xmid   = (mXMax+mXMin)/2.0f;
    float ymid   = (mYMax+mYMin)/2.0f;
    int ticksLeft = mTickCount/2;
    /* push tick points for y axis */
    mYText.push_back(toString(ymid));
    for (int i = 1; i <= ticksLeft; i++) {
        mYText.push_back(toString(ymid + i*-step));
        mYText.push_back(toString(ymid + i*step));
    }
    /* push tick points for x axis */
    mXText.push_back(toString(xmid));
    for (int i = 1; i <= ticksLeft; i++) {
        mXText.push_back(toString(xmid + i*-step));
        mXText.push_back(toString(xmid + i*step));
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

void AbstractChart2D::renderChart(int pWindowId, int pX, int pY, int pVPW, int pVPH)
{
    float w = float(pVPW - (mLeftMargin + mRightMargin + mTickSize));
    float h = float(pVPH - (mTopMargin + mBottomMargin + mTickSize));
    float offset_x = (2.0f * (leftMargin()+tickSize()) + (w - pVPW)) / pVPW;
    float offset_y = (2.0f * (bottomMargin()+tickSize()) + (h - pVPH)) / pVPH;
    float scale_x = w / pVPW;
    float scale_y = h / pVPH;

    CheckGL("Begin Chart::render");

    bindResources(pWindowId);

    /* bind the plotting shader program  */
    glUseProgram(mBorderProgram);

    /* set uniform attributes of shader
     * for drawing the plot borders */
    glm::mat4 trans = glm::translate(glm::scale(glm::mat4(1),
                                                glm::vec3(scale_x, scale_y, 1)),
                                     glm::vec3(offset_x, offset_y, 0));
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
        /* convert text position from [-1,1] range to
         * [0, 1) range and then offset horizontally
         * to compensate for margins and ticksize */
        pos[0] = w*(res.x+1.0f)/2.0f;
        pos[1] = h*(res.y+1.0f)/2.0f;
        /* offset horizontally based on text size to align
         * text center with tick mark position */
        pos[0] -= ((CHART2D_FONT_SIZE*it->length()/2.0f)+(mTickSize * (w/pVPW)));
        fonter->render(pWindowId, pos, WHITE, it->c_str(), CHART2D_FONT_SIZE);
    }
    /* render tick marker texts for x axis */
    for (StringIter it = mXText.begin(); it!=mXText.end(); ++it) {
        int idx = int(it - mXText.begin());
        /* mTickCount offset is needed while reading point coordinates for
         * x axis tick marks */
        glm::vec4 res = trans * glm::vec4(mTickTextX[idx+mTickCount], mTickTextY[idx+mTickCount], 0, 1);
        /* convert text position from [-1,1] range to
         * [0, 1) range and then offset vertically
         * to compensate for margins and ticksize */
        pos[0] = w*(res.x+1.0f)/2.0f;
        pos[1] = h*(res.y+1.0f)/2.0f;
        pos[1] -= ((CHART2D_FONT_SIZE*h/pVPH)+(mTickSize * (w/pVPW)));
        /* offset horizontally based on text size to align
         * text center with tick mark position */
        pos[0] -= (CHART2D_FONT_SIZE*(it->length()-2)/2.0f);
        fonter->render(pWindowId, pos, WHITE, it->c_str(), CHART2D_FONT_SIZE);
    }
    /* render chart axes titles */
    if (!mYTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f);
        pos[0] = w*(res.x+1.0f)/2.0f;
        pos[1] = h*(res.y+1.0f)/2.0f;
        pos[0] += (mTickSize * (w/pVPW));
        fonter->render(pWindowId, pos, WHITE, mYTitle.c_str(), CHART2D_FONT_SIZE, true);
    }
    if (!mXTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(0.0f, -1.0f, 0.0f, 1.0f);
        pos[0] = w*(res.x+1.0f)/2.0f;
        pos[1] = h*(res.y+1.0f)/2.0f;
        pos[1] += (mTickSize * (h/pVPH));
        fonter->render(pWindowId, pos, WHITE, mXTitle.c_str(), CHART2D_FONT_SIZE);
    }

    CheckGL("End Chart::render");
}

}
