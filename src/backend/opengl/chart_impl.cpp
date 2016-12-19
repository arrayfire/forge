/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <font.hpp>

#include <common.hpp>
#include <err_opengl.hpp>
#include <font_impl.hpp>
#include <chart_impl.hpp>
#include <image_impl.hpp>
#include <histogram_impl.hpp>
#include <plot_impl.hpp>
#include <surface_impl.hpp>
#include <window_impl.hpp>
#include <shader_headers/chart_vs.hpp>
#include <shader_headers/chart_fs.hpp>
#include <shader_headers/tick_fs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <mutex>
#include <sstream>
#include <algorithm>

using namespace gl;
using namespace std;

typedef std::vector<std::string>::const_iterator StringIter;

static const int CHART2D_FONT_SIZE = 16;

const std::shared_ptr<forge::opengl::font_impl>& getChartFont()
{
    static forge::common::Font gChartFont;
    static std::once_flag flag;

    std::call_once(flag, []() {
#if defined(OS_WIN)
        gChartFont.loadSystemFont("Calibri");
#else
        gChartFont.loadSystemFont("Vera");
#endif
    });

    return gChartFont.impl();
}

template<typename T>
void pushPoint(vector<T> &points, T x, T y)
{
    points.push_back(x);
    points.push_back(y);
}

template<typename T>
void pushPoint(vector<T> &points, T x, T y, T z)
{
    points.push_back(x);
    points.push_back(y);
    points.push_back(z);
}

namespace forge
{
namespace opengl
{

/********************* BEGIN-AbstractChart *********************/

void AbstractChart::renderTickLabels(
        const int pWindowId, const uint pW, const uint pH,
        const std::vector<std::string> &pTexts,
        const glm::mat4 &pTransformation, const int pCoordsOffset,
        const bool pUseZoffset) const
{
    auto &fonter = getChartFont();
    fonter->setOthro2D(int(pW), int(pH));

    float pos[2];
    for (StringIter it = pTexts.begin(); it!=pTexts.end(); ++it) {
        int idx = int(it - pTexts.begin());
        glm::vec4 p = glm::vec4(mTickTextX[idx+pCoordsOffset],
                                mTickTextY[idx+pCoordsOffset],
                                (pUseZoffset ? mTickTextZ[idx+pCoordsOffset] : 0), 1);
        glm::vec4 res = pTransformation * p;

        /* convert text position from [-1,1] range to
         * [0, 1) range and then offset horizontally
         * to compensate for margins and ticksize */
        pos[0] = pW * (res.x/res.w+1.0f)/2.0f;
        pos[1] = pH * (res.y/res.w+1.0f)/2.0f;

        const float strHalfLen = it->length() / 2.0f;
        /* offset based on text size to align
         * text center with tick mark position */
        if(pCoordsOffset < mTickCount) {
            // offset for y axis labels
            pos[0] -= (CHART2D_FONT_SIZE*strHalfLen+mTickSize);
            pos[1] -= (CHART2D_FONT_SIZE*.36);
        } else if(pCoordsOffset >= mTickCount && pCoordsOffset < 2*mTickCount) {
            // offset for x axis labels
            pos[0] -= (CHART2D_FONT_SIZE*strHalfLen/2.0f);
            pos[1] -= (CHART2D_FONT_SIZE*1.32);
        } else {
            // offsets for 3d chart axes ticks
            pos[0] -= (CHART2D_FONT_SIZE*strHalfLen);
            pos[1] -= (CHART2D_FONT_SIZE);
        }
        fonter->render(pWindowId, pos, BLACK, it->c_str(), CHART2D_FONT_SIZE);
    }
}

AbstractChart::AbstractChart(const int pLeftMargin, const int pRightMargin,
                             const int pTopMargin, const int pBottomMargin)
    : mTickCount(9), mTickSize(10),
      mDefaultLeftMargin(pLeftMargin), mLeftMargin(pLeftMargin), mRightMargin(pRightMargin),
      mTopMargin(pTopMargin), mBottomMargin(pBottomMargin),
      mXMax(0), mXMin(0), mYMax(0), mYMin(0), mZMax(0), mZMin(0),
      mXTitle("X-Axis"), mYTitle("Y-Axis"), mZTitle("Z-Axis"), mDecorVBO(-1),
      mBorderProgram(glsl::chart_vs.c_str(), glsl::chart_fs.c_str()),
      mSpriteProgram(glsl::chart_vs.c_str(), glsl::tick_fs.c_str()),
      mBorderAttribPointIndex(-1), mBorderUniformColorIndex(-1),
      mBorderUniformMatIndex(-1), mSpriteUniformMatIndex(-1),
      mSpriteUniformTickcolorIndex(-1), mSpriteUniformTickaxisIndex(-1),
      mLegendX(0.4f), mLegendY(0.9f)
{
    CheckGL("Begin AbstractChart::AbstractChart");
    /* load font Vera font for chart text
     * renderings, below function actually returns a constant
     * reference to font object used by Chart objects, we are
     * calling it here just to make sure required font glyphs
     * are loaded into the shared Font object */
    getChartFont();

    mBorderAttribPointIndex      = mBorderProgram.getAttributeLocation("point");
    mBorderUniformColorIndex     = mBorderProgram.getUniformLocation("color");
    mBorderUniformMatIndex       = mBorderProgram.getUniformLocation("transform");

    mSpriteUniformTickcolorIndex = mSpriteProgram.getUniformLocation("tick_color");
    mSpriteUniformMatIndex       = mSpriteProgram.getUniformLocation("transform");
    mSpriteUniformTickaxisIndex  = mSpriteProgram.getUniformLocation("isYAxis");

    CheckGL("End AbstractChart::AbstractChart");
}

AbstractChart::~AbstractChart()
{
    CheckGL("Begin AbstractChart::~AbstractChart");
    for (auto it = mVAOMap.begin(); it!=mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mDecorVBO);
    CheckGL("End AbstractChart::~AbstractChart");
}

void AbstractChart::setAxesLimits(const float pXmin, const float pXmax,
                                  const float pYmin, const float pYmax,
                                  const float pZmin, const float pZmax)
{
    mXMin = pXmin; mXMax = pXmax;
    mYMin = pYmin; mYMax = pYmax;
    mZMin = pZmin; mZMax = pZmax;

    /*
     * Once the axes ranges are known, we can generate
     * tick labels. The following functions is a pure
     * virtual function and has to be implemented by the
     * derived class
     */
    generateTickLabels();
}

void AbstractChart::getAxesLimits(float* pXmin, float* pXmax,
                                  float* pYmin, float* pYmax,
                                  float* pZmin, float* pZmax)
{
    *pXmin = mXMin; *pXmax = mXMax;
    *pYmin = mYMin; *pYmax = mYMax;
    *pZmin = mZMin; *pZmax = mZMax;
}

void AbstractChart::setAxesTitles(const char* pXTitle,
                                  const char* pYTitle,
                                  const char* pZTitle)
{
    mXTitle = std::string(pXTitle);
    mYTitle = std::string(pYTitle);
    if (pZTitle)
        mZTitle = std::string(pZTitle);
}

void AbstractChart::setLegendPosition(const float pX, const float pY)
{
    mLegendX = pX;
    mLegendY = pY;
}

float AbstractChart::xmax() const { return mXMax; }
float AbstractChart::xmin() const { return mXMin; }
float AbstractChart::ymax() const { return mYMax; }
float AbstractChart::ymin() const { return mYMin; }
float AbstractChart::zmax() const { return mZMax; }
float AbstractChart::zmin() const { return mZMin; }

void AbstractChart::addRenderable(const std::shared_ptr<AbstractRenderable> pRenderable)
{
    mRenderables.emplace_back(pRenderable);
}

/********************* END-AbstractChart *********************/



/********************* BEGIN-chart2d_impl *********************/

void chart2d_impl::bindResources(const int pWindowId)
{
    CheckGL("Begin chart2d_impl::bindResources");
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(mBorderAttribPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mDecorVBO);
        glVertexAttribPointer(mBorderAttribPointIndex, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }
    glBindVertexArray(mVAOMap[pWindowId]);
    CheckGL("End chart2d_impl::bindResources");
}

void chart2d_impl::unbindResources() const
{
    glBindVertexArray(0);
}

void chart2d_impl::pushTicktextCoords(const float pX, const float pY, const float pZ)
{
    mTickTextX.push_back(pX);
    mTickTextY.push_back(pY);
}

void chart2d_impl::generateChartData()
{
    CheckGL("Begin chart2d_impl::generateChartData");
    static const float border[8] = { -1, -1, 1, -1, 1, 1, -1, 1 };
    static const int nValues = sizeof(border)/sizeof(float);

    std::vector<float> decorData;
    std::copy(border, border+nValues, std::back_inserter(decorData));

    float step = getTickStepSize(-1, 1);
    int ticksLeft = getNumTicksC2E();

    /* push tick points for y axis:
     * push (0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, -1.0f, 0.0f);
    pushTicktextCoords(-1.0f, 0.0f);
    mYText.push_back(toString(0));

    for(int i=1; i<=ticksLeft; ++i) {
        /* [-1, 0) to [-1, -1] */
        float neg = i*-step;
        pushPoint(decorData, -1.0f, neg);
        /* puch tick marks */
        pushTicktextCoords(-1.0f, neg);
        /* push tick text label */
        mYText.push_back(toString(neg));

        /* [-1, 0) to [-1, 1] */
        float pos = i*step;
        pushPoint(decorData, -1.0f, pos);
        /* puch tick marks */
        pushTicktextCoords(-1.0f, pos);
        /* push tick text label */
        mYText.push_back(toString(pos));
    }

    /* push tick points for x axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, 0.0f, -1.0f);
    pushTicktextCoords(0.0f, -1.0f);
    mXText.push_back(toString(0));

    for(int i=1; i<=ticksLeft; ++i) {
        /* (0, -1] to [-1, -1] */
        float neg = i*-step;
        pushPoint(decorData, neg, -1.0f);
        pushTicktextCoords(neg, -1.0f);
        mXText.push_back(toString(neg));

        /* (0, -1] to [1, -1] */
        float pos = i*step;
        pushPoint(decorData, pos, -1.0f);
        pushTicktextCoords(pos, -1.0f);
        mXText.push_back(toString(pos));
    }

    /* push grid lines */
    pushPoint(decorData, -1.0f, 0.0f);
    pushPoint(decorData,  1.0f, 0.0f);
    pushPoint(decorData,  0.0f,-1.0f);
    pushPoint(decorData,  0.0f, 1.0f);
    for (int i=1; i<=ticksLeft; ++i) {
        float delta = i*step;
        pushPoint(decorData, -1.0f,-delta);
        pushPoint(decorData,  1.0f,-delta);
        pushPoint(decorData, -1.0f, delta);
        pushPoint(decorData,  1.0f, delta);
        pushPoint(decorData,-delta, -1.0f);
        pushPoint(decorData,-delta,  1.0f);
        pushPoint(decorData, delta, -1.0f);
        pushPoint(decorData, delta,  1.0f);
    }

    /* check if decoration VBO has been already used(case where
     * tick marks are being changed from default(21) */
    if (mDecorVBO != 0)
        glDeleteBuffers(1, &mDecorVBO);

    /* create vbo that has the border and axis data */
    mDecorVBO = createBuffer<float>(GL_ARRAY_BUFFER, decorData.size(), &(decorData.front()), GL_STATIC_DRAW);
    CheckGL("End chart2d_impl::generateChartData");
}

void chart2d_impl::generateTickLabels()
{
    /* remove all the tick text markers that were generated
     * by default during the base class(chart) creation and
     * update the text markers based on the new axes limits
     * */
    mXText.clear();
    mYText.clear();
    mZText.clear();

    float xstep = getTickStepSize(mXMin, mXMax);
    float ystep = getTickStepSize(mYMin, mYMax);
    float xmid  = (mXMax+mXMin)/2.0f;
    float ymid  = (mYMax+mYMin)/2.0f;

    int ticksLeft = getNumTicksC2E();

    /* push tick points for y axis */
    mYText.push_back(toString(ymid));
    size_t maxYLabelWidth = 0;
    for (int i = 1; i <= ticksLeft; i++) {
        std::string temp = toString(ymid + i*-ystep);
        mYText.push_back(temp);
        maxYLabelWidth = std::max(maxYLabelWidth, temp.length());

        temp = toString(ymid + i*ystep);
        mYText.push_back(temp);
        maxYLabelWidth = std::max(maxYLabelWidth, temp.length());
    }

    mLeftMargin = std::max((int)maxYLabelWidth, mDefaultLeftMargin)+2*CHART2D_FONT_SIZE;

    /* push tick points for x axis */
    mXText.push_back(toString(xmid));
    for (int i = 1; i <= ticksLeft; i++) {
        mXText.push_back(toString(xmid + i*-xstep));
        mXText.push_back(toString(xmid + i*xstep));
    }
}

chart2d_impl::chart2d_impl()
    : AbstractChart(64, 8, 8, 44) {
    generateChartData();
    generateTickLabels();
}

void chart2d_impl::render(const int pWindowId,
                          const int pX, const int pY, const int pVPW, const int pVPH,
                          const glm::mat4& pView, const glm::mat4& pOrient)
{
    CheckGL("Begin chart2d_impl::renderChart");

    float lgap     = mLeftMargin + mTickSize/2;
    float bgap     = mBottomMargin + mTickSize/2;

    float offset_x = (lgap-mRightMargin) / pVPW;
    float offset_y = (bgap-mTopMargin) / pVPH;

    float w        = pVPW - (lgap + mRightMargin);
    float h        = pVPH - (bgap + mTopMargin);
    float scale_x  = w / pVPW;
    float scale_y  = h / pVPH;

    /* set uniform attributes of shader
     * for drawing the plot borders */
    glm::mat4 trans = glm::translate(glm::scale(glm::mat4(1),
                                                glm::vec3(scale_x, scale_y, 1)),
                                     glm::vec3(offset_x, offset_y, 0));

    /* Draw grid */
    chart2d_impl::bindResources(pWindowId);
    mBorderProgram.bind();
    glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    glUniform4fv(mBorderUniformColorIndex, 1, GRAY);
    glDrawArrays(GL_LINES, 4+2*mTickCount, 4*mTickCount);
    mBorderProgram.unbind();
    chart2d_impl::unbindResources();

    glEnable(GL_SCISSOR_TEST);
    glScissor(pX+mLeftMargin, pY+mBottomMargin+mTickSize/2, w, h);
    /* render all renderables */
    for (auto renderable : mRenderables) {
        renderable->setRanges(mXMin, mXMax, mYMin, mYMax, mZMin, mZMax);
        renderable->render(pWindowId, pX, pY, pVPW, pVPH, pView * trans, pOrient);
    }
    glDisable(GL_SCISSOR_TEST);

    chart2d_impl::bindResources(pWindowId);

    mBorderProgram.bind();
    glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    glUniform4fv(mBorderUniformColorIndex, 1, BLACK);
    /* Draw borders */
    glDrawArrays(GL_LINE_LOOP, 0, 4);
    mBorderProgram.unbind();

    /* bind the sprite shader program to
     * draw ticks on x and y axes */
    glPointSize((GLfloat)mTickSize);
    mSpriteProgram.bind();

    glUniform4fv(mSpriteUniformTickcolorIndex, 1, BLACK);
    glUniformMatrix4fv(mSpriteUniformMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    /* Draw tick marks on y axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 1);
    glDrawArrays(GL_POINTS, 4, mTickCount);
    /* Draw tick marks on x axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 0);
    glDrawArrays(GL_POINTS, 4+mTickCount, mTickCount);

    mSpriteProgram.unbind();
    glPointSize(1);
    chart2d_impl::unbindResources();

    renderTickLabels(pWindowId, int(w), int(h), mYText, trans, 0, false);
    renderTickLabels(pWindowId, int(w), int(h), mXText, trans, mTickCount, false);

    auto &fonter = getChartFont();
    fonter->setOthro2D(int(w), int(h));

    float pos[2];
    /* render chart axes titles */
    if (!mYTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f);
        pos[0] = CHART2D_FONT_SIZE; /* additional pixel gap from edge of rendering */
        pos[1] = h*(res.y+1.0f)/2.0f;
        fonter->render(pWindowId, pos, BLACK, mYTitle.c_str(), CHART2D_FONT_SIZE, true);
    }
    if (!mXTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(0.0f, -1.0f, 0.0f, 1.0f);
        pos[0] = w*(res.x+1.0f)/2.0f;
        pos[1] = h*(res.y+1.0f)/2.0f;
        pos[1] -= (4*mTickSize * (h/pVPH));
        fonter->render(pWindowId, pos, BLACK, mXTitle.c_str(), CHART2D_FONT_SIZE);
    }

    /* render all legends of the respective renderables */
    pos[0] = mLegendX;
    pos[1] = mLegendY;

    float lcol[4];

    for (auto renderable : mRenderables) {
        renderable->getColor(lcol[0], lcol[1], lcol[2], lcol[3]);

        float cpos[2];
        glm::vec4 res = trans * glm::vec4(pos[0], pos[1], 0.0f, 1.0f);
        cpos[0] = res.x * w;
        cpos[1] = res.y * h;
        fonter->render(pWindowId, cpos, lcol, renderable->legend().c_str(), CHART2D_FONT_SIZE);
        pos[1] -= (CHART2D_FONT_SIZE/(float)pVPH);
    }

    CheckGL("End chart2d_impl::renderChart");
}

/********************* END-chart2d_impl *********************/



/********************* BEGIN-chart3d_impl *********************/

void chart3d_impl::bindResources(const int pWindowId)
{
    CheckGL("Begin chart3d_impl::bindResources");
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(mBorderAttribPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mDecorVBO);
        glVertexAttribPointer(mBorderAttribPointIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }
    glBindVertexArray(mVAOMap[pWindowId]);
    CheckGL("End chart3d_impl::bindResources");
}

void chart3d_impl::unbindResources() const
{
    glBindVertexArray(0);
}

void chart3d_impl::pushTicktextCoords(const float pX, const float pY, const float pZ)
{
    mTickTextX.push_back(pX);
    mTickTextY.push_back(pY);
    mTickTextZ.push_back(pZ);
}

void chart3d_impl::generateChartData()
{
    CheckGL("Begin chart3d_impl::generateChartData");
    static const float border[] = { -1, -1, 1,  -1, -1, -1,  -1, -1, -1,  1, -1, -1,  1, -1, -1,  1, 1, -1 };
    static const int nValues = sizeof(border)/sizeof(float);

    std::vector<float> decorData;
    std::copy(border, border+nValues, std::back_inserter(decorData));

    float step = getTickStepSize(-1, 1);
    int ticksLeft = getNumTicksC2E();

    /* push tick points for z axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, -1.0f, -1.0f, 0.0f);
    pushTicktextCoords(-1.0f, -1.0f, 0.0f);
    mZText.push_back(toString(0));

    for(int i=1; i<=ticksLeft; ++i) {
        /* (0, -1] to [-1, -1] */
        float neg = i*-step;
        pushPoint(decorData, -1.0f, -1.0f, neg);
        /* push tick marks */
        pushTicktextCoords(-1.0f, -1.0f, neg);
        /* push tick text label */
        mZText.push_back(toString(neg));

        /* (0, -1] to [1, -1] */
        float pos = i*step;
        pushPoint(decorData, -1.0f, -1.0f, pos);
        /* push tick marks */
        pushTicktextCoords(-1.0f, -1.0f, pos);
        /* push tick text label */
        mZText.push_back(toString(pos));
    }
    /* push tick points for y axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, 1.0f, 0.0f, -1.0f);
    pushTicktextCoords(1.0f, 0.0f, -1.0f);
    mYText.push_back(toString(0));

    for(int i=1; i<=ticksLeft; ++i) {
        /* [-1, 0) to [-1, -1] */
        float neg = i*-step;
        pushPoint(decorData, 1.0f, neg, -1.0f);
        pushTicktextCoords(1.0f, neg, -1.0f);
        mYText.push_back(toString(neg));

        /* [-1, 0) to [-1, 1] */
        float pos = i*step;
        pushPoint(decorData, 1.0f, pos, -1.0f);
        pushTicktextCoords(1.0f, pos, -1.0f);
        mYText.push_back(toString(pos));
    }

    /* push tick points for x axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, 0.0f, -1.0f, -1.0f);
    pushTicktextCoords( 0.0f, -1.0f, -1.0f);
    mXText.push_back(toString(0));

    for(int i=1; i<=ticksLeft; ++i) {
        /* (0, -1] to [-1, -1] */
        float neg = i*-step;
        pushPoint(decorData, neg, -1.0f, -1.0f);
        pushTicktextCoords( neg, -1.0f, -1.0f);
        mXText.push_back(toString(neg));

        /* [-1, 0) to [-1, 1] */
        float pos = i*step;
        pushPoint(decorData, pos, -1.0f, -1.0f);
        pushTicktextCoords( pos, -1.0f, -1.0f);
        mXText.push_back(toString(pos));
    }

    /* push grid lines */
    /* xy plane center lines */
    pushPoint(decorData, -1.0f, 0.0f, -1.0f);
    pushPoint(decorData,  1.0f, 0.0f, -1.0f);
    pushPoint(decorData,  0.0f,-1.0f, -1.0f);
    pushPoint(decorData,  0.0f, 1.0f, -1.0f);
    /* xz plane center lines */
    pushPoint(decorData, -1.0f, -1.0f, 0.0f);
    pushPoint(decorData, -1.0f,  1.0f, 0.0f);
    pushPoint(decorData, -1.0f,  0.0f,-1.0f);
    pushPoint(decorData, -1.0f,  0.0f, 1.0f);
    /* yz plane center lines */
    pushPoint(decorData, -1.0f,  1.0f, 0.0f);
    pushPoint(decorData,  1.0f,  1.0f, 0.0f);
    pushPoint(decorData,  0.0f,  1.0f,-1.0f);
    pushPoint(decorData,  0.0f,  1.0f, 1.0f);
    for (int i=1; i<=ticksLeft; ++i) {
        float delta = i*step;
        /* xy plane center lines */
        pushPoint(decorData, -1.0f,-delta, -1.0f);
        pushPoint(decorData,  1.0f,-delta, -1.0f);
        pushPoint(decorData, -1.0f, delta, -1.0f);
        pushPoint(decorData,  1.0f, delta, -1.0f);
        pushPoint(decorData,-delta, -1.0f, -1.0f);
        pushPoint(decorData,-delta,  1.0f, -1.0f);
        pushPoint(decorData, delta, -1.0f, -1.0f);
        pushPoint(decorData, delta,  1.0f, -1.0f);
        /* xz plane center lines */
        pushPoint(decorData, -1.0f, -1.0f,-delta);
        pushPoint(decorData, -1.0f,  1.0f,-delta);
        pushPoint(decorData, -1.0f, -1.0f, delta);
        pushPoint(decorData, -1.0f,  1.0f, delta);
        pushPoint(decorData, -1.0f,-delta, -1.0f);
        pushPoint(decorData, -1.0f,-delta,  1.0f);
        pushPoint(decorData, -1.0f, delta, -1.0f);
        pushPoint(decorData, -1.0f, delta,  1.0f);
        /* yz plane center lines */
        pushPoint(decorData, -1.0f,  1.0f,-delta);
        pushPoint(decorData,  1.0f,  1.0f,-delta);
        pushPoint(decorData, -1.0f,  1.0f, delta);
        pushPoint(decorData,  1.0f,  1.0f, delta);
        pushPoint(decorData,-delta,  1.0f, -1.0f);
        pushPoint(decorData,-delta,  1.0f,  1.0f);
        pushPoint(decorData, delta,  1.0f, -1.0f);
        pushPoint(decorData, delta,  1.0f,  1.0f);
    }

    /* check if decoration VBO has been already used(case where
     * tick marks are being changed from default(21) */
    if (mDecorVBO != 0)
        glDeleteBuffers(1, &mDecorVBO);

    /* create vbo that has the border and axis data */
    mDecorVBO = createBuffer<float>(GL_ARRAY_BUFFER, decorData.size(),
                                    &(decorData.front()), GL_STATIC_DRAW);
    CheckGL("End chart3d_impl::generateChartData");
}

void chart3d_impl::generateTickLabels()
{
    /* remove all the tick text markers that were generated
     * by default during the base class(chart) creation and
     * update the text markers based on the new axes limits*/
    mXText.clear();
    mYText.clear();
    mZText.clear();

    float xstep = getTickStepSize(mXMin, mXMax);
    float ystep = getTickStepSize(mYMin, mYMax);
    float zstep = getTickStepSize(mZMin, mZMax);
    float xmid  = (mXMax+mXMin)/2.0f;
    float ymid  = (mYMax+mYMin)/2.0f;
    float zmid  = (mZMax+mZMin)/2.0f;

    int ticksLeft = getNumTicksC2E();

    /* push tick points for z axis */
    mZText.push_back(toString(zmid));
    for (int i = 1; i <= ticksLeft; i++) {
        mZText.push_back(toString(zmid + i*-zstep));
        mZText.push_back(toString(zmid + i*zstep));
    }
    /* push tick points for y axis */
    mYText.push_back(toString(ymid));
    for (int i = 1; i <= ticksLeft; i++) {
        mYText.push_back(toString(ymid + i*-ystep));
        mYText.push_back(toString(ymid + i*ystep));
    }
    /* push tick points for x axis */
    mXText.push_back(toString(xmid));
    for (int i = 1; i <= ticksLeft; i++) {
        mXText.push_back(toString(xmid + i*-xstep));
        mXText.push_back(toString(xmid + i*xstep));
    }
}

chart3d_impl::chart3d_impl()
    :AbstractChart(32, 32, 32, 32) {
    generateChartData();
    generateTickLabels();
}

void chart3d_impl::render(const int pWindowId,
                          const int pX, const int pY, const int pVPW, const int pVPH,
                          const glm::mat4& pView, const glm::mat4& pOrient)
{
    /* set uniform attributes of shader
     * for drawing the plot borders */
    static const glm::mat4 VIEW = glm::lookAt(glm::vec3(-1.f,0.5f, 1.f),
                                              glm::vec3( 1.f,-1.f,-1.f),
                                              glm::vec3( 0.f, 1.f, 0.f));
    static const glm::mat4 PROJECTION = glm::ortho(-1.75f, 1.75f, -1.75f, 1.75f, -0.001f, 1000.f);
    static const glm::mat4 MODEL = glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(0,1,0)) *
                                   glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(1,0,0));
    static const glm::mat4 PV = PROJECTION * VIEW;
    static const glm::mat4 PVM = PV * MODEL;

    CheckGL("Being chart3d_impl::renderChart");

    /* draw grid */
    chart3d_impl::bindResources(pWindowId);
    mBorderProgram.bind();
    glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE, glm::value_ptr(PVM));
    glUniform4fv(mBorderUniformColorIndex, 1, GRAY);
    glDrawArrays(GL_LINES, 6+3*mTickCount, 12*mTickCount);
    mBorderProgram.unbind();
    chart3d_impl::unbindResources();

    glEnable(GL_SCISSOR_TEST);
    glScissor(pX, pY, pVPW, pVPH);
    glm::mat4 renderableMat = PROJECTION * pView * VIEW;
    /* render all the renderables */
    for (auto renderable : mRenderables) {
        renderable->setRanges(mXMin, mXMax, mYMin, mYMax, mZMin, mZMax);
        renderable->render(pWindowId, pX, pY, pVPW, pVPH, renderableMat, pOrient);
    }
    glDisable(GL_SCISSOR_TEST);

    /* Draw borders */
    chart3d_impl::bindResources(pWindowId);

    mBorderProgram.bind();
    glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE, glm::value_ptr(PVM));
    glUniform4fv(mBorderUniformColorIndex, 1, BLACK);
    glDrawArrays(GL_LINES, 0, 6);
    mBorderProgram.unbind();

    /* bind the sprite shader program to
     * draw ticks on x and y axes */
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize((GLfloat)mTickSize);
    mSpriteProgram.bind();

    glUniform4fv(mSpriteUniformTickcolorIndex, 1, BLACK);
    glUniformMatrix4fv(mSpriteUniformMatIndex, 1, GL_FALSE, glm::value_ptr(PVM));
    /* Draw tick marks on z axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 1);
    glDrawArrays(GL_POINTS, 6, mTickCount);
    /* Draw tick marks on y axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 0);
    glDrawArrays(GL_POINTS, 6 + mTickCount, mTickCount);
    /* Draw tick marks on x axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 0);
    glDrawArrays(GL_POINTS, 6 + (2*mTickCount), mTickCount);

    mSpriteProgram.unbind();
    glPointSize((GLfloat)1);
    glDisable(GL_PROGRAM_POINT_SIZE);

    chart3d_impl::unbindResources();

    float w = float(pVPW - (mLeftMargin + mRightMargin + mTickSize));
    float h = float(pVPH - (mTopMargin + mBottomMargin + mTickSize));

    renderTickLabels(pWindowId, w, h, mZText, PVM, 0);
    renderTickLabels(pWindowId, w, h, mYText, PVM, mTickCount);
    renderTickLabels(pWindowId, w, h, mXText, PVM, 2*mTickCount);

    auto &fonter = getChartFont();
    fonter->setOthro2D(int(w), int(h));
    float pos[2];
    /* render chart axes titles */
    if (!mZTitle.empty()) {
        glm::vec4 res = PVM * glm::vec4(-1.0f, -1.0f, 0.0f, 1.0f);
        pos[0] = w*(res.x/res.w+1.0f)/2.0f;
        pos[1] = h*(res.y/res.w+1.0f)/2.0f;
        pos[0] -= 6*(mTickSize * (w/pVPW));
        pos[1] += mZTitle.length()/2 * CHART2D_FONT_SIZE;
        fonter->render(pWindowId, pos, BLACK, mZTitle.c_str(), CHART2D_FONT_SIZE, true);
    }
    if (!mYTitle.empty()) {
        glm::vec4 res = PVM * glm::vec4(1.0f, 0.0f, -1.0f, 1.0f);
        pos[0] = w*(res.x/res.w+1.0f)/2.0f;
        pos[1] = h*(res.y/res.w+1.0f)/2.0f;
        pos[0] += 0.5 * ((mTickSize * (w/pVPW)) + mYTitle.length()/2 * CHART2D_FONT_SIZE);
        pos[1] -= 4*(mTickSize * (h/pVPH));
        fonter->render(pWindowId, pos, BLACK, mYTitle.c_str(), CHART2D_FONT_SIZE);
    }
    if (!mXTitle.empty()) {
        glm::vec4 res = PVM * glm::vec4(0.0f, -1.0f, -1.0f, 1.0f);
        pos[0] = w*(res.x/res.w+1.0f)/2.0f;
        pos[1] = h*(res.y/res.w+1.0f)/2.0f;
        pos[0] -= (mTickSize * (w/pVPW)) + mXTitle.length()/2 * CHART2D_FONT_SIZE;
        pos[1] -= 4*(mTickSize * (h/pVPH));
        fonter->render(pWindowId, pos, BLACK, mXTitle.c_str(), CHART2D_FONT_SIZE);
    }

    CheckGL("End chart3d_impl::renderChart");
}

}
}
