/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <chart_impl.hpp>
#include <common/font.hpp>
#include <font_impl.hpp>
#include <gl_helpers.hpp>
#include <histogram_impl.hpp>
#include <image_impl.hpp>
#include <plot_impl.hpp>
#include <shader_headers/chart_fs.hpp>
#include <shader_headers/chart_vs.hpp>
#include <shader_headers/tick_fs.hpp>
#include <surface_impl.hpp>
#include <window_impl.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <regex>
#include <sstream>

using namespace std;
using namespace forge::common;

namespace forge {
namespace opengl {

typedef std::vector<std::string>::const_iterator StringIter;

static const int CHART2D_FONT_SIZE = 12;
static const std::regex PRINTF_FIXED_FLOAT_RE("%[0-9]*.[0-9]*f");

const std::shared_ptr<forge::opengl::font_impl>& getChartFont() {
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
void pushPoint(vector<T>& points, T x, T y) {
    points.push_back(x);
    points.push_back(y);
}

template<typename T>
void pushPoint(vector<T>& points, T x, T y, T z) {
    points.push_back(x);
    points.push_back(y);
    points.push_back(z);
}

int calcTrgtFntSize(const float w, const float h) { return CHART2D_FONT_SIZE; }

/********************* BEGIN-AbstractChart *********************/

void AbstractChart::renderTickLabels(const int pWindowId, const uint32_t pW,
                                     const uint32_t pH,
                                     const std::vector<std::string>& pTexts,
                                     const int pFontSize,
                                     const glm::mat4& pTransformation,
                                     const int pCoordsOffset,
                                     const bool pUseZoffset) const {
    auto& fonter = getChartFont();
    fonter->setOthro2D(int(pW), int(pH));

    float pos[2];
    for (StringIter it = pTexts.begin(); it != pTexts.end(); ++it) {
        int idx     = int(it - pTexts.begin());
        glm::vec4 p = glm::vec4(
            mTickTextX[idx + pCoordsOffset], mTickTextY[idx + pCoordsOffset],
            (pUseZoffset ? mTickTextZ[idx + pCoordsOffset] : 0), 1);
        glm::vec4 res = pTransformation * p;

        /* convert text position from [-1,1] range to
         * [0, 1) range and then offset horizontally
         * to compensate for margins and ticksize */
        pos[0] = pW * (res.x / res.w + 1.0f) / 2.0f;
        pos[1] = pH * (res.y / res.w + 1.0f) / 2.0f;

        const float strQlen = it->length() / 4.0f;

        /* offset based on text size to align
         * text center with tick mark position
         * */
        if (pCoordsOffset < mTickCount) {
            // offsets for z axis labels if pUseZoffset is true i.e. 3d chart
            // offsets for y axis labels if pUseZoffset is false i.e. 2d chart

            pos[0] -= (pFontSize + pFontSize * strQlen * 2.0f + getTickSize());
            pos[1] -= (pFontSize * 0.4f);

        } else if (pCoordsOffset >= mTickCount &&
                   pCoordsOffset < 2 * mTickCount) {
            // offsets for y axis labels if pUseZoffset is true i.e. 3d chart
            // offsets for x axis labels if pUseZoffset is false i.e. 2d chart

            pos[0] -= (pFontSize * strQlen);
            pos[1] -= (pUseZoffset ? (pFontSize * 3.0f) : (pFontSize * 1.5f));

        } else {
            // offsets for x axis labels in 3d chart
            // this section gets executed only when pCoordsOffset > 2*mTickCount

            pos[0] -= (pFontSize * strQlen);
            pos[1] -= (pFontSize * 1.5f);
        }

        fonter->render(pWindowId, pos, BLACK, it->c_str(), pFontSize);
    }
}

AbstractChart::AbstractChart(const float pLeftMargin, const float pRightMargin,
                             const float pTopMargin, const float pBottomMargin)
    : mTickCount(9)
    , mTickSize(10.0f)
    , mLeftMargin(pLeftMargin)
    , mRightMargin(pRightMargin)
    , mTopMargin(pTopMargin)
    , mBottomMargin(pBottomMargin)
    , mRenderAxes(true)
    , mXLabelFormat("%4.1f")
    , mXMax(0)
    , mXMin(0)
    , mYLabelFormat("%4.1f")
    , mYMax(0)
    , mYMin(0)
    , mZLabelFormat("%4.1f")
    , mZMax(0)
    , mZMin(0)
    , mXTitle("X-Axis")
    , mYTitle("Y-Axis")
    , mZTitle("Z-Axis")
    , mDecorVBO(-1)
    , mBorderProgram(glsl::chart_vs.c_str(), glsl::chart_fs.c_str())
    , mSpriteProgram(glsl::chart_vs.c_str(), glsl::tick_fs.c_str())
    , mBorderAttribPointIndex(-1)
    , mBorderUniformColorIndex(-1)
    , mBorderUniformMatIndex(-1)
    , mSpriteUniformMatIndex(-1)
    , mSpriteUniformTickcolorIndex(-1)
    , mSpriteUniformTickaxisIndex(-1)
    , mLegendX(0.4f)
    , mLegendY(0.9f) {
    CheckGL("Begin AbstractChart::AbstractChart");
    /* load font Vera font for chart text
     * renderings, below function actually returns a constant
     * reference to font object used by Chart objects, we are
     * calling it here just to make sure required font glyphs
     * are loaded into the shared Font object */
    getChartFont();

    mBorderAttribPointIndex  = mBorderProgram.getAttributeLocation("point");
    mBorderUniformColorIndex = mBorderProgram.getUniformLocation("color");
    mBorderUniformMatIndex   = mBorderProgram.getUniformLocation("transform");

    mSpriteUniformTickcolorIndex =
        mSpriteProgram.getUniformLocation("tick_color");
    mSpriteUniformMatIndex = mSpriteProgram.getUniformLocation("transform");
    mSpriteUniformTickaxisIndex = mSpriteProgram.getUniformLocation("isYAxis");

    CheckGL("End AbstractChart::AbstractChart");
}

AbstractChart::~AbstractChart() {
    for (auto it = mVAOMap.begin(); it != mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mDecorVBO);
}

void AbstractChart::setAxesVisibility(const bool isVisible) {
    mRenderAxes = isVisible;
}

void AbstractChart::setAxesLimits(const float pXmin, const float pXmax,
                                  const float pYmin, const float pYmax,
                                  const float pZmin, const float pZmax) {
    mXMin = pXmin;
    mXMax = pXmax;
    mYMin = pYmin;
    mYMax = pYmax;
    mZMin = pZmin;
    mZMax = pZmax;

    /*
     * Once the axes ranges are known, we can generate
     * tick labels. The following functions is a pure
     * virtual function and has to be implemented by the
     * derived class
     */
    generateTickLabels();
}

void AbstractChart::setAxesLabelFormat(const std::string& pXFormat,
                                       const std::string& pYFormat,
                                       const std::string& pZFormat) {
    mXLabelFormat = std::string(pXFormat);
    mYLabelFormat = std::string(pYFormat);
    mZLabelFormat = std::string(pZFormat);

    /*
     * Re-generate tick labels since label format has
     * been changed by the user explicitly.
     */
    generateTickLabels();
}

void AbstractChart::getAxesLimits(float* pXmin, float* pXmax, float* pYmin,
                                  float* pYmax, float* pZmin, float* pZmax) {
    *pXmin = mXMin;
    *pXmax = mXMax;
    *pYmin = mYMin;
    *pYmax = mYMax;
    *pZmin = mZMin;
    *pZmax = mZMax;
}

void AbstractChart::setAxesTitles(const char* pXTitle, const char* pYTitle,
                                  const char* pZTitle) {
    mXTitle = (pXTitle ? std::string(pXTitle) : std::string("X-Axis"));
    mYTitle = (pYTitle ? std::string(pYTitle) : std::string("Y-Axis"));
    mZTitle = (pZTitle ? std::string(pZTitle) : std::string("Z-Axis"));
}

void AbstractChart::setLegendPosition(const float pX, const float pY) {
    mLegendX = pX;
    mLegendY = pY;
}

float AbstractChart::xmax() const { return mXMax; }
float AbstractChart::xmin() const { return mXMin; }
float AbstractChart::ymax() const { return mYMax; }
float AbstractChart::ymin() const { return mYMin; }
float AbstractChart::zmax() const { return mZMax; }
float AbstractChart::zmin() const { return mZMin; }

void AbstractChart::addRenderable(
    const std::shared_ptr<AbstractRenderable> pRenderable) {
    mRenderables.emplace_back(pRenderable);
}

/********************* END-AbstractChart *********************/

/********************* BEGIN-chart2d_impl *********************/

void chart2d_impl::bindResources(const int pWindowId) {
    CheckGL("Begin chart2d_impl::bindResources");
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(mBorderAttribPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mDecorVBO);
        glVertexAttribPointer(mBorderAttribPointIndex, 2, GL_FLOAT, GL_FALSE, 0,
                              0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }
    glBindVertexArray(mVAOMap[pWindowId]);
    CheckGL("End chart2d_impl::bindResources");
}

void chart2d_impl::unbindResources() const { glBindVertexArray(0); }

void chart2d_impl::pushTicktextCoords(const float pX, const float pY,
                                      const float pZ) {
    mTickTextX.push_back(pX);
    mTickTextY.push_back(pY);
}

void chart2d_impl::generateChartData() {
    CheckGL("Begin chart2d_impl::generateChartData");
    static const float border[8] = {-1, -1, 1, -1, 1, 1, -1, 1};
    static const int nValues     = sizeof(border) / sizeof(float);

    std::vector<float> decorData;
    std::copy(border, border + nValues, std::back_inserter(decorData));

    float step    = getTickStepSize(-1, 1);
    int ticksLeft = getNumTicksC2E();

    /* push tick points for y axis:
     * push (0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, -1.0f, 0.0f);
    pushTicktextCoords(-1.0f, 0.0f);
    mYText.push_back(toString(0, mYLabelFormat));

    for (int i = 1; i <= ticksLeft; ++i) {
        /* [-1, 0) to [-1, -1] */
        float neg = i * -step;
        pushPoint(decorData, -1.0f, neg);
        /* puch tick marks */
        pushTicktextCoords(-1.0f, neg);
        /* push tick text label */
        mYText.push_back(toString(neg, mYLabelFormat));

        /* [-1, 0) to [-1, 1] */
        float pos = i * step;
        pushPoint(decorData, -1.0f, pos);
        /* puch tick marks */
        pushTicktextCoords(-1.0f, pos);
        /* push tick text label */
        mYText.push_back(toString(pos, mYLabelFormat));
    }

    /* push tick points for x axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, 0.0f, -1.0f);
    pushTicktextCoords(0.0f, -1.0f);
    mXText.push_back(toString(0, mXLabelFormat));

    for (int i = 1; i <= ticksLeft; ++i) {
        /* (0, -1] to [-1, -1] */
        float neg = i * -step;
        pushPoint(decorData, neg, -1.0f);
        pushTicktextCoords(neg, -1.0f);
        mXText.push_back(toString(neg, mXLabelFormat));

        /* (0, -1] to [1, -1] */
        float pos = i * step;
        pushPoint(decorData, pos, -1.0f);
        pushTicktextCoords(pos, -1.0f);
        mXText.push_back(toString(pos, mXLabelFormat));
    }

    /* push grid lines */
    pushPoint(decorData, -1.0f, 0.0f);
    pushPoint(decorData, 1.0f, 0.0f);
    pushPoint(decorData, 0.0f, -1.0f);
    pushPoint(decorData, 0.0f, 1.0f);
    for (int i = 1; i < ticksLeft; ++i) {
        float delta = i * step;
        pushPoint(decorData, -1.0f, -delta);
        pushPoint(decorData, 1.0f, -delta);
        pushPoint(decorData, -1.0f, delta);
        pushPoint(decorData, 1.0f, delta);
        pushPoint(decorData, -delta, -1.0f);
        pushPoint(decorData, -delta, 1.0f);
        pushPoint(decorData, delta, -1.0f);
        pushPoint(decorData, delta, 1.0f);
    }

    /* check if decoration VBO has been already used(case where
     * tick marks are being changed from default(21) */
    if (mDecorVBO != 0) glDeleteBuffers(1, &mDecorVBO);

    /* create vbo that has the border and axis data */
    mDecorVBO = createBuffer<float>(GL_ARRAY_BUFFER, decorData.size(),
                                    &(decorData.front()), GL_STATIC_DRAW);
    CheckGL("End chart2d_impl::generateChartData");
}

int getDigitCount(float value) {
    int count = 0;

    float v = std::abs(value);

    if (v < 1.0f) {
        if (v > FLT_EPSILON) {
            while (v < 1) {
                v = v * 10.0f;
                count++;
            }
        }
    } else {
        int num = int(value);
        while (num) {
            num = num / 10;
            count++;
        }
    }

    return count;
}

void chart2d_impl::generateTickLabels() {
    /* remove all the tick text markers that were generated
     * by default during the base class(chart) creation and
     * update the text markers based on the new axes limits
     * */
    mXText.clear();
    mYText.clear();
    mZText.clear();

    // By default chart's axes labels show numbers in
    // fixed floating point format, unless the users requests
    // for any other format explicitly. However, if the string
    // representation of the range of data of given axis exceeds
    // certain length, the numbers are converted to scientific notation.
    // Y Axis label format
    if (toString(std::fabs(mYMax - mYMin), mYLabelFormat).length() > 5 &&
        std::regex_search(mYLabelFormat, PRINTF_FIXED_FLOAT_RE)) {
        mYLabelFormat = std::string("%.2e");
    }
    // X Axis label format
    if (toString(std::fabs(mXMax - mXMin), mXLabelFormat).length() > 5 &&
        std::regex_search(mXLabelFormat, PRINTF_FIXED_FLOAT_RE)) {
        mXLabelFormat = std::string("%.2e");
    }

    float xstep = getTickStepSize(mXMin, mXMax);
    float ystep = getTickStepSize(mYMin, mYMax);
    float xmid  = (mXMax + mXMin) / 2.0f;
    float ymid  = (mYMax + mYMin) / 2.0f;

    int ticksLeft = getNumTicksC2E();

    /* push tick points for y axis */
    mYText.push_back(toString(ymid, mYLabelFormat));
    size_t maxYLabelWidth = 0;
    for (int i = 1; i <= ticksLeft; i++) {
        std::string temp = toString(ymid + i * -ystep, mYLabelFormat);
        mYText.push_back(temp);
        maxYLabelWidth = std::max(maxYLabelWidth, temp.length());

        temp = toString(ymid + i * ystep, mYLabelFormat);
        mYText.push_back(temp);
        maxYLabelWidth = std::max(maxYLabelWidth, temp.length());
    }

    /* push tick points for x axis */
    mXText.push_back(toString(xmid, mXLabelFormat));
    for (int i = 1; i <= ticksLeft; i++) {
        mXText.push_back(toString(xmid + i * -xstep, mXLabelFormat));
        mXText.push_back(toString(xmid + i * xstep, mXLabelFormat));
    }
}

chart2d_impl::chart2d_impl()
    : AbstractChart(0.13139f, 0.1008f, 0.0755f, 0.1077f) {
    generateChartData();
    generateTickLabels();
}

void chart2d_impl::render(const int pWindowId, const int pX, const int pY,
                          const int pVPW, const int pVPH,
                          const glm::mat4& pView, const glm::mat4& pOrient) {
    CheckGL("Begin chart2d_impl::renderChart");

    float lgap = getLeftMargin(pVPW) + getTickSize() / 2.0f;
    float bgap = getBottomMargin(pVPH) + getTickSize() / 2.0f;

    float offset_x = (lgap - getRightMargin(pVPW)) / pVPW;
    float offset_y = (bgap - getTopMargin(pVPH)) / pVPH;

    float w       = pVPW - (lgap + getRightMargin(pVPW));
    float h       = pVPH - (bgap + getTopMargin(pVPH));
    float scale_x = w / pVPW;
    float scale_y = h / pVPH;

    glm::mat4 trans =
        glm::translate(glm::scale(glm::mat4(1), glm::vec3(scale_x, scale_y, 1)),
                       glm::vec3(offset_x, offset_y, 0));

    if (mRenderAxes) {
        /* Draw grid */
        chart2d_impl::bindResources(pWindowId);
        mBorderProgram.bind();
        glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE,
                           glm::value_ptr(trans));
        glUniform4fv(mBorderUniformColorIndex, 1, GRAY);
        glDrawArrays(GL_LINES, 4 + 2 * mTickCount, 8 * mTickCount - 16);
        mBorderProgram.unbind();
        chart2d_impl::unbindResources();
    }

    glEnable(GL_SCISSOR_TEST);
    glScissor(GLint(pX + lgap), GLint(pY + bgap), GLsizei(w), GLsizei(h));

    /* render all renderables */
    for (auto renderable : mRenderables) {
        renderable->setRanges(mXMin, mXMax, mYMin, mYMax, mZMin, mZMax);
        renderable->render(pWindowId, pX, pY, pVPW, pVPH, pView * trans,
                           pOrient);
    }

    glDisable(GL_SCISSOR_TEST);

    const int trgtFntSize = calcTrgtFntSize(w, h);
    auto& fonter          = getChartFont();

    if (mRenderAxes) {
        chart2d_impl::bindResources(pWindowId);

        mBorderProgram.bind();
        glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE,
                           glm::value_ptr(trans));
        glUniform4fv(mBorderUniformColorIndex, 1, BLACK);
        /* Draw borders */
        glDrawArrays(GL_LINE_LOOP, 0, 4);
        mBorderProgram.unbind();

        /* bind the sprite shader program to
         * draw ticks on x and y axes */
        glPointSize((GLfloat)getTickSize());
        mSpriteProgram.bind();

        glUniform4fv(mSpriteUniformTickcolorIndex, 1, BLACK);
        glUniformMatrix4fv(mSpriteUniformMatIndex, 1, GL_FALSE,
                           glm::value_ptr(trans));
        /* Draw tick marks on y axis */
        glUniform1i(mSpriteUniformTickaxisIndex, 1);
        glDrawArrays(GL_POINTS, 4, mTickCount);
        /* Draw tick marks on x axis */
        glUniform1i(mSpriteUniformTickaxisIndex, 0);
        glDrawArrays(GL_POINTS, 4 + mTickCount, mTickCount);

        mSpriteProgram.unbind();
        glPointSize(1);
        chart2d_impl::unbindResources();

        renderTickLabels(pWindowId, int(w), int(h), mYText, trgtFntSize, trans,
                         0, false);
        renderTickLabels(pWindowId, int(w), int(h), mXText, trgtFntSize, trans,
                         mTickCount, false);

        fonter->setOthro2D(int(w), int(h));

        float pos[2];

        /* render chart axes titles */
        if (!mYTitle.empty()) {
            glm::vec4 res = trans * glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f);

            pos[0] = w * (res.x + 1.0f) / 2.0f;
            pos[1] = h * (res.y + 1.0f) / 2.0f;

            pos[0] -= (5.0f * trgtFntSize);
            pos[1] += (trgtFntSize);

            fonter->render(pWindowId, pos, BLACK, mYTitle.c_str(), trgtFntSize,
                           true);
        }
        if (!mXTitle.empty()) {
            glm::vec4 res = trans * glm::vec4(0.0f, -1.0f, 0.0f, 1.0f);

            pos[0] = w * (res.x + 1.0f) / 2.0f;
            pos[1] = h * (res.y + 1.0f) / 2.0f;

            pos[1] -= (2.5f * trgtFntSize);

            fonter->render(pWindowId, pos, BLACK, mXTitle.c_str(), trgtFntSize);
        }
    }

    /* render all legends of the respective renderables */
    float pos[2] = {mLegendX, mLegendY};
    float lcol[4];

    for (auto renderable : mRenderables) {
        renderable->getColor(lcol[0], lcol[1], lcol[2], lcol[3]);

        float cpos[2];
        glm::vec4 res = trans * glm::vec4(pos[0], pos[1], 0.0f, 1.0f);
        cpos[0]       = res.x * w;
        cpos[1]       = res.y * h;
        fonter->render(pWindowId, cpos, lcol, renderable->legend().c_str(),
                       trgtFntSize);
        pos[1] -= (trgtFntSize / (float)pVPH);
    }

    CheckGL("End chart2d_impl::renderChart");
}

/********************* END-chart2d_impl *********************/

/********************* BEGIN-chart3d_impl *********************/

void chart3d_impl::bindResources(const int pWindowId) {
    CheckGL("Begin chart3d_impl::bindResources");
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(mBorderAttribPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mDecorVBO);
        glVertexAttribPointer(mBorderAttribPointIndex, 3, GL_FLOAT, GL_FALSE, 0,
                              0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }
    glBindVertexArray(mVAOMap[pWindowId]);
    CheckGL("End chart3d_impl::bindResources");
}

void chart3d_impl::unbindResources() const { glBindVertexArray(0); }

void chart3d_impl::pushTicktextCoords(const float pX, const float pY,
                                      const float pZ) {
    mTickTextX.push_back(pX);
    mTickTextY.push_back(pY);
    mTickTextZ.push_back(pZ);
}

void chart3d_impl::generateChartData() {
    CheckGL("Begin chart3d_impl::generateChartData");
    static const float border[] = {-1, -1, 1,  -1, -1, -1, -1, -1, -1,
                                   1,  -1, -1, 1,  -1, -1, 1,  1,  -1};
    static const int nValues    = sizeof(border) / sizeof(float);

    std::vector<float> decorData;
    std::copy(border, border + nValues, std::back_inserter(decorData));

    float step    = getTickStepSize(-1, 1);
    int ticksLeft = getNumTicksC2E();

    /* push tick points for z axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, -1.0f, -1.0f, 0.0f);
    pushTicktextCoords(-1.0f, -1.0f, 0.0f);
    mZText.push_back(toString(0, mZLabelFormat));

    for (int i = 1; i <= ticksLeft; ++i) {
        /* (0, -1] to [-1, -1] */
        float neg = i * -step;
        pushPoint(decorData, -1.0f, -1.0f, neg);
        /* push tick marks */
        pushTicktextCoords(-1.0f, -1.0f, neg);
        /* push tick text label */
        mZText.push_back(toString(neg, mZLabelFormat));

        /* (0, -1] to [1, -1] */
        float pos = i * step;
        pushPoint(decorData, -1.0f, -1.0f, pos);
        /* push tick marks */
        pushTicktextCoords(-1.0f, -1.0f, pos);
        /* push tick text label */
        mZText.push_back(toString(pos, mZLabelFormat));
    }
    /* push tick points for y axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, 1.0f, 0.0f, -1.0f);
    pushTicktextCoords(1.0f, 0.0f, -1.0f);
    mYText.push_back(toString(0, mYLabelFormat));

    for (int i = 1; i <= ticksLeft; ++i) {
        /* [-1, 0) to [-1, -1] */
        float neg = i * -step;
        pushPoint(decorData, 1.0f, neg, -1.0f);
        pushTicktextCoords(1.0f, neg, -1.0f);
        mYText.push_back(toString(neg, mYLabelFormat));

        /* [-1, 0) to [-1, 1] */
        float pos = i * step;
        pushPoint(decorData, 1.0f, pos, -1.0f);
        pushTicktextCoords(1.0f, pos, -1.0f);
        mYText.push_back(toString(pos, mYLabelFormat));
    }

    /* push tick points for x axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, 0.0f, -1.0f, -1.0f);
    pushTicktextCoords(0.0f, -1.0f, -1.0f);
    mXText.push_back(toString(0, mXLabelFormat));

    for (int i = 1; i <= ticksLeft; ++i) {
        /* (0, -1] to [-1, -1] */
        float neg = i * -step;
        pushPoint(decorData, neg, -1.0f, -1.0f);
        pushTicktextCoords(neg, -1.0f, -1.0f);
        mXText.push_back(toString(neg, mXLabelFormat));

        /* [-1, 0) to [-1, 1] */
        float pos = i * step;
        pushPoint(decorData, pos, -1.0f, -1.0f);
        pushTicktextCoords(pos, -1.0f, -1.0f);
        mXText.push_back(toString(pos, mXLabelFormat));
    }

    /* push grid lines */
    /* xy plane center lines */
    pushPoint(decorData, -1.0f, 0.0f, -1.0f);
    pushPoint(decorData, 1.0f, 0.0f, -1.0f);
    pushPoint(decorData, 0.0f, -1.0f, -1.0f);
    pushPoint(decorData, 0.0f, 1.0f, -1.0f);
    /* xz plane center lines */
    pushPoint(decorData, -1.0f, -1.0f, 0.0f);
    pushPoint(decorData, -1.0f, 1.0f, 0.0f);
    pushPoint(decorData, -1.0f, 0.0f, -1.0f);
    pushPoint(decorData, -1.0f, 0.0f, 1.0f);
    /* yz plane center lines */
    pushPoint(decorData, -1.0f, 1.0f, 0.0f);
    pushPoint(decorData, 1.0f, 1.0f, 0.0f);
    pushPoint(decorData, 0.0f, 1.0f, -1.0f);
    pushPoint(decorData, 0.0f, 1.0f, 1.0f);
    for (int i = 1; i < ticksLeft; ++i) {
        float delta = i * step;
        /* xy plane center lines */
        pushPoint(decorData, -1.0f, -delta, -1.0f);
        pushPoint(decorData, 1.0f, -delta, -1.0f);
        pushPoint(decorData, -1.0f, delta, -1.0f);
        pushPoint(decorData, 1.0f, delta, -1.0f);
        pushPoint(decorData, -delta, -1.0f, -1.0f);
        pushPoint(decorData, -delta, 1.0f, -1.0f);
        pushPoint(decorData, delta, -1.0f, -1.0f);
        pushPoint(decorData, delta, 1.0f, -1.0f);
        /* xz plane center lines */
        pushPoint(decorData, -1.0f, -1.0f, -delta);
        pushPoint(decorData, -1.0f, 1.0f, -delta);
        pushPoint(decorData, -1.0f, -1.0f, delta);
        pushPoint(decorData, -1.0f, 1.0f, delta);
        pushPoint(decorData, -1.0f, -delta, -1.0f);
        pushPoint(decorData, -1.0f, -delta, 1.0f);
        pushPoint(decorData, -1.0f, delta, -1.0f);
        pushPoint(decorData, -1.0f, delta, 1.0f);
        /* yz plane center lines */
        pushPoint(decorData, -1.0f, 1.0f, -delta);
        pushPoint(decorData, 1.0f, 1.0f, -delta);
        pushPoint(decorData, -1.0f, 1.0f, delta);
        pushPoint(decorData, 1.0f, 1.0f, delta);
        pushPoint(decorData, -delta, 1.0f, -1.0f);
        pushPoint(decorData, -delta, 1.0f, 1.0f);
        pushPoint(decorData, delta, 1.0f, -1.0f);
        pushPoint(decorData, delta, 1.0f, 1.0f);
    }

    /* check if decoration VBO has been already used(case where
     * tick marks are being changed from default(21) */
    if (mDecorVBO != 0) glDeleteBuffers(1, &mDecorVBO);

    /* create vbo that has the border and axis data */
    mDecorVBO = createBuffer<float>(GL_ARRAY_BUFFER, decorData.size(),
                                    &(decorData.front()), GL_STATIC_DRAW);
    CheckGL("End chart3d_impl::generateChartData");
}

void chart3d_impl::generateTickLabels() {
    /* remove all the tick text markers that were generated
     * by default during the base class(chart) creation and
     * update the text markers based on the new axes limits*/
    mXText.clear();
    mYText.clear();
    mZText.clear();

    // By default chart's axes labels show numbers in
    // fixed floating point format, unless the users requests
    // for any other format explicitly. However, if the string
    // representation of the range of data of given axis exceeds
    // certain length, the numbers are converted to scientific notation.
    // Z Axis label format
    if (toString(std::fabs(mZMax - mZMin), mZLabelFormat).length() > 5 &&
        std::regex_search(mZLabelFormat, PRINTF_FIXED_FLOAT_RE)) {
        mZLabelFormat = std::string("%.2e");
    }
    // Y Axis label format
    if (toString(std::fabs(mYMax - mYMin), mYLabelFormat).length() > 5 &&
        std::regex_search(mYLabelFormat, PRINTF_FIXED_FLOAT_RE)) {
        mYLabelFormat = std::string("%.2e");
    }
    // X Axis label format
    if (toString(std::fabs(mXMax - mXMin), mXLabelFormat).length() > 5 &&
        std::regex_search(mXLabelFormat, PRINTF_FIXED_FLOAT_RE)) {
        mXLabelFormat = std::string("%.2e");
    }

    float xstep = getTickStepSize(mXMin, mXMax);
    float ystep = getTickStepSize(mYMin, mYMax);
    float zstep = getTickStepSize(mZMin, mZMax);
    float xmid  = (mXMax + mXMin) / 2.0f;
    float ymid  = (mYMax + mYMin) / 2.0f;
    float zmid  = (mZMax + mZMin) / 2.0f;

    int ticksLeft = getNumTicksC2E();

    /* push tick points for z axis */
    mZText.push_back(toString(zmid, mZLabelFormat));
    size_t maxZLabelWidth = 0;
    for (int i = 1; i <= ticksLeft; i++) {
        std::string temp = toString(zmid + i * -zstep, mZLabelFormat);
        mZText.push_back(temp);
        maxZLabelWidth = std::max(maxZLabelWidth, temp.length());

        temp = toString(zmid + i * zstep, mZLabelFormat);
        mZText.push_back(temp);
        maxZLabelWidth = std::max(maxZLabelWidth, temp.length());
    }

    /* push tick points for y axis */
    mYText.push_back(toString(ymid, mYLabelFormat));
    for (int i = 1; i <= ticksLeft; i++) {
        mYText.push_back(toString(ymid + i * -ystep, mYLabelFormat));
        mYText.push_back(toString(ymid + i * ystep, mYLabelFormat));
    }

    /* push tick points for x axis */
    mXText.push_back(toString(xmid, mXLabelFormat));
    for (int i = 1; i <= ticksLeft; i++) {
        mXText.push_back(toString(xmid + i * -xstep, mXLabelFormat));
        mXText.push_back(toString(xmid + i * xstep, mXLabelFormat));
    }
}

chart3d_impl::chart3d_impl()
    : AbstractChart(0.0933f, 0.03701f, 0.1077f, 0.0085f) {
    generateChartData();
    generateTickLabels();
}

void chart3d_impl::render(const int pWindowId, const int pX, const int pY,
                          const int pVPW, const int pVPH,
                          const glm::mat4& pView, const glm::mat4& pOrient) {
    static const glm::mat4 VIEW =
        glm::lookAt(glm::vec3(-1.0f, 0.5f, 1.0f), glm::vec3(1.0f, -1.0f, -1.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f));
    static const glm::mat4 PROJECTION =
        glm::ortho(-1.64f, 1.64f, -1.64f, 1.64f, -0.001f, 1000.f);
    static const glm::mat4 MODEL =
        glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(0, 1, 0)) *
        glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(1, 0, 0));
    static const glm::mat4 PV  = PROJECTION * VIEW;
    static const glm::mat4 PVM = PV * MODEL;

    CheckGL("Being chart3d_impl::renderChart");

    float lgap = getLeftMargin(pVPW) + getTickSize() / 2.0f;
    float bgap = getBottomMargin(pVPH) + getTickSize() / 2.0f;
    float w    = pVPW - (lgap + getRightMargin(pVPW));
    float h    = pVPH - (bgap + getTopMargin(pVPH));

    float offset_x = getLeftMargin(pVPW) / pVPW;
    float offset_y = getBottomMargin(pVPH) / pVPH;
    float scale_x  = w / pVPW;
    float scale_y  = h / pVPH;

    glm::mat4 trans =
        glm::translate(glm::scale(glm::mat4(1), glm::vec3(scale_x, scale_y, 1)),
                       glm::vec3(offset_x, offset_y, 0));

    trans = trans * PVM;

    /* draw grid */
    chart3d_impl::bindResources(pWindowId);
    mBorderProgram.bind();
    glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE,
                       glm::value_ptr(trans));
    glUniform4fv(mBorderUniformColorIndex, 1, GRAY);
    glDrawArrays(GL_LINES, 6 + 3 * mTickCount, 3 * (8 * mTickCount - 16));
    mBorderProgram.unbind();
    chart3d_impl::unbindResources();

    glEnable(GL_SCISSOR_TEST);
    glScissor(GLint(pX + lgap), GLint(pY + bgap), GLsizei(w), GLsizei(h));

    glm::mat4 renderableMat = PROJECTION * pView * VIEW;

    /* render all the renderables */
    for (auto renderable : mRenderables) {
        renderable->setRanges(mXMin, mXMax, mYMin, mYMax, mZMin, mZMax);
        renderable->render(pWindowId, pX, pY, pVPW, pVPH, renderableMat,
                           pOrient);
    }

    glDisable(GL_SCISSOR_TEST);

    /* Draw borders */
    chart3d_impl::bindResources(pWindowId);

    mBorderProgram.bind();
    glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE,
                       glm::value_ptr(trans));
    glUniform4fv(mBorderUniformColorIndex, 1, BLACK);
    glDrawArrays(GL_LINES, 0, 6);
    mBorderProgram.unbind();

    /* bind the sprite shader program to
     * draw ticks on x and y axes */
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize((GLfloat)getTickSize());
    mSpriteProgram.bind();

    glUniform4fv(mSpriteUniformTickcolorIndex, 1, BLACK);
    glUniformMatrix4fv(mSpriteUniformMatIndex, 1, GL_FALSE,
                       glm::value_ptr(trans));
    /* Draw tick marks on z axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 1);
    glDrawArrays(GL_POINTS, 6, mTickCount);
    /* Draw tick marks on y axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 0);
    glDrawArrays(GL_POINTS, 6 + mTickCount, mTickCount);
    /* Draw tick marks on x axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 0);
    glDrawArrays(GL_POINTS, 6 + (2 * mTickCount), mTickCount);

    mSpriteProgram.unbind();
    glPointSize((GLfloat)1);
    glDisable(GL_PROGRAM_POINT_SIZE);

    chart3d_impl::unbindResources();

    const int trgtFntSize = calcTrgtFntSize(w, h);

    renderTickLabels(pWindowId, uint32_t(w), uint32_t(h), mZText, trgtFntSize,
                     trans, 0);
    renderTickLabels(pWindowId, uint32_t(w), uint32_t(h), mYText, trgtFntSize,
                     trans, mTickCount);
    renderTickLabels(pWindowId, uint32_t(w), uint32_t(h), mXText, trgtFntSize,
                     trans, 2 * mTickCount);

    auto& fonter = getChartFont();
    fonter->setOthro2D(int(w), int(h));

    float pos[2];

    /* render chart axes titles */
    if (!mZTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(-1.0f, -1.0f, 0.0f, 1.0f);

        pos[0] = float(trgtFntSize);
        pos[1] = h * (res.y / res.w + 1.0f) / 2.0f;

        fonter->render(pWindowId, pos, BLACK, mZTitle.c_str(), trgtFntSize,
                       true);
    }

    if (!mYTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(1.0f, 0.0f, -1.0f, 1.0f);

        pos[0] = w * (res.x / res.w + 1.0f) / 2.0f;
        pos[1] = h * (res.y / res.w + 1.0f) / 2.0f;

        pos[1] -= (4.0f * trgtFntSize);

        fonter->render(pWindowId, pos, BLACK, mYTitle.c_str(), trgtFntSize);
    }
    if (!mXTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(0.0f, -1.0f, -1.0f, 1.0f);

        pos[0] = w * (res.x / res.w + 1.0f) / 2.0f;
        pos[1] = h * (res.y / res.w + 1.0f) / 2.0f;

        pos[0] -= (mXTitle.length() * trgtFntSize);
        pos[1] -= (3.0f * trgtFntSize);

        fonter->render(pWindowId, pos, BLACK, mXTitle.c_str(), trgtFntSize);
    }

    CheckGL("End chart3d_impl::renderChart");
}

}  // namespace opengl
}  // namespace forge
