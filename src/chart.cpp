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
"in vec3 point;\n"
"uniform mat4 transform;\n"
"void main(void) {\n"
"   gl_Position = transform * vec4(point.xyz, 1);\n"
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

namespace internal
{

/********************* BEGIN-AbstractChart *********************/

void AbstractChart::renderTickLabels(int pWindowId, unsigned w, unsigned h,
        std::vector<std::string> &texts,
        glm::mat4 &transformation, int coor_offset,
        bool useZoffset)
{
    auto &fonter = getChartFont();
    fonter->setOthro2D(int(w), int(h));

    float pos[2];
    for (StringIter it = texts.begin(); it!=texts.end(); ++it) {
        int idx = int(it - texts.begin());
        glm::vec4 p = glm::vec4(mTickTextX[idx+coor_offset],
                                mTickTextY[idx+coor_offset],
                                (useZoffset ? mTickTextZ[idx+coor_offset] : 0), 1);
        glm::vec4 res = transformation * p;

        /* convert text position from [-1,1] range to
         * [0, 1) range and then offset horizontally
         * to compensate for margins and ticksize */
        pos[0] = w*(res.x/res.w+1.0f)/2.0f;
        pos[1] = h*(res.y/res.w+1.0f)/2.0f;

        /* offset based on text size to align
         * text center with tick mark position */
        if(coor_offset < mTickCount) {
            pos[0] -= ((CHART2D_FONT_SIZE*it->length()/2.0f));
        }else if(coor_offset >= mTickCount && coor_offset < 2*mTickCount) {
            pos[0] -= ((CHART2D_FONT_SIZE*it->length()/2.0f));
            pos[1] -= ((CHART2D_FONT_SIZE));
        }else {
            pos[1] -= ((CHART2D_FONT_SIZE));
        }
        fonter->render(pWindowId, pos, WHITE, it->c_str(), CHART2D_FONT_SIZE);
    }
}

AbstractChart::AbstractChart(int pLeftMargin, int pRightMargin, int pTopMargin, int pBottomMargin)
    : mTickCount(9), mTickSize(10),
      mLeftMargin(pLeftMargin), mRightMargin(pRightMargin),
      mTopMargin(pTopMargin), mBottomMargin(pBottomMargin),
      mXMax(1), mXMin(0), mYMax(1), mYMin(0), mZMax(1), mZMin(0),
      mXTitle("X-Axis"), mYTitle("Y-Axis"), mZTitle("Z-Axis"),
      mDecorVBO(-1), mBorderProgram(-1), mSpriteProgram(-1),
      mBorderAttribPointIndex(-1), mBorderUniformColorIndex(-1),
      mBorderUniformMatIndex(-1), mSpriteUniformMatIndex(-1),
      mSpriteUniformTickcolorIndex(-1), mSpriteUniformTickaxisIndex(-1)
{
    CheckGL("Begin AbstractChart::AbstractChart");
    /* load font Vera font for chart text
     * renderings, below function actually returns a constant
     * reference to font object used by Chart objects, we are
     * calling it here just to make sure required font glyphs
     * are loaded into the shared Font object */
    getChartFont();

    mBorderProgram = initShaders(gChartVertexShaderSrc, gChartFragmentShaderSrc);
    mSpriteProgram = initShaders(gChartVertexShaderSrc, gChartSpriteFragmentShaderSrc);

    mBorderAttribPointIndex      = glGetAttribLocation (mBorderProgram, "point");
    mBorderUniformColorIndex     = glGetUniformLocation(mBorderProgram, "color");
    mBorderUniformMatIndex       = glGetUniformLocation(mBorderProgram, "transform");

    mSpriteUniformTickcolorIndex = glGetUniformLocation(mSpriteProgram, "tick_color");
    mSpriteUniformMatIndex       = glGetUniformLocation(mSpriteProgram, "transform");
    mSpriteUniformTickaxisIndex  = glGetUniformLocation(mSpriteProgram, "isYAxis");

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
    glDeleteProgram(mBorderProgram);
    glDeleteProgram(mSpriteProgram);
    CheckGL("End AbstractChart::~AbstractChart");
}

void AbstractChart::setAxesLimits(float pXmax, float pXmin,
                                  float pYmax, float pYmin,
                                  float pZmax, float pZmin)
{
    mXMax = pXmax; mXMin = pXmin;
    mYMax = pYmax; mYMin = pYmin;
    mZMax = pZmax; mZMin = pZmin;

    /*
     * Once the axes ranges are known, we can generate
     * tick labels. The following functions is a pure
     * virtual function and has to be implemented by the
     * derived class
     */
    generateTickLabels();
}

void AbstractChart::setAxesTitles(const char* pXTitle, const char* pYTitle, const char* pZTitle)
{
    mXTitle = std::string(pXTitle);
    mYTitle = std::string(pYTitle);
    mZTitle = std::string(pZTitle);
}

float AbstractChart::xmax() const { return mXMax; }
float AbstractChart::xmin() const { return mXMin; }
float AbstractChart::ymax() const { return mYMax; }
float AbstractChart::ymin() const { return mYMin; }
float AbstractChart::zmax() const { return mZMax; }
float AbstractChart::zmin() const { return mZMin; }

/********************* END-AbstractChart *********************/



/********************* BEGIN-Chart2D *********************/

void Chart2D::bindResources(int pWindowId)
{
    CheckGL("Begin Chart2D::bindResources");
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
    CheckGL("End Chart2D::bindResources");
}

void Chart2D::unbindResources() const
{
    glBindVertexArray(0);
}

void Chart2D::pushTicktextCoords(float x, float y, float z)
{
    mTickTextX.push_back(x);
    mTickTextY.push_back(y);
}

void Chart2D::generateChartData()
{
    CheckGL("Begin Chart2D::generateChartData");
    static const float border[8] = { -1, -1, 1, -1, 1, 1, -1, 1 };
    static const int nValues = sizeof(border)/sizeof(float);

    std::vector<float> decorData;
    std::copy(border, border+nValues, std::back_inserter(decorData));

    float step = 2.0f/(mTickCount);
    /* push tick points for y axis:
     * push (0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    int ticksLeft = mTickCount/2;
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

    /* check if decoration VBO has been already used(case where
     * tick marks are being changed from default(21) */
    if (mDecorVBO != 0)
        glDeleteBuffers(1, &mDecorVBO);

    /* create vbo that has the border and axis data */
    mDecorVBO = createBuffer<float>(GL_ARRAY_BUFFER, decorData.size(),
                                    &(decorData.front()), GL_STATIC_DRAW);
    CheckGL("End Chart2D::generateChartData");
}

void Chart2D::generateTickLabels()
{
    /* remove all the tick text markers that were generated
     * by default during the base class(chart) creation and
     * update the text markers based on the new axes limits
     * */
    mXText.clear();
    mYText.clear();
    mZText.clear();

    float xstep = (mXMax-mXMin)/(mTickCount);
    float ystep = (mYMax-mYMin)/(mTickCount);
    float xmid = mXMin + (mXMax-mXMin)/2.0f;
    float ymid = mYMin + (mYMax-mYMin)/2.0f;
    int ticksLeft = mTickCount/2;
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

void Chart2D::renderChart(int pWindowId, int pX, int pY, int pVPW, int pVPH)
{
    CheckGL("Begin Chart2D::renderChart");

    float w = float(pVPW - (mLeftMargin + mRightMargin + mTickSize));
    float h = float(pVPH - (mTopMargin + mBottomMargin + mTickSize));
    float offset_x = (2.0f * (mLeftMargin+mTickSize) + (w - pVPW)) / pVPW;
    float offset_y = (2.0f * (mBottomMargin+mTickSize) + (h - pVPH)) / pVPH;
    float scale_x = w / pVPW;
    float scale_y = h / pVPH;

    Chart2D::bindResources(pWindowId);

    /* bind the plotting shader program  */
    glUseProgram(mBorderProgram);

    /* set uniform attributes of shader
     * for drawing the plot borders */
    glm::mat4 trans = glm::translate(glm::scale(glm::mat4(1),
                                                glm::vec3(scale_x, scale_y, 1)),
                                     glm::vec3(offset_x, offset_y, 0));
    glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    glUniform4fv(mBorderUniformColorIndex, 1, WHITE);

    /* Draw borders */
    glDrawArrays(GL_LINE_LOOP, 0, 4);

    /* reset shader program binding */
    glUseProgram(0);

    /* bind the sprite shader program to
     * draw ticks on x and y axes */
    glPointSize((GLfloat)mTickSize);

    glUseProgram(mSpriteProgram);
    glUniform4fv(mSpriteUniformTickcolorIndex, 1, WHITE);
    glUniformMatrix4fv(mSpriteUniformMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    /* Draw tick marks on y axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 1);
    glDrawArrays(GL_POINTS, 4, mTickCount);
    /* Draw tick marks on x axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 0);
    glDrawArrays(GL_POINTS, 4+mTickCount, mTickCount);

    glUseProgram(0);
    glPointSize(1);
    Chart2D::unbindResources();

    renderTickLabels(pWindowId, int(w), int(h), mYText, trans, 0, false);
    renderTickLabels(pWindowId, int(w), int(h), mXText, trans, mTickCount, false);

    auto &fonter = getChartFont();
    fonter->setOthro2D(int(w), int(h));
    float pos[2];
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

    CheckGL("End Chart2D::renderChart");
}

/********************* END-Chart2D *********************/



/********************* BEGIN-Chart3D *********************/

void Chart3D::bindResources(int pWindowId)
{
    CheckGL("Begin Chart3D::bindResources");
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
    CheckGL("End Chart3D::bindResources");
}

void Chart3D::unbindResources() const
{
    glBindVertexArray(0);
}

void Chart3D::pushTicktextCoords(float x, float y, float z)
{
    mTickTextX.push_back(x);
    mTickTextY.push_back(y);
    mTickTextZ.push_back(z);
}

void Chart3D::generateChartData()
{
    CheckGL("Begin Chart3D::generateChartData");
    static const float border[] = { -1, -1, -1,  1, -1, -1,  -1, -1, -1,  -1, 1, -1,  -1, 1, -1,  -1, 1, 1 };
    static const int nValues = sizeof(border)/sizeof(float);

    std::vector<float> decorData;
    std::copy(border, border+nValues, std::back_inserter(decorData));

    float step = 2.0f/(mTickCount);

    /* push tick points for z axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, -1.0f, 1.0f, 0.0f);
    pushTicktextCoords(-1.0f, 1.0f, 0.0f);
    mZText.push_back(toString(0));

    int ticksLeft = mTickCount/2;
    for(int i=1; i<=ticksLeft; ++i) {
        /* (0, -1] to [-1, -1] */
        float neg = i*-step;
        pushPoint(decorData, -1.0f, 1.0f, neg);
        /* push tick marks */
        pushTicktextCoords(-1.0f, 1.0f, neg);
        /* push tick text label */
        mZText.push_back(toString(neg));

        /* (0, -1] to [1, -1] */
        float pos = i*step;
        pushPoint(decorData, -1.0f, 1.0f, pos);
        /* push tick marks */
        pushTicktextCoords(-1.0f, 1.0f, pos);
        /* push tick text label */
        mZText.push_back(toString(pos));
    }
    /* push tick points for y axis:
     * push (0,0) first followed by
     * [-1, 0) ticks and then
     * (0, 1] ticks  */
    pushPoint(decorData, -1.0f, 0.0f, -1.0f);
    pushTicktextCoords(-1.0f, 0.0f, -1.0f);
    mYText.push_back(toString(0));

    for(int i=1; i<=ticksLeft; ++i) {
        /* [-1, 0) to [-1, -1] */
        float neg = i*-step;
        float pos = i*step;
        pushPoint(decorData, -1.0f, neg, -1.0f);
        pushTicktextCoords(-1.0f, pos, -1.0f);
        mYText.push_back(toString(neg));

        /* [-1, 0) to [-1, 1] */
        pushPoint(decorData, -1.0f, pos, -1.0f);
        pushTicktextCoords(-1.0f, neg, -1.0f);
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

    /* check if decoration VBO has been already used(case where
     * tick marks are being changed from default(21) */
    if (mDecorVBO != 0)
        glDeleteBuffers(1, &mDecorVBO);

    /* create vbo that has the border and axis data */
    mDecorVBO = createBuffer<float>(GL_ARRAY_BUFFER, decorData.size(),
                                    &(decorData.front()), GL_STATIC_DRAW);
    CheckGL("End Chart3D::generateChartData");
}

void Chart3D::generateTickLabels()
{
    /* remove all the tick text markers that were generated
     * by default during the base class(chart) creation and
     * update the text markers based on the new axes limits*/
    mXText.clear();
    mYText.clear();
    mZText.clear();

    float xstep = (mXMax-mXMin)/(mTickCount);
    float ystep = (mYMax-mYMin)/(mTickCount);
    float zstep = (mZMax-mZMin)/(mTickCount);
    float xmid = mXMin + (mXMax-mXMin)/2.0f;
    float ymid = mYMin + (mYMax-mYMin)/2.0f;
    float zmid = mZMin + (mZMax-mZMin)/2.0f;
    int ticksLeft = mTickCount/2;
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

void Chart3D::renderChart(int pWindowId, int pX, int pY, int pVPW, int pVPH)
{
    CheckGL("Being Chart3D::renderChart");
    float w = float(pVPW - (mLeftMargin + mRightMargin + mTickSize));
    float h = float(pVPH - (mTopMargin + mBottomMargin + mTickSize));

    Chart3D::bindResources(pWindowId);

    /* bind the plotting shader program  */
    glUseProgram(mBorderProgram);

    /* set uniform attributes of shader
     * for drawing the plot borders */
    glm::mat4 model = glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(1,0,0)) * glm::scale(glm::mat4(1.f), glm::vec3(1.0f, 1.0f, 1.0f));
    glm::mat4 view = glm::lookAt(glm::vec3(-1,0.5f,1.0f), glm::vec3(1,-1,-1),glm::vec3(0,1,0));
    glm::mat4 projection = glm::ortho(-2.f, 2.f, -2.f, 2.f, -1.1f, 10.f);
    glm::mat4 mvp = projection * view * model;

    glm::mat4 trans = mvp;
    glUniformMatrix4fv(mBorderUniformMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    glUniform4fv(mBorderUniformColorIndex, 1, WHITE);

    /* Draw borders */
    glDrawArrays(GL_LINES, 0, 6);

    /* reset shader program binding */
    glUseProgram(0);

    /* bind the sprite shader program to
     * draw ticks on x and y axes */
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize((GLfloat)mTickSize);

    glUseProgram(mSpriteProgram);
    glUniform4fv(mSpriteUniformTickcolorIndex, 1, WHITE);
    glUniformMatrix4fv(mSpriteUniformMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    /* Draw tick marks on z axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 1);
    glDrawArrays(GL_POINTS, 6, mTickCount);
    /* Draw tick marks on y axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 0);
    glDrawArrays(GL_POINTS, 6 + mTickCount, mTickCount);
    /* Draw tick marks on x axis */
    glUniform1i(mSpriteUniformTickaxisIndex, 0);
    glDrawArrays(GL_POINTS, 6 + (2*mTickCount), mTickCount);

    glUseProgram(0);
    glPointSize(1);
    glDisable(GL_PROGRAM_POINT_SIZE);
    Chart3D::unbindResources();

    renderTickLabels(pWindowId, w, h, mZText, trans, 0);
    renderTickLabels(pWindowId, w, h, mYText, trans, mTickCount);
    renderTickLabels(pWindowId, w, h, mXText, trans, 2*mTickCount);

    auto &fonter = getChartFont();
    fonter->setOthro2D(int(w), int(h));
    float pos[2];
    /* render chart axes titles */
    if (!mZTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(-1.0f, 1.0f, 0.0f, 1.0f);
        pos[0] = w*(res.x/res.w+1.0f)/2.0f;
        pos[1] = h*(res.y/res.w+1.0f)/2.0f;
        pos[0] -= 6*(mTickSize * (w/pVPW));
        pos[1] += mZTitle.length()/2 * CHART2D_FONT_SIZE;
        fonter->render(pWindowId, pos, WHITE, mZTitle.c_str(), CHART2D_FONT_SIZE, true);
    }
    if (!mYTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(-1.0f, 0.0f, -1.0f, 1.0f);
        pos[0] = w*(res.x/res.w+1.0f)/2.0f;
        pos[1] = h*(res.y/res.w+1.0f)/2.0f;
        pos[0] -= 2*(mTickSize * (w/pVPW)) + mYTitle.length()/2 * CHART2D_FONT_SIZE;
        pos[1] -= 3*(mTickSize * (h/pVPH));
        fonter->render(pWindowId, pos, WHITE, mYTitle.c_str(), CHART2D_FONT_SIZE);
    }
    if (!mXTitle.empty()) {
        glm::vec4 res = trans * glm::vec4(0.0f, -1.0f, -1.0f, 1.0f);
        pos[0] = w*(res.x/res.w+1.0f)/2.0f;
        pos[1] = h*(res.y/res.w+1.0f)/2.0f;
        pos[0] += 3*(mTickSize * (w/pVPW));
        pos[1] -= 3*(mTickSize * (h/pVPH));
        fonter->render(pWindowId, pos, WHITE, mXTitle.c_str(), CHART2D_FONT_SIZE);
    }

    CheckGL("End Chart3D::renderChart");
}

}
