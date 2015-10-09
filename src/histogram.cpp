/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common.hpp>
#include <fg/histogram.h>
#include <histogram.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

using namespace std;

const char *gHistBarVertexShaderSrc =
"#version 330\n"
"in vec2 point;\n"
"in float freq;\n"
"uniform float ymax;\n"
"uniform float nbins;\n"
"uniform mat4 transform;\n"
"void main(void) {\n"
"   float binId = gl_InstanceID;\n"
"   float deltax = 2.0f/nbins;\n"
"   float deltay = 2.0f/ymax;\n"
"   float xcurr = -1.0f + binId * deltax;\n"
"   if (point.x==1) {\n"
"        xcurr  += deltax;\n"
"   }\n"
"   float ycurr = -1.0f;\n"
"   if (point.y==1) {\n"
"       ycurr += deltay * freq;\n"
"   }\n"
"   gl_Position = transform * vec4(xcurr, ycurr, 0, 1);\n"
"}";

const char *gHistBarFragmentShaderSrc =
"#version 330\n"
"uniform vec4 barColor;\n"
"out vec4 outColor;\n"
"void main(void) {\n"
"   outColor = barColor;\n"
"}";

namespace internal
{

void hist_impl::bindResources(int pWindowId)
{
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(mPointIndex);
        glEnableVertexAttribArray(mFreqIndex);
        // attach histogram bar vertices
        glBindBuffer(GL_ARRAY_BUFFER, mDecorVBO);
        glVertexAttribPointer(mPointIndex, 2, GL_FLOAT, GL_FALSE, 0, 0);
        // attach histogram frequencies
        glBindBuffer(GL_ARRAY_BUFFER, mHistogramVBO);
        glVertexAttribPointer(mFreqIndex, 1, mGLType, GL_FALSE, 0, 0);
        glVertexAttribDivisor(mFreqIndex, 1);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void hist_impl::unbindResources() const
{
    glBindVertexArray(0);
    //glVertexAttribDivisor(mFreqIndex, 0);
}

hist_impl::hist_impl(unsigned pNBins, fg::dtype pDataType)
 : Chart2D(), mDataType(pDataType), mGLType(gl_dtype(mDataType)),
   mNBins(pNBins), mHistogramVBO(0), mHistogramVBOSize(0), mHistBarProgram(0),
   mHistBarMatIndex(0), mHistBarColorIndex(0), mHistBarYMaxIndex(0),
   mPointIndex(0), mFreqIndex(0)
{
    CheckGL("Begin hist_impl::hist_impl");
    mHistBarProgram = initShaders(gHistBarVertexShaderSrc, gHistBarFragmentShaderSrc);

    mPointIndex        = glGetAttribLocation (mHistBarProgram, "point");
    mFreqIndex         = glGetAttribLocation (mHistBarProgram, "freq");
    mHistBarColorIndex = glGetUniformLocation(mHistBarProgram, "barColor");
    mHistBarMatIndex   = glGetUniformLocation(mHistBarProgram, "transform");
    mHistBarNBinsIndex = glGetUniformLocation(mHistBarProgram, "nbins");
    mHistBarYMaxIndex  = glGetUniformLocation(mHistBarProgram, "ymax");

    switch(mGLType) {
        case GL_FLOAT:
            mHistogramVBO = createBuffer<float>(GL_ARRAY_BUFFER, mNBins, NULL, GL_DYNAMIC_DRAW);
            mHistogramVBOSize = mNBins*sizeof(float);
            break;
        case GL_INT:
            mHistogramVBO = createBuffer<int>(GL_ARRAY_BUFFER, mNBins, NULL, GL_DYNAMIC_DRAW);
            mHistogramVBOSize = mNBins*sizeof(int);
            break;
        case GL_UNSIGNED_INT:
            mHistogramVBO = createBuffer<unsigned>(GL_ARRAY_BUFFER, mNBins, NULL, GL_DYNAMIC_DRAW);
            mHistogramVBOSize = mNBins*sizeof(unsigned);
            break;
        case GL_SHORT:
            mHistogramVBO = createBuffer<short>(GL_ARRAY_BUFFER, mNBins, NULL, GL_DYNAMIC_DRAW);
            mHistogramVBOSize = mNBins*sizeof(short);
            break;
        case GL_UNSIGNED_SHORT:
            mHistogramVBO = createBuffer<unsigned short>(GL_ARRAY_BUFFER, mNBins, NULL, GL_DYNAMIC_DRAW);
            mHistogramVBOSize = mNBins*sizeof(unsigned short);
            break;
        case GL_UNSIGNED_BYTE:
            mHistogramVBO = createBuffer<unsigned char>(GL_ARRAY_BUFFER, mNBins, NULL, GL_DYNAMIC_DRAW);
            mHistogramVBOSize = mNBins*sizeof(unsigned char);
            break;
        default: fg::TypeError("Plot::Plot", __LINE__, 1, mDataType);
    }
    CheckGL("End hist_impl::hist_impl");
}

hist_impl::~hist_impl()
{
    CheckGL("Begin hist_impl::~hist_impl");
    for (auto it = mVAOMap.begin(); it!=mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mHistogramVBO);
    glDeleteProgram(mHistBarProgram);
    CheckGL("End hist_impl::~hist_impl");
}

void hist_impl::setBarColor(float r, float g, float b)
{
    mBarColor[0] = r;
    mBarColor[1] = g;
    mBarColor[2] = b;
    mBarColor[3] = 1.0f;
}

GLuint hist_impl::vbo() const
{
    return mHistogramVBO;
}

size_t hist_impl::size() const
{
    return mHistogramVBOSize;
}

void hist_impl::render(int pWindowId, int pX, int pY, int pVPW, int pVPH)
{
    float w = float(pVPW - (mLeftMargin+mRightMargin+mTickSize));
    float h = float(pVPH - (mBottomMargin+mTopMargin+mTickSize));
    float offset_x = (2.0f * (mLeftMargin+mTickSize) + (w - pVPW)) / pVPW;
    float offset_y = (2.0f * (mBottomMargin+mTickSize) + (h - pVPH)) / pVPH;
    float scale_x = w / pVPW;
    float scale_y = h / pVPH;

    CheckGL("Begin Histogram::render");
    /* Enavle scissor test to discard anything drawn beyond viewport.
     * Set scissor rectangle to clip fragments outside of viewport */
    glScissor(pX+mLeftMargin+mTickSize, pY+mBottomMargin+mTickSize,
            pVPW - (mLeftMargin+mRightMargin+mTickSize),
            pVPH - (mBottomMargin+mTopMargin+mTickSize));
    glEnable(GL_SCISSOR_TEST);

    glm::mat4 trans = glm::translate(glm::scale(glm::mat4(1),
                                                glm::vec3(scale_x, scale_y, 1)),
                                     glm::vec3(offset_x, offset_y, 0));

    glUseProgram(mHistBarProgram);
    glUniformMatrix4fv(mHistBarMatIndex, 1, GL_FALSE, glm::value_ptr(trans));
    glUniform4fv(mHistBarColorIndex, 1, mBarColor);
    glUniform1f(mHistBarNBinsIndex, (GLfloat)mNBins);
    glUniform1f(mHistBarYMaxIndex, ymax());

    /* render a rectangle for each bin. Same
     * rectangle is scaled and translated accordingly
     * for each bin. This is done by OpenGL feature of
     * instanced rendering */
    hist_impl::bindResources(pWindowId);
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, mNBins);
    hist_impl::unbindResources();

    glUseProgram(0);
    /* Stop clipping */
    glDisable(GL_SCISSOR_TEST);

    renderChart(pWindowId, pX, pY, pVPW, pVPH);
    CheckGL("End Histogram::render");
}

}

namespace fg
{

Histogram::Histogram(unsigned pNBins, fg::dtype pDataType)
{
    value = new internal::_Histogram(pNBins, pDataType);
}

Histogram::Histogram(const Histogram& other)
{
    value = new internal::_Histogram(*other.get());
}

Histogram::~Histogram()
{
    delete value;
}

void Histogram::setBarColor(fg::Color col)
{
    float r = (((int) col >> 24 ) & 0xFF ) / 255.f;
    float g = (((int) col >> 16 ) & 0xFF ) / 255.f;
    float b = (((int) col >> 8  ) & 0xFF ) / 255.f;
    // float a = (((int) col       ) & 0xFF ) / 255.f;
    value->setBarColor(r, g, b);
}

void Histogram::setBarColor(float r, float g, float b)
{
    value->setBarColor(r, g, b);
}

void Histogram::setAxesLimits(float pXmax, float pXmin, float pYmax, float pYmin)
{
    value->setAxesLimits(pXmax, pXmin, pYmax, pYmin);
}

void Histogram::setAxesTitles(const char* pXTitle, const char* pYTitle)
{
    value->setAxesTitles(pXTitle, pYTitle);
}

float Histogram::xmax() const
{
    return value->xmax();
}

float Histogram::xmin() const
{
    return value->xmin();
}

float Histogram::ymax() const
{
    return value->ymax();
}

float Histogram::ymin() const
{
    return value->ymin();
}

unsigned Histogram::vbo() const
{
    return value->vbo();
}

unsigned Histogram::size() const
{
    return (unsigned)value->size();
}

internal::_Histogram* Histogram::get() const
{
    return value;
}

}
