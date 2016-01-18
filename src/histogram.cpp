/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common.hpp>
#include <fg/window.h>
#include <fg/histogram.h>
#include <histogram.hpp>
#include <shader_headers/histogram_vs.hpp>
#include <shader_headers/histogram_fs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

using namespace std;

namespace internal
{

void hist_impl::bindResources(const int pWindowId)
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
        glBindBuffer(GL_ARRAY_BUFFER, screenQuadVBO(pWindowId));
        glVertexAttribPointer(mPointIndex, 2, GL_FLOAT, GL_FALSE, 0, 0);
        // attach histogram frequencies
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(mFreqIndex, 1, mGLType, GL_FALSE, 0, 0);
        glVertexAttribDivisor(mFreqIndex, 1);
        // attach histogram bar colors
        glBindBuffer(GL_ARRAY_BUFFER, mCBO);
        glVertexAttribPointer(mColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisor(mColorIndex, 1);
        // attach histogram bar alphas
        glBindBuffer(GL_ARRAY_BUFFER, mABO);
        glVertexAttribPointer(mAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisor(mAlphaIndex, 1);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void hist_impl::unbindResources() const
{
    glVertexAttribDivisor(mFreqIndex, 0);
    glBindVertexArray(0);
}

hist_impl::hist_impl(const uint pNBins, const fg::dtype pDataType)
 :  mDataType(pDataType), mGLType(dtype2gl(mDataType)), mNBins(pNBins),
    mIsPVCOn(0), mProgram(0), mYMaxIndex(-1), mNBinsIndex(-1),
    mMatIndex(-1), mPointIndex(-1), mFreqIndex(-1), mColorIndex(-1),
    mAlphaIndex(-1), mPVCIndex(-1), mBColorIndex(-1)
{
    mColor[0] = 0.8f;
    mColor[1] = 0.6f;
    mColor[2] = 0.0f;
    mColor[3] = 1.0f;
    mLegend   = std::string("");

    CheckGL("Begin hist_impl::hist_impl");
    mProgram = initShaders(glsl::histogram_vs.c_str(), glsl::histogram_fs.c_str());

    mYMaxIndex   = glGetUniformLocation(mProgram, "ymax"     );
    mNBinsIndex  = glGetUniformLocation(mProgram, "nbins"    );
    mMatIndex    = glGetUniformLocation(mProgram, "transform");
    mPointIndex  = glGetAttribLocation (mProgram, "point"    );
    mFreqIndex   = glGetAttribLocation (mProgram, "freq"     );
    mColorIndex  = glGetUniformLocation(mProgram, "color"    );
    mAlphaIndex  = glGetAttribLocation (mProgram, "alpha"    );
    mPVCIndex    = glGetUniformLocation(mProgram, "isPVCOn"  );
    mBColorIndex = glGetAttribLocation (mProgram, "barColor" );

    mVBOSize = mNBins;
    mCBOSize = 3*mVBOSize;
    mABOSize = mNBins;

#define HIST_CREATE_BUFFERS(type)   \
    mVBO = createBuffer<type>(GL_ARRAY_BUFFER, mVBOSize, NULL, GL_DYNAMIC_DRAW);  \
    mCBO = createBuffer<float>(GL_ARRAY_BUFFER, mCBOSize, NULL, GL_DYNAMIC_DRAW); \
    mABO = createBuffer<float>(GL_ARRAY_BUFFER, mABOSize, NULL, GL_DYNAMIC_DRAW); \
    mVBOSize *= sizeof(type);   \
    mCBOSize *= sizeof(float);  \
    mABOSize *= sizeof(float);

    switch(mGLType) {
        case GL_FLOAT          : HIST_CREATE_BUFFERS(float) ; break;
        case GL_INT            : HIST_CREATE_BUFFERS(int)   ; break;
        case GL_UNSIGNED_INT   : HIST_CREATE_BUFFERS(uint)  ; break;
        case GL_SHORT          : HIST_CREATE_BUFFERS(short) ; break;
        case GL_UNSIGNED_SHORT : HIST_CREATE_BUFFERS(ushort); break;
        case GL_UNSIGNED_BYTE  : HIST_CREATE_BUFFERS(float) ; break;
        default: fg::TypeError("hist_impl::hist_impl", __LINE__, 1, mDataType);
    }
#undef HIST_CREATE_BUFFERS

    CheckGL("End hist_impl::hist_impl");
}

hist_impl::~hist_impl()
{
    CheckGL("Begin hist_impl::~hist_impl");
    for (auto it = mVAOMap.begin(); it!=mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mVBO);
    glDeleteBuffers(1, &mCBO);
    glDeleteBuffers(1, &mABO);
    glDeleteProgram(mProgram);
    CheckGL("End hist_impl::~hist_impl");
}

void hist_impl::render(const int pWindowId,
                       const int pX, const int pY, const int pVPW, const int pVPH,
                       const glm::mat4& pTransform)
{
    CheckGL("Begin hist_impl::render");
    glScissor(pX, pY, pVPW, pVPH);
    glEnable(GL_SCISSOR_TEST);
    glUseProgram(mProgram);

    glUniform1f(mYMaxIndex, mRange[3]);
    glUniform1f(mNBinsIndex, (GLfloat)mNBins);
    glUniformMatrix4fv(mMatIndex, 1, GL_FALSE, glm::value_ptr(pTransform));
    glUniform1i(mPVCIndex, mIsPVCOn);
    glUniform4fv(mColorIndex, 1, mColor);

    /* render a rectangle for each bin. Same
     * rectangle is scaled and translated accordingly
     * for each bin. This is done by OpenGL feature of
     * instanced rendering */
    hist_impl::bindResources(pWindowId);
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, mNBins);
    hist_impl::unbindResources();

    glUseProgram(0);
    glDisable(GL_SCISSOR_TEST);
    CheckGL("End hist_impl::render");
}

}

namespace fg
{

Histogram::Histogram(const uint pNBins, const dtype pDataType)
{
    mValue = new internal::_Histogram(pNBins, pDataType);
}

Histogram::Histogram(const Histogram& pOther)
{
    mValue = new internal::_Histogram(*pOther.get());
}

Histogram::~Histogram()
{
    delete mValue;
}

void Histogram::setColor(const Color pColor)
{
    float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
    float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
    float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
    float a = (((int) pColor       ) & 0xFF ) / 255.f;
    mValue->setColor(r, g, b, a);
}

void Histogram::setColor(const float pRed, const float pGreen,
                         const float pBlue, const float pAlpha)
{
    mValue->setColor(pRed, pGreen, pBlue, pAlpha);
}

void Histogram::setLegend(const std::string pLegend)
{
    mValue->setLegend(pLegend);
}

uint Histogram::vertices() const
{
    return mValue->vbo();
}

uint Histogram::colors() const
{
    return mValue->cbo();
}

uint Histogram::alphas() const
{
    return mValue->abo();
}

uint Histogram::verticesSize() const
{
    return (uint)mValue->vboSize();
}

uint Histogram::colorsSize() const
{
    return (uint)mValue->cboSize();
}

uint Histogram::alphasSize() const
{
    return (uint)mValue->aboSize();
}

internal::_Histogram* Histogram::get() const
{
    return mValue;
}

}
