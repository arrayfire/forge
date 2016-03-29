/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/histogram.h>
#include <fg/window.h>

#include <common.hpp>
#include <err_opengl.hpp>
#include <histogram_impl.hpp>
#include <shader_headers/histogram_vs.hpp>
#include <shader_headers/histogram_fs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

using namespace std;

namespace opengl
{

void histogram_impl::bindResources(const int pWindowId)
{
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        CheckGL("Begin histogram_impl::bindResources");
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach histogram bar vertices
        glEnableVertexAttribArray(mPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, screenQuadVBO(pWindowId));
        glVertexAttribPointer(mPointIndex, 2, GL_FLOAT, GL_FALSE, 0, 0);
        // attach histogram frequencies
        glEnableVertexAttribArray(mFreqIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(mFreqIndex, 1, mGLType, GL_FALSE, 0, 0);
        glVertexAttribDivisor(mFreqIndex, 1);
        // attach histogram bar colors
        glEnableVertexAttribArray(mColorIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mCBO);
        glVertexAttribPointer(mColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisor(mColorIndex, 1);
        // attach histogram bar alphas
        glEnableVertexAttribArray(mAlphaIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mABO);
        glVertexAttribPointer(mAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glVertexAttribDivisor(mAlphaIndex, 1);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
        CheckGL("End histogram_impl::bindResources");
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void histogram_impl::unbindResources() const
{
    glBindVertexArray(0);
}

histogram_impl::histogram_impl(const uint pNBins, const fg::dtype pDataType)
 :  mDataType(pDataType), mGLType(dtype2gl(mDataType)), mNBins(pNBins),
    mProgram(0), mYMaxIndex(-1), mNBinsIndex(-1), mMatIndex(-1), mPointIndex(-1),
    mFreqIndex(-1), mColorIndex(-1), mAlphaIndex(-1), mPVCIndex(-1), mPVAIndex(-1),
    mBColorIndex(-1)
{
    CheckGL("Begin histogram_impl::histogram_impl");
    mIsPVCOn = false;
    mIsPVAOn = false;

    setColor(0.8f, 0.6f, 0.0f, 1.0f);
    mLegend  = std::string("");

    mProgram = initShaders(glsl::histogram_vs.c_str(), glsl::histogram_fs.c_str());

    mYMaxIndex   = glGetUniformLocation(mProgram, "ymax"     );
    mNBinsIndex  = glGetUniformLocation(mProgram, "nbins"    );
    mMatIndex    = glGetUniformLocation(mProgram, "transform");
    mPVCIndex    = glGetUniformLocation(mProgram, "isPVCOn"  );
    mPVAIndex    = glGetUniformLocation(mProgram, "isPVAOn"  );
    mBColorIndex = glGetUniformLocation(mProgram, "barColor" );
    mPointIndex  = glGetAttribLocation (mProgram, "point"    );
    mFreqIndex   = glGetAttribLocation (mProgram, "freq"     );
    mColorIndex  = glGetAttribLocation (mProgram, "color"    );
    mAlphaIndex  = glGetAttribLocation (mProgram, "alpha"    );

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
        default: fg::TypeError("histogram_impl::histogram_impl", __LINE__, 1, mDataType);
    }
#undef HIST_CREATE_BUFFERS

    CheckGL("End histogram_impl::histogram_impl");
}

histogram_impl::~histogram_impl()
{
    CheckGL("Begin histogram_impl::~histogram_impl");
    for (auto it = mVAOMap.begin(); it!=mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mVBO);
    glDeleteBuffers(1, &mCBO);
    glDeleteBuffers(1, &mABO);
    glDeleteProgram(mProgram);
    CheckGL("End histogram_impl::~histogram_impl");
}

void histogram_impl::render(const int pWindowId,
                       const int pX, const int pY, const int pVPW, const int pVPH,
                       const glm::mat4& pTransform)
{
    CheckGL("Begin histogram_impl::render");
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_SCISSOR_TEST);

    glScissor(pX, pY, pVPW, pVPH);
    glUseProgram(mProgram);

    glUniform1f(mYMaxIndex, mRange[3]);
    glUniform1f(mNBinsIndex, (GLfloat)mNBins);
    glUniformMatrix4fv(mMatIndex, 1, GL_FALSE, glm::value_ptr(pTransform));
    glUniform1i(mPVCIndex, mIsPVCOn);
    glUniform1i(mPVAIndex, mIsPVAOn);
    glUniform4fv(mBColorIndex, 1, mColor);

    /* render a rectangle for each bin. Same
     * rectangle is scaled and translated accordingly
     * for each bin. OpenGL instanced rendering is used to do it.*/
    histogram_impl::bindResources(pWindowId);
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, mNBins);
    histogram_impl::unbindResources();

    glUseProgram(0);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    CheckGL("End histogram_impl::render");
}

}
