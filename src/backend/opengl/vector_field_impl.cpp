/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <vector_field_impl.hpp>
#include <shader_headers/vector_field2d_vs.hpp>
#include <shader_headers/vector_field2d_gs.hpp>
#include <shader_headers/vector_field_vs.hpp>
#include <shader_headers/vector_field_gs.hpp>
#include <shader_headers/histogram_fs.hpp>

#include <cmath>

using namespace std;

namespace forge {
namespace opengl {

void vector_field_impl::bindResources(const int pWindowId)
{
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach vertices
        glEnableVertexAttribArray(mFieldPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(
                mFieldPointIndex, mDimension, mGLType, GL_FALSE, 0, 0);
        // attach colors
        glEnableVertexAttribArray(mFieldColorIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mCBO);
        glVertexAttribPointer(mFieldColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        // attach alphas
        glEnableVertexAttribArray(mFieldAlphaIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mABO);
        glVertexAttribPointer(mFieldAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        // attach field directions
        glEnableVertexAttribArray(mFieldDirectionIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mDBO);
        glVertexAttribPointer(
                mFieldDirectionIndex, mDimension, GL_FLOAT, GL_FALSE, 0, 0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void vector_field_impl::unbindResources() const
{
    glBindVertexArray(0);
}

vector_field_impl::vector_field_impl(const uint pNumPoints,
                                     const forge::dtype pDataType, const int pD)
    : mDimension(pD), mNumPoints(pNumPoints), mDataType(pDataType),
    mGLType(dtype2gl(mDataType)),
    mFieldProgram(pD==2 ? glsl::vector_field2d_vs.c_str()
                        : glsl::vector_field_vs.c_str(),
                  glsl::histogram_fs.c_str(),
                  pD==2 ? glsl::vector_field2d_gs.c_str()
                        : glsl::vector_field_gs.c_str()),
    mDBO(-1), mDBOSize(0), mFieldPointIndex(-1),
    mFieldColorIndex(-1), mFieldAlphaIndex(-1), mFieldDirectionIndex(-1),
    mFieldTrMatIndex(-1), mFieldArrowSizeIndex(-1),
    mFieldPVCOnIndex(-1), mFieldPVAOnIndex(-1), mFieldUColorIndex(-1)
{
    CheckGL("Begin vector_field_impl::vector_field_impl");

    setColor(0, 1, 0, 1);

    if (mDimension==2) {
        mVBOSize = 2*mNumPoints;
        mDBOSize = 2*mNumPoints;
    } else {
        mVBOSize = 3*mNumPoints;
        mDBOSize = 3*mNumPoints;
    }

    mCBOSize = 3*mNumPoints;
    mABOSize = mNumPoints;

    mFieldPointIndex  = mFieldProgram.getAttributeLocation("point");
    mFieldColorIndex  = mFieldProgram.getAttributeLocation("color");
    mFieldAlphaIndex  = mFieldProgram.getAttributeLocation("alpha");
    mFieldDirectionIndex = mFieldProgram.getAttributeLocation("direction");

    mFieldTrMatIndex  = mFieldProgram.getUniformLocation("modelViewMat");
    mFieldArrowSizeIndex = mFieldProgram.getUniformLocation("ArrowSize");

    mFieldPVCOnIndex  = mFieldProgram.getUniformLocation("isPVCOn");
    mFieldPVAOnIndex  = mFieldProgram.getUniformLocation("isPVAOn");
    mFieldUColorIndex = mFieldProgram.getUniformLocation("barColor");

#define PLOT_CREATE_BUFFERS(type)   \
        mVBO = createBuffer<type>(GL_ARRAY_BUFFER, mVBOSize, NULL, GL_DYNAMIC_DRAW);    \
        mCBO = createBuffer<float>(GL_ARRAY_BUFFER, mCBOSize, NULL, GL_DYNAMIC_DRAW);   \
        mABO = createBuffer<float>(GL_ARRAY_BUFFER, mABOSize, NULL, GL_DYNAMIC_DRAW);   \
        mDBO = createBuffer<float>(GL_ARRAY_BUFFER, mDBOSize, NULL, GL_DYNAMIC_DRAW);   \
        mVBOSize *= sizeof(type);   \
        mCBOSize *= sizeof(float);  \
        mABOSize *= sizeof(float);  \
        mDBOSize *= sizeof(float);

        switch(mGLType) {
            case GL_FLOAT          : PLOT_CREATE_BUFFERS(float) ; break;
            case GL_INT            : PLOT_CREATE_BUFFERS(int)   ; break;
            case GL_UNSIGNED_INT   : PLOT_CREATE_BUFFERS(uint)  ; break;
            case GL_SHORT          : PLOT_CREATE_BUFFERS(short) ; break;
            case GL_UNSIGNED_SHORT : PLOT_CREATE_BUFFERS(ushort); break;
            case GL_UNSIGNED_BYTE  : PLOT_CREATE_BUFFERS(float) ; break;
            default                : TYPE_ERROR(1, mDataType);
        }
#undef PLOT_CREATE_BUFFERS
        CheckGL("End vector_field_impl::vector_field_impl");
}

vector_field_impl::~vector_field_impl()
{
    for (auto it = mVAOMap.begin(); it!=mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        if (vao) { glDeleteVertexArrays(1, &vao); }
    }
    if (mDBO) { glDeleteBuffers(1, &mDBO); }
}

GLuint vector_field_impl::directions()
{
    return mDBO;
}

size_t vector_field_impl::directionsSize() const
{
    return mDBOSize;
}

void vector_field_impl::render(const int pWindowId,
                       const int pX, const int pY,
                       const int pVPW, const int pVPH,
                       const glm::mat4& pView, const glm::mat4& pModel)
{
    const float ArrowSize = 0.03f;

    CheckGL("Begin vector_field_impl::render");
    if (mIsPVAOn) {
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    auto modelNormMat = getModelTransform();

    auto transform = pView * pModel * modelNormMat;

    mFieldProgram.bind();

    glUniformMatrix4fv(mFieldTrMatIndex, 1, GL_FALSE,
                       glm::value_ptr(transform));
    glUniform1f(mFieldArrowSizeIndex, ArrowSize);
    glUniform1i(mFieldPVCOnIndex, mIsPVCOn);
    glUniform1i(mFieldPVAOnIndex, mIsPVAOn);
    glUniform4fv(mFieldUColorIndex, 1, mColor);

    if (mDimension==3)
        glEnable(GL_CULL_FACE);
    vector_field_impl::bindResources(pWindowId);
    glDrawArrays(GL_POINTS, 0, mNumPoints);
    vector_field_impl::unbindResources();
    if (mDimension==3)
        glDisable(GL_CULL_FACE);

    mFieldProgram.unbind();

    if (mIsPVAOn) {
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }
    CheckGL("End vector_field_impl::render");
}

}
}
