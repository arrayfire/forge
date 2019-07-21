/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <pie_impl.hpp>

#include <common/err_handling.hpp>
#include <gl_helpers.hpp>
#include <shader_headers/histogram_fs.hpp>
#include <shader_headers/pie_gs.hpp>
#include <shader_headers/pie_vs.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

using namespace std;

namespace forge {
namespace opengl {

void pie_impl::bindResources(const int pWindowId) {
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach vertices
        glEnableVertexAttribArray(mSectorRangeIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(mSectorRangeIndex, 2, dtype2gl(mDataType),
                              GL_FALSE, 0, 0);
        // attach colors
        glEnableVertexAttribArray(mSectorColorIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mCBO);
        glVertexAttribPointer(mSectorColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        // attach alphas
        glEnableVertexAttribArray(mSectorAlphaIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mABO);
        glVertexAttribPointer(mSectorAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void pie_impl::unbindResources() const { glBindVertexArray(0); }

pie_impl::pie_impl(const uint pNSectors, const forge::dtype pDataType)
    : mNSectors(pNSectors)
    , mDataType(pDataType)
    , mSectorProgram(glsl::pie_vs.c_str(), glsl::histogram_fs.c_str(),
                     glsl::pie_gs.c_str())
    , mSectorRangeIndex(-1)
    , mSectorColorIndex(-1)
    , mSectorAlphaIndex(-1)
    , mMaxValueIndex(-1)
    , mSectorPVMatIndex(-1)
    , mSectorPVCOnIndex(-1)
    , mSectorPVAOnIndex(-1) {
    CheckGL("Begin pie_impl::pie_impl");

    setColor(0, 1, 0, 1);

    mVBOSize = 2 * mNSectors;
    mCBOSize = 3 * mNSectors;
    mABOSize = mNSectors;

    mSectorRangeIndex = mSectorProgram.getAttributeLocation("range");
    mSectorColorIndex = mSectorProgram.getAttributeLocation("color");
    mSectorAlphaIndex = mSectorProgram.getAttributeLocation("alpha");

    mMaxValueIndex    = mSectorProgram.getUniformLocation("maxValue");
    mSectorPVMatIndex = mSectorProgram.getUniformLocation("viewMat");

    mSectorPVCOnIndex = mSectorProgram.getUniformLocation("isPVCOn");
    mSectorPVAOnIndex = mSectorProgram.getUniformLocation("isPVAOn");

#define PLOT_CREATE_BUFFERS(type)                                              \
    mVBO =                                                                     \
        createBuffer<type>(GL_ARRAY_BUFFER, mVBOSize, NULL, GL_DYNAMIC_DRAW);  \
    mCBO =                                                                     \
        createBuffer<float>(GL_ARRAY_BUFFER, mCBOSize, NULL, GL_DYNAMIC_DRAW); \
    mABO =                                                                     \
        createBuffer<float>(GL_ARRAY_BUFFER, mABOSize, NULL, GL_DYNAMIC_DRAW); \
    mVBOSize *= sizeof(type);                                                  \
    mCBOSize *= sizeof(float);                                                 \
    mABOSize *= sizeof(float);

    switch (dtype2gl(mDataType)) {
        case GL_FLOAT: PLOT_CREATE_BUFFERS(float); break;
        case GL_INT: PLOT_CREATE_BUFFERS(int); break;
        case GL_UNSIGNED_INT: PLOT_CREATE_BUFFERS(uint); break;
        case GL_SHORT: PLOT_CREATE_BUFFERS(short); break;
        case GL_UNSIGNED_SHORT: PLOT_CREATE_BUFFERS(ushort); break;
        case GL_UNSIGNED_BYTE: PLOT_CREATE_BUFFERS(float); break;
        default: TYPE_ERROR(1, mDataType);
    }
#undef PLOT_CREATE_BUFFERS
    CheckGL("End pie_impl::pie_impl");
}

pie_impl::~pie_impl() {
    for (auto it = mVAOMap.begin(); it != mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
}

void pie_impl::render(const int pWindowId, const int pX, const int pY,
                      const int pVPW, const int pVPH, const glm::mat4& pView,
                      const glm::mat4& pOrient) {
    CheckGL("Begin pie_impl::render");
    if (mIsPVAOn) {
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    mSectorProgram.bind();

    glUniformMatrix4fv(mSectorPVMatIndex, 1, GL_FALSE, glm::value_ptr(pView));
    glUniform1f(mMaxValueIndex, mRange[3]);
    glUniform1i(mSectorPVCOnIndex, mIsPVCOn);
    glUniform1i(mSectorPVAOnIndex, mIsPVAOn);

    pie_impl::bindResources(pWindowId);
    glDrawArrays(GL_POINTS, 0, mNSectors);
    pie_impl::unbindResources();

    mSectorProgram.unbind();

    if (mIsPVAOn) {
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }
    CheckGL("End pie_impl::render");
}

}  // namespace opengl
}  // namespace forge
