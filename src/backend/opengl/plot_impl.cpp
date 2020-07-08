/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_handling.hpp>
#include <gl_helpers.hpp>
#include <plot_impl.hpp>
#include <shader_headers/histogram_fs.hpp>
#include <shader_headers/marker2d_vs.hpp>
#include <shader_headers/marker_fs.hpp>
#include <shader_headers/plot3_fs.hpp>
#include <shader_headers/plot3_vs.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

using namespace forge::common;
using namespace glm;
using namespace std;

namespace forge {
namespace opengl {

void plot_impl::bindResources(const int pWindowId) {
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach vertices
        glEnableVertexAttribArray(mPlotPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(mPlotPointIndex, mDimension, dtype2gl(mDataType),
                              GL_FALSE, 0, 0);
        // attach colors
        glEnableVertexAttribArray(mPlotColorIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mCBO);
        glVertexAttribPointer(mPlotColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        // attach alphas
        glEnableVertexAttribArray(mPlotAlphaIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mABO);
        glVertexAttribPointer(mPlotAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        // attach radii
        glEnableVertexAttribArray(mMarkerRadiiIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mRBO);
        glVertexAttribPointer(mMarkerRadiiIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void plot_impl::unbindResources() const { glBindVertexArray(0); }

glm::mat4 plot_impl::computeTransformMat(const glm::mat4& pView,
                                         const glm::mat4& pOrient) {
    static const glm::mat4 MODEL =
        glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(0, 1, 0)) *
        glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(1, 0, 0));

    float xRange = mRange[1] - mRange[0];
    float yRange = mRange[3] - mRange[2];
    float zRange = mRange[5] - mRange[4];

    float xDataScale = std::abs(xRange) < 1.0e-3 ? 1.0f : 2 / (xRange);
    float yDataScale = std::abs(yRange) < 1.0e-3 ? 1.0f : 2 / (yRange);
    float zDataScale = std::abs(zRange) < 1.0e-3 ? 1.0f : 2 / (zRange);

    float xDataOffset = (-mRange[0] * xDataScale);
    float yDataOffset = (-mRange[2] * yDataScale);
    float zDataOffset = (-mRange[4] * zDataScale);

    glm::vec3 scaleVector(xDataScale, yDataScale, zDataScale);

    glm::vec3 shiftVector =
        glm::vec3(-1 + xDataOffset, -1 + yDataOffset, -1 + zDataOffset);

    return pView * pOrient * MODEL *
           glm::scale(glm::translate(IDENTITY, shiftVector), scaleVector);
}

void plot_impl::bindDimSpecificUniforms() {
    glUniform2fv(mPlotRangeIndex, 3, mRange);
}

plot_impl::plot_impl(const uint32_t pNumPoints, const forge::dtype pDataType,
                     const forge::PlotType pPlotType,
                     const forge::MarkerType pMarkerType, const int pD)
    : mDimension(pD)
    , mMarkerSize(12)
    , mNumPoints(pNumPoints)
    , mDataType(pDataType)
    , mMarkerType(pMarkerType)
    , mPlotType(pPlotType)
    , mIsPVROn(false)
    , mPlotProgram(
          pD == 2 ? glsl::marker2d_vs.c_str() : glsl::plot3_vs.c_str(),
          pD == 2 ? glsl::histogram_fs.c_str() : glsl::plot3_fs.c_str())
    , mMarkerProgram(
          pD == 2 ? glsl::marker2d_vs.c_str() : glsl::plot3_vs.c_str(),
          glsl::marker_fs.c_str())
    , mRBO(-1)
    , mPlotMatIndex(-1)
    , mPlotPVCOnIndex(-1)
    , mPlotPVAOnIndex(-1)
    , mPlotUColorIndex(-1)
    , mPlotRangeIndex(-1)
    , mPlotPointIndex(-1)
    , mPlotColorIndex(-1)
    , mPlotAlphaIndex(-1)
    , mMarkerPVCOnIndex(-1)
    , mMarkerPVAOnIndex(-1)
    , mMarkerTypeIndex(-1)
    , mMarkerColIndex(-1)
    , mMarkerMatIndex(-1)
    , mMarkerPointIndex(-1)
    , mMarkerColorIndex(-1)
    , mMarkerAlphaIndex(-1)
    , mMarkerRadiiIndex(-1) {
    CheckGL("Begin plot_impl::plot_impl");

    setColor(0, 1, 0, 1);

    if (mDimension == 2) {
        mPlotUColorIndex = mPlotProgram.getUniformLocation("barColor");
        mVBOSize         = 2 * mNumPoints;
    } else {
        mPlotRangeIndex = mPlotProgram.getUniformLocation("minmaxs");
        mVBOSize        = 3 * mNumPoints;
    }

    mCBOSize = 3 * mNumPoints;
    mABOSize = mNumPoints;
    mRBOSize = mNumPoints;

    mPlotMatIndex   = mPlotProgram.getUniformLocation("transform");
    mPlotPVCOnIndex = mPlotProgram.getUniformLocation("isPVCOn");
    mPlotPVAOnIndex = mPlotProgram.getUniformLocation("isPVAOn");
    mPlotPointIndex = mPlotProgram.getAttributeLocation("point");
    mPlotColorIndex = mPlotProgram.getAttributeLocation("color");
    mPlotAlphaIndex = mPlotProgram.getAttributeLocation("alpha");

    mMarkerMatIndex   = mMarkerProgram.getUniformLocation("transform");
    mMarkerPVCOnIndex = mMarkerProgram.getUniformLocation("isPVCOn");
    mMarkerPVAOnIndex = mMarkerProgram.getUniformLocation("isPVAOn");
    mMarkerPVROnIndex = mMarkerProgram.getUniformLocation("isPVROn");
    mMarkerTypeIndex  = mMarkerProgram.getUniformLocation("marker_type");
    mMarkerColIndex   = mMarkerProgram.getUniformLocation("marker_color");
    mMarkerPSizeIndex = mMarkerProgram.getUniformLocation("psize");
    mMarkerPointIndex = mMarkerProgram.getAttributeLocation("point");
    mMarkerColorIndex = mMarkerProgram.getAttributeLocation("color");
    mMarkerAlphaIndex = mMarkerProgram.getAttributeLocation("alpha");
    mMarkerRadiiIndex = mMarkerProgram.getAttributeLocation("pointsize");

#define PLOT_CREATE_BUFFERS(type)                                              \
    mVBO =                                                                     \
        createBuffer<type>(GL_ARRAY_BUFFER, mVBOSize, NULL, GL_DYNAMIC_DRAW);  \
    mCBO =                                                                     \
        createBuffer<float>(GL_ARRAY_BUFFER, mCBOSize, NULL, GL_DYNAMIC_DRAW); \
    mABO =                                                                     \
        createBuffer<float>(GL_ARRAY_BUFFER, mABOSize, NULL, GL_DYNAMIC_DRAW); \
    mRBO =                                                                     \
        createBuffer<float>(GL_ARRAY_BUFFER, mRBOSize, NULL, GL_DYNAMIC_DRAW); \
    mVBOSize *= sizeof(type);                                                  \
    mCBOSize *= sizeof(float);                                                 \
    mABOSize *= sizeof(float);                                                 \
    mRBOSize *= sizeof(float);

    switch (dtype2gl(mDataType)) {
        case GL_FLOAT: PLOT_CREATE_BUFFERS(float); break;
        case GL_INT: PLOT_CREATE_BUFFERS(int); break;
        case GL_UNSIGNED_INT: PLOT_CREATE_BUFFERS(uint32_t); break;
        case GL_SHORT: PLOT_CREATE_BUFFERS(short); break;
        case GL_UNSIGNED_SHORT: PLOT_CREATE_BUFFERS(uint16_t); break;
        case GL_UNSIGNED_BYTE: PLOT_CREATE_BUFFERS(float); break;
        default: TYPE_ERROR(1, mDataType);
    }
#undef PLOT_CREATE_BUFFERS
    CheckGL("End plot_impl::plot_impl");
}

plot_impl::~plot_impl() {
    for (auto it = mVAOMap.begin(); it != mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mRBO);
}

void plot_impl::setMarkerSize(const float pMarkerSize) {
    mMarkerSize = pMarkerSize;
}

uint32_t plot_impl::markers() {
    mIsPVROn = true;
    return mRBO;
}

size_t plot_impl::markersSizes() const { return mRBOSize; }

void plot_impl::render(const int pWindowId, const int pX, const int pY,
                       const int pVPW, const int pVPH, const glm::mat4& pView,
                       const glm::mat4& pOrient) {
    CheckGL("Begin plot_impl::render");
    if (mIsPVAOn) {
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    glm::mat4 viewModelMatrix = this->computeTransformMat(pView, pOrient);

    if (mPlotType == FG_PLOT_LINE) {
        mPlotProgram.bind();

        this->bindDimSpecificUniforms();
        glUniformMatrix4fv(mPlotMatIndex, 1, GL_FALSE,
                           glm::value_ptr(viewModelMatrix));
        glUniform1i(mPlotPVCOnIndex, mIsPVCOn);
        glUniform1i(mPlotPVAOnIndex, mIsPVAOn);

        plot_impl::bindResources(pWindowId);
        glDrawArrays(GL_LINE_STRIP, 0, mNumPoints);
        plot_impl::unbindResources();

        mPlotProgram.unbind();
    }

    if (mMarkerType != FG_MARKER_NONE) {
        glEnable(GL_PROGRAM_POINT_SIZE);
        mMarkerProgram.bind();

        glUniformMatrix4fv(mMarkerMatIndex, 1, GL_FALSE,
                           glm::value_ptr(viewModelMatrix));
        glUniform1i(mMarkerPVCOnIndex, mIsPVCOn);
        glUniform1i(mMarkerPVAOnIndex, mIsPVAOn);
        glUniform1i(mMarkerPVROnIndex, mIsPVROn);
        glUniform1i(mMarkerTypeIndex, mMarkerType);
        glUniform4fv(mMarkerColIndex, 1, mColor);
        glUniform1f(mMarkerPSizeIndex, mMarkerSize);

        plot_impl::bindResources(pWindowId);
        glDrawArrays(GL_POINTS, 0, mNumPoints);
        plot_impl::unbindResources();

        mMarkerProgram.unbind();
        glDisable(GL_PROGRAM_POINT_SIZE);
    }

    if (mIsPVAOn) {
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }
    CheckGL("End plot_impl::render");
}

glm::mat4 plot2d_impl::computeTransformMat(const glm::mat4& pView,
                                           const glm::mat4& /*pOrient*/) {
    float xRange = mRange[1] - mRange[0];
    float yRange = mRange[3] - mRange[2];

    float xDataScale = std::abs(xRange) < 1.0e-3 ? 1.0f : 2 / (xRange);
    float yDataScale = std::abs(yRange) < 1.0e-3 ? 1.0f : 2 / (yRange);

    float xDataOffset = (-mRange[0] * xDataScale);
    float yDataOffset = (-mRange[2] * yDataScale);

    glm::vec3 scaleVector(xDataScale, yDataScale, 1);
    glm::vec3 shiftVector = glm::vec3(-1 + xDataOffset, -1 + yDataOffset, 0);

    return pView *
           glm::scale(glm::translate(IDENTITY, shiftVector), scaleVector);
}

void plot2d_impl::bindDimSpecificUniforms() {
    glUniform4fv(mPlotUColorIndex, 1, mColor);
}

}  // namespace opengl
}  // namespace forge
