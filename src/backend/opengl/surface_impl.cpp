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
#include <shader_headers/marker_fs.hpp>
#include <shader_headers/plot3_fs.hpp>
#include <shader_headers/plot3_vs.hpp>
#include <surface_impl.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

using namespace std;
using namespace forge::common;

namespace forge {
namespace opengl {

void generateGridIndices(std::vector<unsigned int>& indices,
                         unsigned short rows, unsigned short cols) {
    const int numDegens        = 2 * (rows - 2);
    const int verticesPerStrip = 2 * cols;

    // reserve the size of vector
    indices.reserve(verticesPerStrip + numDegens);

    for (int r = 0; r < (rows - 1); ++r) {
        if (r > 0) {
            // repeat first vertex for degenerate triangle
            indices.push_back(r * rows);
        }

        for (int c = 0; c < cols; ++c) {
            // One part of the strip
            indices.push_back(r * rows + c);
            indices.push_back((r + 1) * rows + c);
        }

        if (r < (rows - 2)) {
            // repeat last vertex for degenerate triangle
            indices.push_back(((r + 1) * rows) + (cols - 1));
        }
    }
}

void surface_impl::bindResources(const int pWindowId) {
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach plot vertices
        glEnableVertexAttribArray(mSurfPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(mSurfPointIndex, 3, dtype2gl(mDataType), GL_FALSE,
                              0, 0);
        glEnableVertexAttribArray(mSurfColorIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mCBO);
        glVertexAttribPointer(mSurfColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(mSurfAlphaIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mABO);
        glVertexAttribPointer(mSurfAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        // attach indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void surface_impl::unbindResources() const { glBindVertexArray(0); }

glm::mat4 surface_impl::computeTransformMat(const glm::mat4& pView,
                                            const glm::mat4& pOrient) {
    static const glm::mat4 MODEL =
        glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(0, 1, 0)) *
        glm::rotate(glm::mat4(1.0f), -glm::radians(90.f), glm::vec3(1, 0, 0));

    float xRange = mRange[1] - mRange[0];
    float yRange = mRange[3] - mRange[2];
    float zRange = mRange[5] - mRange[4];
    // set scale to zero if input is constant array
    // otherwise compute scale factor by standard equation
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

void surface_impl::renderGraph(const int pWindowId,
                               const glm::mat4& transform) {
    CheckGL("Begin surface_impl::renderGraph");

    mSurfProgram.bind();

    glUniformMatrix4fv(mSurfMatIndex, 1, GL_FALSE, glm::value_ptr(transform));
    glUniform2fv(mSurfRangeIndex, 3, mRange);
    glUniform1i(mSurfPVCIndex, mIsPVCOn);
    glUniform1i(mSurfPVAIndex, mIsPVAOn);
    glUniform1i(mSurfAssistDrawFlagIndex, false);
    glUniform4fv(mSurfUniformColorIndex, 1, mColor);

    bindResources(pWindowId);
    glDrawElements(GL_TRIANGLE_STRIP, GLsizei(mIBOSize), GL_UNSIGNED_INT,
                   (void*)0);
    unbindResources();
    mSurfProgram.unbind();

    if (mMarkerType != FG_MARKER_NONE) {
        glEnable(GL_PROGRAM_POINT_SIZE);
        mMarkerProgram.bind();

        glUniformMatrix4fv(mMarkerMatIndex, 1, GL_FALSE,
                           glm::value_ptr(transform));
        glUniform1i(mMarkerPVCIndex, mIsPVCOn);
        glUniform1i(mMarkerPVAIndex, mIsPVAOn);
        glUniform1i(mMarkerTypeIndex, mMarkerType);
        glUniform4fv(mMarkerColIndex, 1, mColor);

        bindResources(pWindowId);
        glDrawElements(GL_POINTS, GLsizei(mIBOSize), GL_UNSIGNED_INT, (void*)0);
        unbindResources();

        mMarkerProgram.unbind();
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
    CheckGL("End surface_impl::renderGraph");
}

surface_impl::surface_impl(unsigned pNumXPoints, unsigned pNumYPoints,
                           forge::dtype pDataType,
                           forge::MarkerType pMarkerType)
    : mNumXPoints(pNumXPoints)
    , mNumYPoints(pNumYPoints)
    , mDataType(pDataType)
    , mMarkerType(pMarkerType)
    , mIBO(0)
    , mIBOSize(0)
    , mMarkerProgram(glsl::plot3_vs.c_str(), glsl::marker_fs.c_str())
    , mSurfProgram(glsl::plot3_vs.c_str(), glsl::plot3_fs.c_str())
    , mMarkerMatIndex(-1)
    , mMarkerPointIndex(-1)
    , mMarkerColorIndex(-1)
    , mMarkerAlphaIndex(-1)
    , mMarkerPVCIndex(-1)
    , mMarkerPVAIndex(-1)
    , mMarkerTypeIndex(-1)
    , mMarkerColIndex(-1)
    , mSurfMatIndex(-1)
    , mSurfRangeIndex(-1)
    , mSurfPointIndex(-1)
    , mSurfColorIndex(-1)
    , mSurfAlphaIndex(-1)
    , mSurfPVCIndex(-1)
    , mSurfPVAIndex(-1)
    , mSurfUniformColorIndex(-1)
    , mSurfAssistDrawFlagIndex(-1) {
    CheckGL("Begin surface_impl::surface_impl");
    setColor(0.9f, 0.5f, 0.6f, 1.0f);

    mMarkerMatIndex   = mMarkerProgram.getUniformLocation("transform");
    mMarkerPVCIndex   = mMarkerProgram.getUniformLocation("isPVCOn");
    mMarkerPVAIndex   = mMarkerProgram.getUniformLocation("isPVAOn");
    mMarkerTypeIndex  = mMarkerProgram.getUniformLocation("marker_type");
    mMarkerColIndex   = mMarkerProgram.getUniformLocation("marker_color");
    mMarkerPointIndex = mMarkerProgram.getAttributeLocation("point");
    mMarkerColorIndex = mMarkerProgram.getAttributeLocation("color");
    mMarkerAlphaIndex = mMarkerProgram.getAttributeLocation("alpha");

    mSurfMatIndex            = mSurfProgram.getUniformLocation("transform");
    mSurfRangeIndex          = mSurfProgram.getUniformLocation("minmaxs");
    mSurfPVCIndex            = mSurfProgram.getUniformLocation("isPVCOn");
    mSurfPVAIndex            = mSurfProgram.getUniformLocation("isPVAOn");
    mSurfPointIndex          = mSurfProgram.getAttributeLocation("point");
    mSurfColorIndex          = mSurfProgram.getAttributeLocation("color");
    mSurfAlphaIndex          = mSurfProgram.getAttributeLocation("alpha");
    mSurfUniformColorIndex   = mSurfProgram.getUniformLocation("lineColor");
    mSurfAssistDrawFlagIndex = mSurfProgram.getUniformLocation("isAssistDraw");

    unsigned totalPoints = mNumXPoints * mNumYPoints;

    mVBOSize = 3 * totalPoints;
    mCBOSize = 3 * totalPoints;
    mABOSize = totalPoints;
#define SURF_CREATE_BUFFERS(type)                                              \
    mVBO =                                                                     \
        createBuffer<type>(GL_ARRAY_BUFFER, mVBOSize, NULL, GL_DYNAMIC_DRAW);  \
    mCBO =                                                                     \
        createBuffer<float>(GL_ARRAY_BUFFER, mCBOSize, NULL, GL_DYNAMIC_DRAW); \
    mABO =                                                                     \
        createBuffer<float>(GL_ARRAY_BUFFER, mABOSize, NULL, GL_DYNAMIC_DRAW); \
    mVBOSize *= sizeof(type);                                                  \
    mCBOSize *= sizeof(float);                                                 \
    mABOSize *= sizeof(float);

    switch (dtype2gl(pDataType)) {
        case GL_FLOAT: SURF_CREATE_BUFFERS(float); break;
        case GL_INT: SURF_CREATE_BUFFERS(int); break;
        case GL_UNSIGNED_INT: SURF_CREATE_BUFFERS(uint32_t); break;
        case GL_SHORT: SURF_CREATE_BUFFERS(short); break;
        case GL_UNSIGNED_SHORT: SURF_CREATE_BUFFERS(uint16_t); break;
        case GL_UNSIGNED_BYTE: SURF_CREATE_BUFFERS(float); break;
        default: TYPE_ERROR(1, pDataType);
    }

#undef SURF_CREATE_BUFFERS

    std::vector<unsigned int> indices;

    generateGridIndices(indices, mNumXPoints, mNumYPoints);

    mIBOSize = indices.size();

    mIBO = createBuffer<uint32_t>(GL_ELEMENT_ARRAY_BUFFER, mIBOSize,
                                  indices.data(), GL_STATIC_DRAW);

    CheckGL("End surface_impl::surface_impl");
}

surface_impl::~surface_impl() {
    for (auto it = mVAOMap.begin(); it != mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mIBO);
}

void surface_impl::render(const int pWindowId, const int pX, const int pY,
                          const int pVPW, const int pVPH,
                          const glm::mat4& pView, const glm::mat4& pOrient) {
    CheckGL("Begin surface_impl::render");
    // FIXME: even when per vertex alpha is enabled
    // primitives of transparent object should be sorted
    // from the furthest to closest primitive
    if (mIsPVAOn) {
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    renderGraph(pWindowId, computeTransformMat(pView, pOrient));

    if (mIsPVAOn) {
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }
    CheckGL("End surface_impl::render");
}

void scatter3_impl::renderGraph(const int pWindowId,
                                const glm::mat4& transform) {
    if (mMarkerType != FG_MARKER_NONE) {
        glEnable(GL_PROGRAM_POINT_SIZE);
        mMarkerProgram.bind();

        glUniformMatrix4fv(mMarkerMatIndex, 1, GL_FALSE,
                           glm::value_ptr(transform));
        glUniform1i(mMarkerPVCIndex, mIsPVCOn);
        glUniform1i(mMarkerPVAIndex, mIsPVAOn);
        glUniform1i(mMarkerTypeIndex, mMarkerType);
        glUniform4fv(mMarkerColIndex, 1, mColor);

        bindResources(pWindowId);
        glDrawElements(GL_POINTS, GLsizei(mIBOSize), GL_UNSIGNED_INT, (void*)0);
        unbindResources();

        mMarkerProgram.unbind();
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
}

}  // namespace opengl
}  // namespace forge
