/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <abstract_renderable.hpp>
#include <shader_program.hpp>

#include <cstdint>
#include <map>

namespace forge {
namespace opengl {

class surface_impl : public AbstractRenderable {
   protected:
    /* plot points characteristics */
    uint32_t mNumXPoints;
    uint32_t mNumYPoints;
    forge::dtype mDataType;
    forge::MarkerType mMarkerType;
    /* OpenGL Objects */
    uint32_t mIBO;
    size_t mIBOSize;
    ShaderProgram mMarkerProgram;
    ShaderProgram mSurfProgram;
    /* shared variable index locations */
    uint32_t mMarkerMatIndex;
    uint32_t mMarkerPointIndex;
    uint32_t mMarkerColorIndex;
    uint32_t mMarkerAlphaIndex;
    uint32_t mMarkerPVCIndex;
    uint32_t mMarkerPVAIndex;
    uint32_t mMarkerTypeIndex;
    uint32_t mMarkerColIndex;

    uint32_t mSurfMatIndex;
    uint32_t mSurfRangeIndex;
    uint32_t mSurfPointIndex;
    uint32_t mSurfColorIndex;
    uint32_t mSurfAlphaIndex;
    uint32_t mSurfPVCIndex;
    uint32_t mSurfPVAIndex;
    uint32_t mSurfUniformColorIndex;
    uint32_t mSurfAssistDrawFlagIndex;

    std::map<int, uint32_t> mVAOMap;

    /* bind and unbind helper functions
     * for rendering resources */
    void bindResources(const int pWindowId);
    void unbindResources() const;
    glm::mat4 computeTransformMat(const glm::mat4& pView,
                                  const glm::mat4& pOrient);
    virtual void renderGraph(const int pWindowId, const glm::mat4& transform);

   public:
    surface_impl(const uint32_t pNumXpoints, const uint32_t pNumYpoints,
                 const forge::dtype pDataType,
                 const forge::MarkerType pMarkerType);
    ~surface_impl();

    void render(const int pWindowId, const int pX, const int pY, const int pVPW,
                const int pVPH, const glm::mat4& pView,
                const glm::mat4& pOrient);

    inline void usePerVertexColors(const bool pFlag = true) {
        mIsPVCOn = pFlag;
    }

    inline void usePerVertexAlphas(const bool pFlag = true) {
        mIsPVAOn = pFlag;
    }

    bool isRotatable() const { return true; }
};

class scatter3_impl : public surface_impl {
   private:
    void renderGraph(const int pWindowId, const glm::mat4& transform);

   public:
    scatter3_impl(const uint32_t pNumXPoints, const uint32_t pNumYPoints,
                  const forge::dtype pDataType,
                  const forge::MarkerType pMarkerType = FG_MARKER_NONE)
        : surface_impl(pNumXPoints, pNumYPoints, pDataType, pMarkerType) {}
};

}  // namespace opengl
}  // namespace forge
