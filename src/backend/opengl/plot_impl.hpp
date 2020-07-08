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
#include <fg/defines.h>
#include <shader_program.hpp>

#include <cstdint>
#include <map>

namespace forge {
namespace opengl {

class plot_impl : public AbstractRenderable {
   protected:
    uint32_t mDimension;
    float mMarkerSize;
    /* plot points characteristics */
    uint32_t mNumPoints;
    forge::dtype mDataType;
    forge::MarkerType mMarkerType;
    forge::PlotType mPlotType;
    bool mIsPVROn;
    /* OpenGL Objects */
    ShaderProgram mPlotProgram;
    ShaderProgram mMarkerProgram;
    uint32_t mRBO;
    size_t mRBOSize;
    /* shader variable index locations */
    uint32_t mPlotMatIndex;
    uint32_t mPlotPVCOnIndex;
    uint32_t mPlotPVAOnIndex;
    uint32_t mPlotUColorIndex;
    uint32_t mPlotRangeIndex;
    uint32_t mPlotPointIndex;
    uint32_t mPlotColorIndex;
    uint32_t mPlotAlphaIndex;

    uint32_t mMarkerPVCOnIndex;
    uint32_t mMarkerPVAOnIndex;
    uint32_t mMarkerPVROnIndex;
    uint32_t mMarkerTypeIndex;
    uint32_t mMarkerColIndex;
    uint32_t mMarkerMatIndex;
    uint32_t mMarkerPSizeIndex;
    uint32_t mMarkerPointIndex;
    uint32_t mMarkerColorIndex;
    uint32_t mMarkerAlphaIndex;
    uint32_t mMarkerRadiiIndex;

    std::map<int, uint32_t> mVAOMap;

    /* bind and unbind helper functions
     * for rendering resources */
    void bindResources(const int pWindowId);
    void unbindResources() const;

    virtual glm::mat4 computeTransformMat(const glm::mat4& pView,
                                          const glm::mat4& pOrient);

    virtual void
    bindDimSpecificUniforms();  // has to be called only after shaders are bound

   public:
    plot_impl(const uint32_t pNumPoints, const forge::dtype pDataType,
              const forge::PlotType pPlotType,
              const forge::MarkerType pMarkerType, const int pDimension = 3);
    ~plot_impl();

    void setMarkerSize(const float pMarkerSize);

    uint32_t markers();
    size_t markersSizes() const;

    virtual void render(const int pWindowId, const int pX, const int pY,
                        const int pVPW, const int pVPH, const glm::mat4 &pView,
                        const glm::mat4 &pOrient);
};

class plot2d_impl : public plot_impl {
   protected:
    glm::mat4 computeTransformMat(const glm::mat4& pView,
                                  const glm::mat4& pOrient) override;

    void bindDimSpecificUniforms()
        override;  // has to be called only after shaders are bound

   public:
    plot2d_impl(const uint32_t pNumPoints, const forge::dtype pDataType,
                const forge::PlotType pPlotType,
                const forge::MarkerType pMarkerType)
        : plot_impl(pNumPoints, pDataType, pPlotType, pMarkerType, 2) {}
};

}  // namespace opengl
}  // namespace forge
