/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <fg/defines.h>
#include <common.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include <map>

namespace forge
{
namespace opengl
{

class plot_impl : public AbstractRenderable {
    protected:
        GLuint            mDimension;
        GLfloat           mMarkerSize;
        GLuint            mNumPoints;
        forge::dtype      mDataType;
        GLenum            mGLType;
        forge::MarkerType mMarkerType;
        forge::PlotType   mPlotType;
        bool              mIsPVROn;
        /* OpenGL Objects */
        ShaderProgram mPlotProgram;
        ShaderProgram mMarkerProgram;
        GLuint        mRBO;
        size_t        mRBOSize;
        /* shader variable index locations */
        GLuint mPlotMatIndex;
        GLuint mPlotPVCOnIndex;
        GLuint mPlotPVAOnIndex;
        GLuint mPlotUColorIndex;
        GLuint mPlotRangeIndex;
        GLuint mPlotPointIndex;
        GLuint mPlotColorIndex;
        GLuint mPlotAlphaIndex;
        GLuint mMarkerPVCOnIndex;
        GLuint mMarkerPVAOnIndex;
        GLuint mMarkerPVROnIndex;
        GLuint mMarkerTypeIndex;
        GLuint mMarkerColIndex;
        GLuint mMarkerMatIndex;
        GLuint mMarkerPSizeIndex;
        GLuint mMarkerPointIndex;
        GLuint mMarkerColorIndex;
        GLuint mMarkerAlphaIndex;
        GLuint mMarkerRadiiIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;

        // cal only after shaders are bound
        virtual void bindDimSpecificUniforms();

    public:
        plot_impl(const uint pNumPoints, const forge::dtype pDataType,
                  const forge::PlotType pPlotType,
                  const forge::MarkerType pMarkerType,
                  const int pDimension=3);
        ~plot_impl();
        void setMarkerSize(const float pMarkerSize);
        uint markers();
        size_t markersSizes() const;
        void render(const int pWindowId, const int pX, const int pY,
                    const int pVPW, const int pVPH,
                    const glm::mat4 &pView, const glm::mat4 &pModel);
};

class plot2d_impl : public plot_impl {
    protected:
        // cal only after shaders are bound
        void bindDimSpecificUniforms() override;

    public:
        plot2d_impl(const uint pNumPoints, const forge::dtype pDataType,
                    const forge::PlotType pPlotType,
                    const forge::MarkerType pMarkerType)
            : plot_impl(pNumPoints, pDataType, pPlotType, pMarkerType, 2) {}
};

}
}
