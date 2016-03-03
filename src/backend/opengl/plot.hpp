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
#include <shader_headers/marker2d_vs.hpp>
#include <shader_headers/marker_fs.hpp>
#include <shader_headers/histogram_fs.hpp>
#include <shader_headers/plot3_vs.hpp>
#include <shader_headers/plot3_fs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include <map>

namespace opengl
{

class plot_impl : public AbstractRenderable {
    protected:
        GLuint    mDimension;
        GLfloat   mMarkerSize;
        /* plot points characteristics */
        GLuint    mNumPoints;
        fg::dtype mDataType;
        GLenum    mGLType;
        fg::MarkerType mMarkerType;
        fg::PlotType   mPlotType;
        bool      mIsPVROn;
        /* OpenGL Objects */
        GLuint    mPlotProgram;
        GLuint    mMarkerProgram;
        GLuint    mRBO;
        size_t    mRBOSize;
        /* shader variable index locations */
        GLuint    mPlotMatIndex;
        GLuint    mPlotPVCOnIndex;
        GLuint    mPlotPVAOnIndex;
        GLuint    mPlotUColorIndex;
        GLuint    mPlotRangeIndex;
        GLuint    mPlotPointIndex;
        GLuint    mPlotColorIndex;
        GLuint    mPlotAlphaIndex;

        GLuint    mMarkerPVCOnIndex;
        GLuint    mMarkerPVAOnIndex;
        GLuint    mMarkerPVROnIndex;
        GLuint    mMarkerTypeIndex;
        GLuint    mMarkerColIndex;
        GLuint    mMarkerMatIndex;
        GLuint    mMarkerPSizeIndex;
        GLuint    mMarkerPointIndex;
        GLuint    mMarkerColorIndex;
        GLuint    mMarkerAlphaIndex;
        GLuint    mMarkerRadiiIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;

        virtual void computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput,
                                         const int pX, const int pY,
                                         const int pVPW, const int pVPH);
        virtual void bindDimSpecificUniforms(); // has to be called only after shaders are bound

    public:
        plot_impl(const uint pNumPoints, const fg::dtype pDataType,
                  const fg::PlotType pPlotType, const fg::MarkerType pMarkerType,
                  const int pDimension=3);
        ~plot_impl();

        void setMarkerSize(const float pMarkerSize);

        GLuint markers();
        size_t markersSizes() const;

        virtual void render(const int pWindowId,
                            const int pX, const int pY, const int pVPW, const int pVPH,
                            const glm::mat4& pTransform);
};

class plot2d_impl : public plot_impl {
    protected:
        void computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput,
                                 const int pX, const int pY,
                                 const int pVPW, const int pVPH) override;
        void bindDimSpecificUniforms() override; // has to be called only after shaders are bound

    public:
        plot2d_impl(const uint pNumPoints, const fg::dtype pDataType,
                    const fg::PlotType pPlotType, const fg::MarkerType pMarkerType)
            : plot_impl(pNumPoints, pDataType, pPlotType, pMarkerType, 2) {}
};

}
