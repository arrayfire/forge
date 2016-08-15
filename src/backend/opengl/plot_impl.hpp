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

namespace opengl
{

class plot_impl : public AbstractRenderable {
    protected:
        gl::GLuint     mDimension;
        gl::GLfloat    mMarkerSize;
        /* plot points characteristics */
        gl::GLuint     mNumPoints;
        fg::dtype      mDataType;
        gl::GLenum     mGLType;
        fg::MarkerType mMarkerType;
        fg::PlotType   mPlotType;
        bool           mIsPVROn;
        /* OpenGL Objects */
        ShaderProgram  mPlotProgram;
        ShaderProgram  mMarkerProgram;
        gl::GLuint    mRBO;
        size_t        mRBOSize;
        /* shader variable index locations */
        gl::GLuint    mPlotMatIndex;
        gl::GLuint    mPlotPVCOnIndex;
        gl::GLuint    mPlotPVAOnIndex;
        gl::GLuint    mPlotUColorIndex;
        gl::GLuint    mPlotRangeIndex;
        gl::GLuint    mPlotPointIndex;
        gl::GLuint    mPlotColorIndex;
        gl::GLuint    mPlotAlphaIndex;

        gl::GLuint    mMarkerPVCOnIndex;
        gl::GLuint    mMarkerPVAOnIndex;
        gl::GLuint    mMarkerPVROnIndex;
        gl::GLuint    mMarkerTypeIndex;
        gl::GLuint    mMarkerColIndex;
        gl::GLuint    mMarkerMatIndex;
        gl::GLuint    mMarkerPSizeIndex;
        gl::GLuint    mMarkerPointIndex;
        gl::GLuint    mMarkerColorIndex;
        gl::GLuint    mMarkerAlphaIndex;
        gl::GLuint    mMarkerRadiiIndex;

        std::map<int, gl::GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;

        virtual glm::mat4 computeTransformMat(const glm::mat4 pView, const glm::mat4 pOrient);

        virtual void bindDimSpecificUniforms(); // has to be called only after shaders are bound

    public:
        plot_impl(const uint pNumPoints, const fg::dtype pDataType,
                  const fg::PlotType pPlotType, const fg::MarkerType pMarkerType,
                  const int pDimension=3);
        ~plot_impl();

        void setMarkerSize(const float pMarkerSize);

        uint markers();
        size_t markersSizes() const;

        virtual void render(const int pWindowId,
                            const int pX, const int pY, const int pVPW, const int pVPH,
                            const glm::mat4 &pView, const glm::mat4 &pOrient);
};

class plot2d_impl : public plot_impl {
    protected:
        glm::mat4 computeTransformMat(const glm::mat4 pView, const glm::mat4 pOrient) override;

        void bindDimSpecificUniforms() override; // has to be called only after shaders are bound

    public:
        plot2d_impl(const uint pNumPoints, const fg::dtype pDataType,
                    const fg::PlotType pPlotType, const fg::MarkerType pMarkerType)
            : plot_impl(pNumPoints, pDataType, pPlotType, pMarkerType, 2) {}
};

}
