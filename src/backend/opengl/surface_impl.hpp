/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <common.hpp>

#include <glm/glm.hpp>

#include <memory>
#include <map>

namespace forge
{
namespace opengl
{

class surface_impl : public AbstractRenderable {
    protected:
        /* plot points characteristics */
        gl::GLuint    mNumXPoints;
        gl::GLuint    mNumYPoints;
        gl::GLenum    mDataType;
        bool          mIsPVCOn;
        bool          mIsPVAOn;
        forge::MarkerType mMarkerType;
        /* OpenGL Objects */
        gl::GLuint    mIBO;
        size_t        mIBOSize;
        ShaderProgram mMarkerProgram;
        ShaderProgram mSurfProgram;
        /* shared variable index locations */
        gl::GLuint    mMarkerMatIndex;
        gl::GLuint    mMarkerPointIndex;
        gl::GLuint    mMarkerColorIndex;
        gl::GLuint    mMarkerAlphaIndex;
        gl::GLuint    mMarkerPVCIndex;
        gl::GLuint    mMarkerPVAIndex;
        gl::GLuint    mMarkerTypeIndex;
        gl::GLuint    mMarkerColIndex;

        gl::GLuint    mSurfMatIndex;
        gl::GLuint    mSurfRangeIndex;
        gl::GLuint    mSurfPointIndex;
        gl::GLuint    mSurfColorIndex;
        gl::GLuint    mSurfAlphaIndex;
        gl::GLuint    mSurfPVCIndex;
        gl::GLuint    mSurfPVAIndex;

        std::map<int, gl::GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;
        glm::mat4  computeTransformMat(const glm::mat4& pView, const glm::mat4& pOrient);
        virtual void renderGraph(const int pWindowId, const glm::mat4& transform);

    public:
        surface_impl(const uint pNumXpoints, const uint pNumYpoints,
                     const forge::dtype pDataType, const forge::MarkerType pMarkerType);
        ~surface_impl();

        void render(const int pWindowId,
                    const int pX, const int pY, const int pVPW, const int pVPH,
                    const glm::mat4 &pView, const glm::mat4 &pOrient);

        inline void usePerVertexColors(const bool pFlag=true) {
            mIsPVCOn = pFlag;
        }

        inline void usePerVertexAlphas(const bool pFlag=true) {
            mIsPVAOn = pFlag;
        }
};

class scatter3_impl : public surface_impl {
   private:
        void renderGraph(const int pWindowId, const glm::mat4& transform);

   public:
       scatter3_impl(const uint pNumXPoints, const uint pNumYPoints,
                     const forge::dtype pDataType, const forge::MarkerType pMarkerType=FG_MARKER_NONE)
           : surface_impl(pNumXPoints, pNumYPoints, pDataType, pMarkerType) {}
};

}
}
