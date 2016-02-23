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

namespace internal
{

class surface_impl : public AbstractRenderable {
    protected:
        /* plot points characteristics */
        GLuint    mNumXPoints;
        GLuint    mNumYPoints;
        GLenum    mDataType;
        bool      mIsPVCOn;
        bool      mIsPVAOn;
        fg::MarkerType mMarkerType;
        /* OpenGL Objects */
        GLuint    mIBO;
        size_t    mIBOSize;
        GLuint    mMarkerProgram;
        GLuint    mSurfProgram;
        /* shared variable index locations */
        GLuint    mMarkerMatIndex;
        GLuint    mMarkerPointIndex;
        GLuint    mMarkerColorIndex;
        GLuint    mMarkerAlphaIndex;
        GLuint    mMarkerPVCIndex;
        GLuint    mMarkerPVAIndex;
        GLuint    mMarkerTypeIndex;
        GLuint    mMarkerColIndex;

        GLuint    mSurfMatIndex;
        GLuint    mSurfRangeIndex;
        GLuint    mSurfPointIndex;
        GLuint    mSurfColorIndex;
        GLuint    mSurfAlphaIndex;
        GLuint    mSurfPVCIndex;
        GLuint    mSurfPVAIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId);
        void unbindResources() const;
        void computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput);
        virtual void renderGraph(const int pWindowId, const glm::mat4& transform);

    public:
        surface_impl(const uint pNumXpoints, const uint pNumYpoints,
                     const fg::dtype pDataType, const fg::MarkerType pMarkerType);
        ~surface_impl();

        void render(const int pWindowId,
                    const int pX, const int pY, const int pVPW, const int pVPH,
                    const glm::mat4 &pTransform);

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
                     const fg::dtype pDataType, const fg::MarkerType pMarkerType=fg::FG_NONE)
           : surface_impl(pNumXPoints, pNumYPoints, pDataType, pMarkerType) {}
};

class _Surface {
    private:
        std::shared_ptr<surface_impl> mSurface;

    public:
        _Surface(const uint pNumXPoints, const uint pNumYPoints,
                 const fg::dtype pDataType, const fg::PlotType pPlotType=fg::FG_SURFACE,
                 const fg::MarkerType pMarkerType=fg::FG_NONE) {
            switch(pPlotType){
                case(fg::FG_SURFACE):
                    mSurface = std::make_shared<surface_impl>(pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                    break;
                case(fg::FG_SCATTER):
                    mSurface = std::make_shared<scatter3_impl>(pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                    break;
                default:
                    mSurface = std::make_shared<surface_impl>(pNumXPoints, pNumYPoints, pDataType, pMarkerType);
            };
        }

        inline const std::shared_ptr<surface_impl>& impl() const {
            return mSurface;
        }

        inline void setColor(const float pRed, const float pGreen,
                             const float pBlue, const float pAlpha) {
            mSurface->setColor(pRed, pGreen, pBlue, pAlpha);
        }

        inline void setLegend(const char* pLegend) {
            mSurface->setLegend(pLegend);
        }

        inline GLuint vbo() const {
            return mSurface->vbo();
        }

        inline GLuint cbo() const {
            return mSurface->cbo();
        }

        inline GLuint abo() const {
            return mSurface->abo();
        }

        inline size_t vboSize() const {
            return mSurface->vboSize();
        }

        inline size_t cboSize() const {
            return mSurface->cboSize();
        }

        inline size_t aboSize() const {
            return mSurface->aboSize();
        }

        inline void render(const int pWindowId,
                           const int pX, const int pY, const int pVPW, const int pVPH,
                           const glm::mat4& pTransform) const {
            mSurface->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
        }
};

}
