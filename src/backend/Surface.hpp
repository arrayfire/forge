/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <backend.hpp>
#include <surface.hpp>

#include <memory>

namespace common
{

class Surface {
    private:
        std::shared_ptr<detail::surface_impl> mSurface;

    public:
        Surface(const uint pNumXPoints, const uint pNumYPoints,
                 const fg::dtype pDataType, const fg::PlotType pPlotType=FG_SURFACE,
                 const fg::MarkerType pMarkerType=FG_NONE) {
            switch(pPlotType){
                case(FG_SURFACE):
                    mSurface = std::make_shared<detail::surface_impl>(pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                    break;
                case(FG_SCATTER):
                    mSurface = std::make_shared<detail::scatter3_impl>(pNumXPoints, pNumYPoints, pDataType, pMarkerType);
                    break;
                default:
                    mSurface = std::make_shared<detail::surface_impl>(pNumXPoints, pNumYPoints, pDataType, pMarkerType);
            };
        }

        inline const std::shared_ptr<detail::surface_impl>& impl() const {
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
