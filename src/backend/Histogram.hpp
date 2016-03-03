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
#include <histogram.hpp>

#include <memory>

namespace common
{

class Histogram {
    private:
        std::shared_ptr<detail::hist_impl> mHistogram;

    public:
        Histogram(uint pNBins, fg::dtype pDataType)
            : mHistogram(std::make_shared<detail::hist_impl>(pNBins, pDataType)) {}

        inline const std::shared_ptr<detail::hist_impl>& impl() const {
            return mHistogram;
        }

        inline void setColor(const float pRed, const float pGreen,
                             const float pBlue, const float pAlpha) {
            mHistogram->setColor(pRed, pGreen, pBlue, pAlpha);
        }

        inline void setLegend(const char* pLegend) {
            mHistogram->setLegend(pLegend);
        }

        inline GLuint vbo() const {
            return mHistogram->vbo();
        }

        inline GLuint cbo() const {
            return mHistogram->cbo();
        }

        inline GLuint abo() const {
            return mHistogram->abo();
        }

        inline size_t vboSize() const {
            return mHistogram->vboSize();
        }

        inline size_t cboSize() const {
            return mHistogram->cboSize();
        }

        inline size_t aboSize() const {
            return mHistogram->aboSize();
        }

        inline void render(const int pWindowId,
                           const int pX, const int pY, const int pVPW, const int pVPH,
                           const glm::mat4& pTransform) const {
            mHistogram->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
        }
};

}
