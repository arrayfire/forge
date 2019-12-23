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
#include <glm/glm.hpp>

#include <memory>

namespace forge {
namespace common {

// Objects of type `RenderableType` in the following class definition
// should implement all the member functons of ChartRenderableBase
// class, otherwise you cannot use this class.
template<class RenderableType>
class ChartRenderableBase {
   protected:
    std::shared_ptr<RenderableType> mShrdPtr;

   public:
    ChartRenderableBase() {}

    ChartRenderableBase(const std::shared_ptr<RenderableType>& pValue)
        : mShrdPtr(pValue) {}

    inline const std::shared_ptr<RenderableType>& impl() const {
        return mShrdPtr;
    }

    inline void setColor(const float pRed, const float pGreen,
                         const float pBlue, const float pAlpha) {
        mShrdPtr->setColor(pRed, pGreen, pBlue, pAlpha);
    }

    inline void setLegend(const char* pLegend) { mShrdPtr->setLegend(pLegend); }

    inline unsigned vbo() const { return mShrdPtr->vbo(); }

    inline unsigned cbo() const { return mShrdPtr->cbo(); }

    inline unsigned abo() const { return mShrdPtr->abo(); }

    inline size_t vboSize() const { return mShrdPtr->vboSize(); }

    inline size_t cboSize() const { return mShrdPtr->cboSize(); }

    inline size_t aboSize() const { return mShrdPtr->aboSize(); }

    inline void render(const int pWindowId, const int pX, const int pY,
                       const int pVPW, const int pVPH,
                       const glm::mat4& pTransform) const {
        mShrdPtr->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
    }
};

}  // namespace common
}  // namespace forge
