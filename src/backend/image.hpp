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
#include <image_impl.hpp>

#include <glm/glm.hpp>

#include <memory>

namespace common
{

class Image {
    private:
        std::shared_ptr<detail::image_impl> mImage;

    public:
        Image(const uint pWidth, const uint pHeight,
               const fg::ChannelFormat pFormat, const fg::dtype pDataType)
            : mImage(std::make_shared<detail::image_impl>(pWidth, pHeight, pFormat, pDataType)) {}

        Image(const fg_image pOther) {
            mImage = reinterpret_cast<Image*>(pOther)->impl();
        }

        inline const std::shared_ptr<detail::image_impl>& impl() const { return mImage; }

        inline void setAlpha(const float pAlpha) { mImage->setAlpha(pAlpha); }

        inline void keepAspectRatio(const bool pKeep) { mImage->keepAspectRatio(pKeep); }

        inline uint width() const { return mImage->width(); }

        inline uint height() const { return mImage->height(); }

        inline fg::ChannelFormat pixelFormat() const { return mImage->pixelFormat(); }

        inline fg::dtype channelType() const { return mImage->channelType(); }

        inline GLuint pbo() const { return mImage->pbo(); }

        inline size_t size() const { return mImage->size(); }

        inline void render(const int pWindowId,
                           const int pX, const int pY, const int pVPW, const int pVPH,
                           const glm::mat4& pTransform) const {
            mImage->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
        }
};

}
