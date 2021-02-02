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

namespace forge {
namespace opengl {

class image_impl : public AbstractRenderable {
   private:
    uint32_t mWidth;
    uint32_t mHeight;
    forge::ChannelFormat mFormat;
    forge::dtype mDataType;
    float mAlpha;
    bool mKeepARatio;
    size_t mFormatSize;
    /* internal resources for interop */
    size_t mPBOsize;
    uint32_t mPBO;
    uint32_t mTex;
    ShaderProgram mProgram;
    uint32_t mMatIndex;
    uint32_t mTexIndex;
    uint32_t mNumCIndex;
    uint32_t mAlphaIndex;
    uint32_t mCMapLenIndex;
    uint32_t mCMapIndex;
    /* color map details */
    uint32_t mColorMapUBO;
    uint32_t mUBOSize;

    /* helper functions to bind and unbind
     * resources for render quad primitive */
    void bindResources(int pWindowId) const;
    void unbindResources() const;

   public:
    image_impl(const uint32_t pWidth, const uint32_t pHeight,
               const forge::ChannelFormat pFormat,
               const forge::dtype pDataType);
    ~image_impl();

    void setColorMapUBOParams(const uint32_t pUBO, const uint32_t pSize);
    void setAlpha(const float pAlpha);
    void keepAspectRatio(const bool pKeep = true);

    uint32_t width() const;
    uint32_t height() const;
    forge::ChannelFormat pixelFormat() const;
    forge::dtype channelType() const;
    uint32_t pbo() const;
    uint32_t size() const;

    void render(const int pWindowId, const int pX, const int pY, const int pVPW,
                const int pVPH, const glm::mat4 &pView,
                const glm::mat4 &pOrient);

    bool isRotatable() const;
};

}  // namespace opengl
}  // namespace forge
