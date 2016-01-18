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

#include <memory>

namespace internal
{

class image_impl : public AbstractRenderable {
    private:
        uint   mWidth;
        uint   mHeight;
        fg::ChannelFormat mFormat;
        GLenum mGLformat;
        GLenum mGLiformat;
        fg::dtype mDataType;
        GLenum mGLType;
        float  mAlpha;
        bool   mKeepARatio;
        /* internal resources for interop */
        size_t mPBOsize;
        GLuint mPBO;
        GLuint mTex;
        GLuint mProgram;
        GLuint mMatIndex;
        GLuint mTexIndex;
        GLuint mIsGrayIndex;
        GLuint mCMapLenIndex;
        GLuint mCMapIndex;
        /* color map details */
        GLuint mColorMapUBO;
        GLuint mUBOSize;

        /* helper functions to bind and unbind
         * resources for render quad primitive */
        void bindResources(int pWindowId) const;
        void unbindResources() const;

    public:
        image_impl(const uint pWidth, const uint pHeight,
                   const fg::ChannelFormat pFormat, const fg::dtype pDataType);
        ~image_impl();

        void setColorMapUBOParams(const GLuint pUBO, const GLuint pSize);
        void setAlpha(const float pAlpha);
        void keepAspectRatio(const bool pKeep=true);

        uint width() const;
        uint height() const;
        fg::ChannelFormat pixelFormat() const;
        fg::dtype channelType() const;
        uint pbo() const;
        uint size() const;

        void render(const int pWindowId,
                    const int pX, const int pY, const int pVPW, const int pVPH,
                    const glm::mat4& pTransform);
};

class _Image {
    private:
        std::shared_ptr<image_impl> mImage;

    public:
        _Image(const uint pWidth, const uint pHeight,
               const fg::ChannelFormat pFormat, const fg::dtype pDataType)
            : mImage(std::make_shared<image_impl>(pWidth, pHeight, pFormat, pDataType)) {}

        inline const std::shared_ptr<image_impl>& impl() const { return mImage; }

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
