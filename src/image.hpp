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
        unsigned  mWidth;
        unsigned  mHeight;
        fg::ChannelFormat mFormat;
        GLenum    mGLformat;
        GLenum    mGLiformat;
        fg::dtype mDataType;
        GLenum    mGLType;
        /* internal resources for interop */
        size_t   mPBOsize;
        GLuint   mPBO;
        GLuint   mTex;
        GLuint   mProgram;

        GLuint   mColorMapUBO;
        GLuint   mUBOSize;
        bool     mKeepARatio;

        /* helper functions to bind and unbind
         * resources for render quad primitive */
        void bindResources(int pWindowId);
        void unbindResources() const;

    public:
        image_impl(unsigned pWidth, unsigned pHeight, fg::ChannelFormat pFormat, fg::dtype pDataType);
        ~image_impl();

        void setColorMapUBOParams(GLuint ubo, GLuint size);
        void keepAspectRatio(const bool keep=true);

        unsigned width() const;
        unsigned height() const;
        fg::ChannelFormat pixelFormat() const;
        fg::dtype channelType() const;
        unsigned pbo() const;
        unsigned size() const;

        void render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight);
};

class _Image {
    private:
        std::shared_ptr<image_impl> img;

    public:
        _Image(unsigned pWidth, unsigned pHeight, fg::ChannelFormat pFormat, fg::dtype pDataType)
            : img(std::make_shared<image_impl>(pWidth, pHeight, pFormat, pDataType)) {}

        inline const std::shared_ptr<image_impl>& impl() const { return img; }

        inline void keepAspectRatio(const bool keep) { img->keepAspectRatio(keep); }

        inline unsigned width() const { return img->width(); }

        inline unsigned height() const { return img->height(); }

        inline fg::ChannelFormat pixelFormat() const { return img->pixelFormat(); }

        inline fg::dtype channelType() const { return img->channelType(); }

        inline GLuint pbo() const { return img->pbo(); }

        inline size_t size() const { return img->size(); }
};

}
