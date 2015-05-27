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
        fg::ColorMode mFormat;
        GLenum    mGLformat;
        GLenum    mDataType;
        /* internal resources for interop */
        size_t   mPBOsize;
        GLuint   mPBO;
        GLuint   mTex;
        GLuint   mProgram;

        /* helper functions to bind and unbind
         * resources for render quad primitive */
        void bindResources() const;
        void unbindResources() const;

    public:
        image_impl(unsigned pWidth, unsigned pHeight, fg::ColorMode pFormat, GLenum pDataType);
        ~image_impl();

        unsigned width() const;
        unsigned height() const;
        fg::ColorMode pixelFormat() const;
        GLenum channelType() const;
        GLuint pbo() const;
        size_t size() const;

        void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const;
};

class _Image {
    private:
        std::shared_ptr<image_impl> img;

    public:
        _Image(unsigned pWidth, unsigned pHeight, fg::ColorMode pFormat, GLenum pDataType)
            : img(std::make_shared<image_impl>(pWidth, pHeight, pFormat, pDataType)) {}

        inline const std::shared_ptr<image_impl>& impl() const { return img; }

        inline unsigned width() const { return img->width(); }

        inline unsigned height() const { return img->height(); }

        inline fg::ColorMode pixelFormat() const { return img->pixelFormat(); }

        inline GLenum channelType() const { return img->channelType(); }

        inline GLuint pbo() const { return img->pbo(); }

        inline size_t size() const { return img->size(); }

        inline void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const {
            img->render(pX, pY, pViewPortWidth, pViewPortHeight);
        }
};

}
