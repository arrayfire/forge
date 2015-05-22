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

struct image_impl {
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

    image_impl(unsigned pWidth, unsigned pHeight, fg::ColorMode pFormat, GLenum pDataType);
    ~image_impl();
    void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const;
};

class _Image {
    private:
        std::shared_ptr<image_impl> img;

    public:
        _Image(unsigned pWidth, unsigned pHeight, fg::ColorMode pFormat, GLenum pDataType)
            : img(std::make_shared<image_impl>(pWidth, pHeight, pFormat, pDataType)) {}

        unsigned width() const { return img->mWidth; }
        unsigned height() const { return img->mHeight; }
        fg::ColorMode pixelFormat() const { return img->mFormat; }
        GLenum channelType() const { return img->mDataType; }
        GLuint pbo() const { return img->mPBO; }
        size_t size() const { return img->mPBOsize; }

        inline void render(int pX, int pY, int pViewPortWidth, int pViewPortHeight) const {
            img->render(pX, pY, pViewPortWidth, pViewPortHeight);
        }
};

}
