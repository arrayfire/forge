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

namespace internal
{

class _Image {
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

    public:
        _Image(unsigned pWidth, unsigned pHeight, fg::ColorMode pFormat, GLenum pDataType);
        ~_Image();

        unsigned width() const;
        unsigned height() const;
        fg::ColorMode pixelFormat() const;
        GLenum channelType() const;
        GLuint pbo() const;
        size_t size() const;

        void render() const;
};

}