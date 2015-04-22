/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fg/window.h>
#include <vector>

namespace fg
{

class FGAPI Image {
    private:
        unsigned  mWidth;
        unsigned  mHeight;
        ColorMode mFormat;
        GLenum    mGLformat;
        GLenum    mDataType;
        /* internal resources for interop */
        size_t   mPBOsize;
        GLuint   mPBO;
        GLuint   mTex;
        GLuint   mProgram;

    public:
        Image(unsigned pWidth, unsigned pHeight, ColorMode pFormat, GLenum pDataType);
        ~Image();

        unsigned width() const;
        unsigned height() const;
        ColorMode pixelFormat() const;
        GLenum channelType() const;
        GLuint pbo() const;
        size_t size() const;

        void render() const;
};

FGAPI void drawImage(Window* pWindow, const Image& pImage);

FGAPI void drawImages(Window* pWindow, int pRows, int pCols, const std::vector<Image>& pImages);

}
