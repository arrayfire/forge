/*******************************************************
 * Copyright (c) 2014, ArrayFire
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
        unsigned mWidth;
        unsigned mHeight;
        GLint    mFormat;
        GLenum   mDataType;
        /* internal resources for interop */
        GLuint   mPBO;
        GLuint   mTex;
        GLuint   mProgram;

    public:
        Image(unsigned pWidth, unsigned pHeight, GLint pFormat, GLenum pDataType);
        ~Image();

        unsigned width() const;
        unsigned height() const;
        GLint pixelFormat() const;
        GLenum channelType() const;
        GLuint pbo() const;

        void render() const;
};

FGAPI void drawImage(Window* pWindow, const Image& pImage);

FGAPI void drawImages(Window* pWindow, int pRows, int pCols, const std::vector<Image>& pImages);

}
