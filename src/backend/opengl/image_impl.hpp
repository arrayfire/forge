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

namespace forge
{
namespace opengl
{

class image_impl : public AbstractRenderable {
    private:
        uint   mWidth;
        uint   mHeight;
        forge::ChannelFormat mFormat;
        GLenum mGLformat;
        GLenum mGLiformat;
        forge::dtype mDataType;
        GLenum mGLType;
        float  mAlpha;
        bool   mKeepARatio;
        size_t mFormatSize;
        /* internal resources for interop */
        size_t mPBOsize;
        GLuint mPBO;
        GLuint mTex;
        ShaderProgram mProgram;
        GLuint mMatIndex;
        GLuint mTexIndex;
        GLuint mNumCIndex;
        GLuint mAlphaIndex;
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
                   const forge::ChannelFormat pFormat, const forge::dtype pDataType);
        ~image_impl();

        void setColorMapUBOParams(const GLuint pUBO, const GLuint pSize);
        void setAlpha(const float pAlpha);
        void keepAspectRatio(const bool pKeep=true);

        uint width() const;
        uint height() const;
        forge::ChannelFormat pixelFormat() const;
        forge::dtype channelType() const;
        uint pbo() const;
        uint size() const;

        void render(const int pWindowId,
                    const int pX, const int pY, const int pVPW, const int pVPH,
                    const glm::mat4 &pView, const glm::mat4 &pOrient);
};

}
}
