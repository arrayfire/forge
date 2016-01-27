/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/image.h>
#include <fg/window.h>
#include <image.hpp>
#include <window.hpp>
#include <common.hpp>
#include <shader_headers/image_vs.hpp>
#include <shader_headers/image_fs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <map>
#include <mutex>

namespace internal
{

void image_impl::bindResources(int pWindowId) const
{
    glBindVertexArray(screenQuadVAO(pWindowId));
}

void image_impl::unbindResources() const
{
    glBindVertexArray(0);
}

image_impl::image_impl(const uint pWidth, const uint pHeight,
                       const fg::ChannelFormat pFormat, const fg::dtype pDataType)
    : mWidth(pWidth), mHeight(pHeight), mFormat(pFormat),
      mGLformat(ctype2gl(mFormat)), mGLiformat(ictype2gl(mFormat)),
      mDataType(pDataType), mGLType(dtype2gl(mDataType)), mAlpha(1.0f),
      mKeepARatio(true), mFormatSize(1), mMatIndex(-1), mTexIndex(-1),
      mNumCIndex(-1), mAlphaIndex(-1), mCMapLenIndex(-1), mCMapIndex(-1)
{
    CheckGL("Begin image_impl::image_impl");

    mProgram      = initShaders(glsl::image_vs.c_str(), glsl::image_fs.c_str());
    mMatIndex     = glGetUniformLocation(mProgram, "matrix");
    mCMapIndex    = glGetUniformBlockIndex(mProgram, "ColorMap");
    mCMapLenIndex = glGetUniformLocation(mProgram, "cmaplen");
    mTexIndex     = glGetUniformLocation(mProgram, "tex");
    mNumCIndex    = glGetUniformLocation(mProgram, "numcomps");
    mAlphaIndex   = glGetUniformLocation(mProgram, "alpha");

    // Initialize OpenGL Items
    glGenTextures(1, &(mTex));
    glBindTexture(GL_TEXTURE_2D, mTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, mGLiformat, mWidth, mHeight, 0, mGLformat, mGLType, NULL);

    CheckGL("Before PBO Initialization");
    glGenBuffers(1, &mPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPBO);
    size_t typeSize = 0;
    switch(mGLType) {
        case GL_INT:            typeSize = sizeof(int   ); break;
        case GL_UNSIGNED_INT:   typeSize = sizeof(uint  ); break;
        case GL_SHORT:          typeSize = sizeof(short ); break;
        case GL_UNSIGNED_SHORT: typeSize = sizeof(ushort); break;
        case GL_BYTE:           typeSize = sizeof(char  ); break;
        case GL_UNSIGNED_BYTE:  typeSize = sizeof(uchar ); break;
        default: typeSize = sizeof(float); break;
    }
    switch(mFormat) {
        case fg::FG_GRAYSCALE: mFormatSize = 1;   break;
        case fg::FG_RG:        mFormatSize = 2;   break;
        case fg::FG_RGB:       mFormatSize = 3;   break;
        case fg::FG_BGR:       mFormatSize = 3;   break;
        case fg::FG_RGBA:      mFormatSize = 4;   break;
        case fg::FG_BGRA:      mFormatSize = 4;   break;
        default: mFormatSize = 1; break;
    }
    mPBOsize = mWidth * mHeight * mFormatSize * typeSize;
    glBufferData(GL_PIXEL_UNPACK_BUFFER, mPBOsize, NULL, GL_STREAM_COPY);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CheckGL("End image_impl::image_impl");
}

image_impl::~image_impl()
{
    CheckGL("Begin image_impl::~image_impl");
    glDeleteBuffers(1, &mPBO);
    glDeleteTextures(1, &mTex);
    glDeleteProgram(mProgram);
    CheckGL("End image_impl::~image_impl");
}

void image_impl::setColorMapUBOParams(const GLuint pUBO, const GLuint pSize)
{
    mColorMapUBO = pUBO;
    mUBOSize = pSize;
}

void image_impl::setAlpha(const float pAlpha)
{
    mAlpha = pAlpha;
}

void image_impl::keepAspectRatio(const bool pKeep)
{
    mKeepARatio = pKeep;
}

uint image_impl::width() const { return mWidth; }

uint image_impl::height() const { return mHeight; }

fg::ChannelFormat image_impl::pixelFormat() const { return mFormat; }

fg::dtype image_impl::channelType() const { return mDataType; }

uint image_impl::pbo() const { return mPBO; }

uint image_impl::size() const { return (uint)mPBOsize; }

void image_impl::render(const int pWindowId,
                        const int pX, const int pY, const int pVPW, const int pVPH,
                        const glm::mat4& pTransform)
{
    CheckGL("Begin image_impl::render");

    float xscale = 1.f;
    float yscale = 1.f;
    if (mKeepARatio) {
        if (mWidth > mHeight) {
            float trgtH = pVPW * float(mHeight)/float(mWidth);
            float trgtW = trgtH * float(mWidth)/float(mHeight);
            xscale = trgtW/pVPW;
            yscale = trgtH/pVPH;
        } else {
            float trgtW = pVPH * float(mWidth)/float(mHeight);
            float trgtH = trgtW * float(mHeight)/float(mWidth);
            xscale = trgtW/pVPW;
            yscale = trgtH/pVPH;
        }
    }

    glm::mat4 strans = glm::scale(pTransform, glm::vec3(xscale, yscale, 1));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(mProgram);

    glUniform1i(mNumCIndex, mFormatSize);
    glUniform1f(mAlphaIndex, mAlpha);

    // load texture from PBO
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(mTexIndex, 0);
    glBindTexture(GL_TEXTURE_2D, mTex);
    // bind PBO to load data into texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPBO);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, mGLformat, mGLType, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glUniformMatrix4fv(mMatIndex, 1, GL_FALSE, glm::value_ptr(strans));

    glUniform1f(mCMapLenIndex, (GLfloat)mUBOSize);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, mColorMapUBO);
    glUniformBlockBinding(mProgram, mCMapIndex, 0);

    // Draw to screen
    bindResources(pWindowId);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    unbindResources();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // ubind the shader program
    glUseProgram(0);
    glDisable(GL_BLEND);

    CheckGL("End image_impl::render");
}

}

namespace fg
{

Image::Image(const uint pWidth, const uint pHeight,
             const ChannelFormat pFormat, const dtype pDataType)
{
    mValue = new internal::_Image(pWidth, pHeight, pFormat, pDataType);
}

Image::Image(const Image& pOther)
{
    mValue = new internal::_Image(*pOther.get());
}

Image::~Image()
{
    delete mValue;
}

void Image::setAlpha(const float pAlpha)
{
    mValue->setAlpha(pAlpha);
}

void Image::keepAspectRatio(const bool pKeep)
{
    mValue->keepAspectRatio(pKeep);
}

uint Image::width() const
{
    return mValue->width();
}

uint Image::height() const
{
    return mValue->height();
}

ChannelFormat Image::pixelFormat() const
{
    return mValue->pixelFormat();
}

fg::dtype Image::channelType() const
{
    return mValue->channelType();
}

GLuint Image::pbo() const
{
    return mValue->pbo();
}

uint Image::size() const
{
    return (uint)mValue->size();
}

void Image::render(const Window& pWindow,
                   const int pX, const int pY, const int pVPW, const int pVPH,
                   const std::vector<float>& pTransform) const
{
    if (pTransform.size() < 16) {
        throw ArgumentError("Image::render", __LINE__, 5,
                "Insufficient transform matrix data");
    }
    mValue->render(pWindow.get()->getID(),
                   pX, pY, pVPW, pVPH,
                   glm::make_mat4(pTransform.data()));
}


internal::_Image* Image::get() const
{
    return mValue;
}

}
