/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common.hpp>
#include <err_opengl.hpp>
#include <image_impl.hpp>
#include <window_impl.hpp>
#include <shader_headers/image_vs.hpp>
#include <shader_headers/image_fs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <map>
#include <mutex>

using namespace gl;

namespace forge
{
namespace opengl
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
                       const forge::ChannelFormat pFormat, const forge::dtype pDataType)
    : mWidth(pWidth), mHeight(pHeight), mFormat(pFormat),
      mGLformat(ctype2gl(mFormat)), mGLiformat(ictype2gl(mFormat)),
      mDataType(pDataType), mGLType(dtype2gl(mDataType)), mAlpha(1.0f),
      mKeepARatio(true), mFormatSize(1), mPBOsize(1), mPBO(0), mTex(0),
      mProgram(glsl::image_vs.c_str(), glsl::image_fs.c_str()),
      mMatIndex(-1), mTexIndex(-1), mNumCIndex(-1),
      mAlphaIndex(-1), mCMapLenIndex(-1), mCMapIndex(-1)
{
    CheckGL("Begin image_impl::image_impl");

    mMatIndex     = mProgram.getUniformLocation("matrix");
    mCMapIndex    = mProgram.getUniformBlockIndex("ColorMap");
    mCMapLenIndex = mProgram.getUniformLocation("cmaplen");
    mTexIndex     = mProgram.getUniformLocation("tex");
    mNumCIndex    = mProgram.getUniformLocation("numcomps");
    mAlphaIndex   = mProgram.getUniformLocation("alpha");

    // Initialize OpenGL Items
    glGenTextures(1, &(mTex));
    glBindTexture(GL_TEXTURE_2D, mTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_NEAREST));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_NEAREST));

    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(mGLiformat), mWidth, mHeight, 0, mGLformat, mGLType, NULL);

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
        case FG_GRAYSCALE: mFormatSize = 1;   break;
        case FG_RG:        mFormatSize = 2;   break;
        case FG_RGB:       mFormatSize = 3;   break;
        case FG_BGR:       mFormatSize = 3;   break;
        case FG_RGBA:      mFormatSize = 4;   break;
        case FG_BGRA:      mFormatSize = 4;   break;
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
    glDeleteBuffers(1, &mPBO);
    glDeleteTextures(1, &mTex);
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

forge::ChannelFormat image_impl::pixelFormat() const { return mFormat; }

forge::dtype image_impl::channelType() const { return mDataType; }

uint image_impl::pbo() const { return mPBO; }

uint image_impl::size() const { return (uint)mPBOsize; }

void image_impl::render(const int pWindowId,
                        const int pX, const int pY, const int pVPW, const int pVPH,
                        const glm::mat4 &pView, const glm::mat4 &pOrient)
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

    glm::mat4 strans = glm::scale(pView, glm::vec3(xscale, yscale, 1));

    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    mProgram.bind();

    glUniform1i(mNumCIndex, gl::GLint(mFormatSize));
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
    glUniformBlockBinding(mProgram.getProgramId(), mCMapIndex, 0);

    // Draw to screen
    bindResources(pWindowId);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    unbindResources();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // ubind the shader program
    mProgram.unbind();

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);

    CheckGL("End image_impl::render");
}

}
}
