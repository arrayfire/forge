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
#include <vector>
#include <memory>

static const int NUM_CHARS = 95;

namespace internal
{

struct font_impl {
    /* attributes */
    bool mIsFontLoaded;
    std::string mTTFfile;
    std::vector<float> mVertexData;
    int mWidth;
    int mHeight;
    GLuint mVAO;
    GLuint mVBO;
    GLuint mProgram;
    GLuint mSampler;

    GLuint mCharTextures[NUM_CHARS];
    int mAdvX[NUM_CHARS], mAdvY[NUM_CHARS];
    int mBearingX[NUM_CHARS], mBearingY[NUM_CHARS];
    int mCharWidth[NUM_CHARS], mCharHeight[NUM_CHARS];
    int mLoadedPixelSize, mNewLine;

    /* helper function to extract glyph of
    * ASCII character pointed by pIndex*/
    void extractGlyph(int pIndex);

    /* helper to destroy GL objects created for
    * given font face and size if required */
    void destroyGLResources();

    font_impl();
    ~font_impl();

    inline void setOthro2D(int w, int h) {
        mWidth = w;
        mHeight = h;
    }

    inline void setReady(bool value) { mIsFontLoaded = value; }
    inline void setFontFile(const std::string& value) { mTTFfile = value; }
    inline void setPixelSize(int value) { mLoadedPixelSize = value; }
    inline void setVBO(GLuint value) { mVBO = value; }
    inline void setVAO(GLuint value) { mVAO = value; }

    inline int width() const { return mWidth; }
    inline int height() const { return mHeight; }
    inline bool isReady() const { return mIsFontLoaded; }
    inline const std::string& fontfile() const { return mTTFfile; }
    inline int pixelSize() const { return mLoadedPixelSize; }
    inline GLuint vbo() const { return mVBO; }
    inline GLuint vao() const { return mVAO; }
    inline GLuint sampler() const { return mSampler; }
    inline GLuint prog() const { return mProgram; }
    inline int newline() const { return mNewLine; }
};

class _Font {
    private:
        std::shared_ptr<font_impl> fnt;

    public:
        _Font() : fnt(std::make_shared<font_impl>()) {}

        void loadFont(const char* const pFile, int pFontSize);
        void loadSystemFont(const char* const pName, int pFontSize);

        void setOthro2D(int pWidth, int pHeight);

        void render(const float pPos[2], const float pColor[4], const char* pText, int pFontSize = -1, bool pIsVertical = false);
};

}