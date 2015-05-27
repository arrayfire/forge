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

class font_impl {
    private:
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

    public:
        font_impl();
        ~font_impl();

        void setOthro2D(int pWidth, int pHeight);
        void loadFont(const char* const pFile, int pFontSize);
        void loadSystemFont(const char* const pName, int pFontSize);

        void render(const float pPos[2], const float pColor[4], const char* pText, int pFontSize = -1, bool pIsVertical = false);
};

class _Font {
    private:
        std::shared_ptr<font_impl> fnt;

    public:
        _Font() : fnt(std::make_shared<font_impl>()) {}

        const std::shared_ptr<font_impl>& impl() const {
            return fnt;
        }

        inline void setOthro2D(int pWidth, int pHeight) {
            fnt->setOthro2D(pWidth, pHeight);
        }

        inline void loadFont(const char* const pFile, int pFontSize) {
            fnt->loadFont(pFile, pFontSize);
        }

        inline void loadSystemFont(const char* const pName, int pFontSize) {
            fnt->loadSystemFont(pName, pFontSize);
        }

        inline void render(const float pPos[2], const float pColor[4], const char* pText,
                int pFontSize = -1, bool pIsVertical = false) {
            fnt->render(pPos, pColor, pText, pFontSize, pIsVertical);
        }
};

}
