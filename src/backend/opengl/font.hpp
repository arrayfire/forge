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
#include <font_atlas.hpp>

#include <map>
#include <vector>
#include <memory>

static const size_t MIN_FONT_SIZE = 8;
static const size_t MAX_FONT_SIZE = 36;

namespace opengl
{

typedef std::vector<Glyph*> GlyphList;

class font_impl {
    private:
        /* VAO map to store a vertex array object
         * for each valid window context */
        std::map<int, GLuint> mVAOMap;

        /* attributes */
        std::string mTTFfile;
        bool        mIsFontLoaded;
        FontAtlas*  mAtlas;
        GLuint      mVBO;
        GLuint      mProgram;
        int         mOrthoW;
        int         mOrthoH;

        std::vector<GlyphList> mGlyphLists;

        /* OpenGL Data */
        glm::mat4   mProjMat;
        GLuint      mPMatIndex;
        GLuint      mMMatIndex;
        GLuint      mTexIndex;
        GLuint      mClrIndex;

        /* load all glyphs and create character atlas */
        void loadAtlasWithGlyphs(const size_t pFontSize);

        /* helper functions to bind and unbind
         * rendering resources */
        void bindResources(int pWindowId);
        void unbindResources() const;

        /* helper to destroy GL objects created for
         * given font face and size if required */
        void destroyGLResources();

    public:
        font_impl();
        ~font_impl();

        void setOthro2D(int pWidth, int pHeight);
        void loadFont(const char* const pFile);
        void loadSystemFont(const char* const pName);

        void render(int pWindowId,
                    const float pPos[2], const float pColor[4], const char* pText,
                    size_t pFontSize, bool pIsVertical = false);
};

}
