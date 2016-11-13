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
#include <font_atlas_impl.hpp>

#include <map>
#include <vector>
#include <memory>

static const float MIN_FONT_SIZE = 4.0f;
static const float MAX_FONT_SIZE = 18.0f;

namespace forge
{
namespace opengl
{

typedef std::vector<Glyph*> GlyphList;

class font_impl {
    private:
        /* VAO map to store a vertex array object
         * for each valid window context */
        std::map<int, gl::GLuint> mVAOMap;

        /* attributes */
        std::string mTTFfile;
        bool        mIsFontLoaded;
        FontAtlas*  mAtlas;
        gl::GLuint  mVBO;
        ShaderProgram mProgram;
        int         mOrthoW;
        int         mOrthoH;

        std::vector<GlyphList> mGlyphLists;

        /* OpenGL Data */
        glm::mat4   mProjMat;
        gl::GLuint  mPMatIndex;
        gl::GLuint  mMMatIndex;
        gl::GLuint  mTexIndex;
        gl::GLuint  mClrIndex;

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
}
