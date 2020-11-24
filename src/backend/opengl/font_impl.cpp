/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_handling.hpp>
#include <common/util.hpp>
#include <font_impl.hpp>
#include <gl_helpers.hpp>
#include <shader_headers/font_fs.hpp>
#include <shader_headers/font_vs.hpp>

#include <ft2build.h>
#include <glm/gtc/matrix_transform.hpp>
#include FT_FREETYPE_H
#include FT_STROKER_H
#include FT_ERRORS_H
#if !defined(OS_WIN)
#include <fontconfig/fontconfig.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#if defined(OS_WIN)
#include <windows.h>
#include <regex>
#endif

//// ASCII printable characters (character code 32-127)
////
//// Codes 32-127 are common for all the different variations of the ASCII
/// table, / they are called printable characters, represent letters, digits,
/// punctuation / marks, and a few miscellaneous symbols. You will find almost
/// every character / on your keyboard. Character 127 represents the command
/// DEL.
#define START_CHAR 32
#define END_CHAR 126

/* freetype library types */

namespace forge {
namespace opengl {

const float MIN_FONT_SIZE = 8.0f;
const float MAX_FONT_SIZE = 24.0f;

#ifdef NDEBUG
/* Relase Mode */
#define FT_THROW_ERROR(msg, error) FG_ERROR(msg, error)
#else
/* Debug Mode */
#if FREETYPE_MAJOR >= 2 && FREETYPE_MINOR > 8
#define FT_THROW_ERROR(msg, err)                        \
    do {                                                \
        std::ostringstream ss;                          \
        ss << msg << FT_Error_String(err) << std::endl; \
        FG_ERROR(ss.str().c_str(), err);                \
    } while (0)
#else
#define FT_THROW_ERROR(msg, error) FG_ERROR(msg, error)
#endif
#endif

void font_impl::loadAtlasWithGlyphs(const size_t pFontSize) {
    FT_Library library;
    FT_Face face;
    /* Initialize freetype font library */
    FT_Error bError = FT_Init_FreeType(&library);
    if (bError)
        FT_THROW_ERROR("Freetype Initialization failed", FG_ERR_FREETYPE_ERROR);
    /* get font face for requested font */
    bError = FT_New_Face(library, mTTFfile.c_str(), 0, &face);
    if (bError) {
        FT_Done_FreeType(library);
        FT_THROW_ERROR("Freetype face initilization", FG_ERR_FREETYPE_ERROR);
    }
    /* Select charmap */
    bError = FT_Select_Charmap(face, FT_ENCODING_UNICODE);
    if (bError) {
        FT_Done_Face(face);
        FT_Done_FreeType(library);
        FT_THROW_ERROR("Freetype charmap set failed", FG_ERR_FREETYPE_ERROR);
    }
    /* set the pixel size of font */
    bError = FT_Set_Pixel_Sizes(face, 0, (FT_UInt)pFontSize);
    if (bError) {
        FT_Done_Face(face);
        FT_Done_FreeType(library);
        FT_THROW_ERROR("Freetype char size set failed", FG_ERR_FREETYPE_ERROR);
    }

    size_t missed = 0;

    /* retrieve the list of current font size */
    auto& currList = mGlyphLists[pFontSize - size_t(MIN_FONT_SIZE)];

    for (int i = START_CHAR; i <= END_CHAR; ++i) {
        FT_UInt glyphIndex = FT_Get_Char_Index(face, (FT_ULong)i);

        FT_Int32 flags = 0;

        /* solid outline */
        flags |= FT_LOAD_NO_BITMAP;
        flags |= FT_LOAD_FORCE_AUTOHINT;

        /* load glyph */
        FT_Error bError = FT_Load_Glyph(face, glyphIndex, flags);
        if (bError) {
            FT_Done_Face(face);
            FT_Done_FreeType(library);
            FT_THROW_ERROR("FT_Load_Glyph failed", FG_ERR_FREETYPE_ERROR);
        }

        FT_Glyph currGlyph;
        ;

        bError = FT_Get_Glyph(face->glyph, &currGlyph);
        if (bError) {
            FT_Done_Face(face);
            FT_Done_FreeType(library);
            FT_THROW_ERROR("FT_Get_Glyph", FG_ERR_FREETYPE_ERROR);
        }

        /* fixed channel depth of 1 */
        bError = FT_Glyph_To_Bitmap(&currGlyph, FT_RENDER_MODE_NORMAL, 0, 1);
        if (bError) {
            // FIXME Renable when outline strokes are working
            // FT_Stroker_Done(stroker);
            FT_Done_Face(face);
            FT_Done_FreeType(library);
            FT_THROW_ERROR("FT_Glyph_To_Bitmap", FG_ERR_FREETYPE_ERROR);
        }

        FT_BitmapGlyph bmpGlyph = (FT_BitmapGlyph)currGlyph;
        FT_Bitmap bmp           = bmpGlyph->bitmap;

        int w = bmp.width + 1;
        int h = bmp.rows + 1;

        glm::vec4 region = mAtlas->getRegion(w, h);

        if (region.x < 0 || region.y < 0) {
            missed++;
            std::cerr << "Texture atlas is full" << std::endl;
            continue;
        }

        w = w - 1;  // reduce by one again to leave one pixel border
        h = h - 1;  // reduce by one again to leave one pixel border

        int x = int(region.x);
        int y = int(region.y);

        mAtlas->setRegion(x, y, w, h, bmp.buffer, bmp.pitch);

        Glyph* glyph = new Glyph();

        glyph->mWidth  = w;
        glyph->mHeight = h;

        glyph->mBearingX = face->glyph->metrics.horiBearingX >> 6;
        glyph->mBearingY = face->glyph->metrics.horiBearingY >> 6;

        glyph->mAdvanceX = float(face->glyph->advance.x >> 6);
        glyph->mAdvanceY = float(
            (face->glyph->metrics.height - face->glyph->metrics.horiBearingY) >>
            6);

        glyph->mS0 = x / (float)mAtlas->width();
        glyph->mT1 = y / (float)mAtlas->height();
        glyph->mS1 = (x + glyph->mWidth) / (float)mAtlas->width();
        glyph->mT0 = (y + glyph->mHeight) / (float)mAtlas->height();

        currList.push_back(glyph);

        FT_Done_Glyph(currGlyph);
    }

    /* cleanup freetype variables */
    FT_Done_Face(face);
    FT_Done_FreeType(library);
}

void font_impl::bindResources(int pWindowId) {
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        size_t sz = 2 * sizeof(float);
        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, GLsizei(2 * sz), 0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, GLsizei(2 * sz),
                              reinterpret_cast<void*>(sz));
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }
    glBindVertexArray(mVAOMap[pWindowId]);
}

void font_impl::unbindResources() const { glBindVertexArray(0); }

void font_impl::destroyGLResources() {
    for (auto it = mVAOMap.begin(); it != mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    mVAOMap.clear();

    if (mVBO) glDeleteBuffers(1, &mVBO);
    /* remove all glyph structures from heap */
    for (auto it : mGlyphLists) {
        /* for each font size glyph list */
        for (auto& m : it) { delete m; /* delete Glyph structure */ }
        it.clear();
    }
    /* clear list */
    mGlyphLists.clear();
}

font_impl::font_impl()
    : mTTFfile("")
    , mIsFontLoaded(false)
    , mAtlas(new FontAtlas(512, 512, 1))
    , mVBO(0)
    , mProgram(glsl::font_vs.c_str(), glsl::font_fs.c_str())
    , mOrthoW(1)
    , mOrthoH(1) {
    mPMatIndex = mProgram.getUniformLocation("projectionMatrix");
    mMMatIndex = mProgram.getUniformLocation("modelViewMatrix");
    mTexIndex  = mProgram.getUniformLocation("tex");
    mClrIndex  = mProgram.getUniformLocation("textColor");

    mGlyphLists.resize(size_t(MAX_FONT_SIZE - MIN_FONT_SIZE) + 1, GlyphList());
}

font_impl::~font_impl() {
    destroyGLResources();
    /* clean glyph texture atlas */
    mAtlas.reset();
}

void font_impl::setOthro2D(size_t pWidth, size_t pHeight) {
    mOrthoW  = pWidth;
    mOrthoH  = pHeight;
    mProjMat = glm::ortho(0.0f, float(mOrthoW), 0.0f, float(mOrthoH));
}

void font_impl::loadFont(const char* const pFile) {
    CheckGL("Begin font_impl::loadFont");

    /* Check if font is already loaded. If yes, check if current font load
     * request is same as earlier. If so, return from the function,
     * otherwise, cleanup currently used resources and mark accordingly for
     * subsequent font loading.*/
    if (mIsFontLoaded) {
        if (pFile == mTTFfile)
            return;
        else {
            destroyGLResources();
            mIsFontLoaded = false;
        }
    }

    mTTFfile = pFile;
    /* Load different font sizes into font atlas */
    for (size_t s = size_t(MIN_FONT_SIZE); s <= size_t(MAX_FONT_SIZE); ++s) {
        loadAtlasWithGlyphs(s);
    }

    mAtlas->upload();

    /* push each glyphs vertex and texture data into VBO */
    std::vector<float> vdata;

    uint32_t index = 0;

    for (size_t f = 0; f < mGlyphLists.size(); ++f) {
        auto& list = mGlyphLists[f];

        for (size_t l = 0; l < list.size(); ++l) {
            Glyph* g = list[l];

            std::vector<float> data(16, 0.0f);
            data[0] = 0.0f;
            data[1] = float(-g->mAdvanceY + g->mHeight);
            data[2] = g->mS0;
            data[3] = g->mT1;

            data[4] = 0.0f;
            data[5] = float(-g->mAdvanceY);
            data[6] = g->mS0;
            data[7] = g->mT0;

            data[8]  = float(g->mWidth);
            data[9]  = float(-g->mAdvanceY + g->mHeight);
            data[10] = g->mS1;
            data[11] = g->mT1;

            data[12] = float(g->mWidth);
            data[13] = float(-g->mAdvanceY);
            data[14] = g->mS1;
            data[15] = g->mT0;

            vdata.insert(vdata.end(), data.begin(), data.end());

            g->mOffset = index;
            index += 4;
        }
    }

    mVBO = createBuffer(GL_ARRAY_BUFFER, vdata.size(), vdata.data(),
                        GL_STATIC_DRAW);

    mIsFontLoaded = true;

    CheckGL("End Font::loadFont");
}

void font_impl::loadSystemFont(const char* const pName) {
    std::string ttf_file_path;

#if !defined(OS_WIN)
    // use fontconfig to get the file
    FcConfig* config = FcInitLoadConfigAndFonts();
    if (!config) {
        FG_ERROR("Fontconfig init failed", FG_ERR_FONTCONFIG_ERROR);
    }
    // configure the search pattern,
    FcPattern* pat = FcNameParse((const FcChar8*)(pName));
    if (!pat) {
        FG_ERROR("Fontconfig name parse failed", FG_ERR_FONTCONFIG_ERROR);
    }

    FcConfigSubstitute(config, pat, FcMatchPattern);
    FcDefaultSubstitute(pat);

    // find the font
    FcResult res;
    FcPattern* font = FcFontMatch(config, pat, &res);

    FcConfigSubstitute(config, pat, FcMatchPattern);
    if (font) {
        FcChar8* file = NULL;
        if (FcPatternGetString(font, FC_FILE, 0, &file) == FcResultMatch) {
            // save the file to another std::string
            ttf_file_path = (char*)file;
        }
        FcPatternDestroy(font);
    }
    // destroy fontconfig pattern object
    FcPatternDestroy(pat);
    // destroy Fc config object
    FcConfigDestroy(config);
#else
    char buf[512];
    GetWindowsDirectory(buf, 512);

    std::regex fontRegex(std::string(pName), std::regex_constants::egrep |
                                                 std::regex_constants::icase);
    std::vector<std::string> fontFiles;
    std::vector<std::string> matchedFontFiles;

    common::getFontFilePaths(fontFiles, std::string(buf) + "\\Fonts\\",
                             std::string("ttf"));
    for (const auto& fontName : fontFiles) {
        if (std::regex_search(fontName, fontRegex)) {
            matchedFontFiles.push_back(fontName);
        }
    }
    /* out of all the possible matches, we choose the
       first possible match for given input font name parameter
    */
    if (matchedFontFiles.size() == 0)
        FT_THROW_ERROR("loadSystemFont failed to find the given font name",
                       FG_ERR_FREETYPE_ERROR);

    ttf_file_path = buf;
    ttf_file_path += "\\Fonts\\";
    ttf_file_path += matchedFontFiles[0];
#endif

    loadFont(ttf_file_path.c_str());
}

void font_impl::render(int pWindowId, const float pPos[], const float pColor[],
                       const char* pText, size_t pFontSize, bool pIsVertical) {
    static const glm::mat4 I(1);

    CheckGL("Begin font_impl::render ");
    if (!mIsFontLoaded) { return; }

    glDepthMask(GL_FALSE);
    glDepthFunc(GL_ALWAYS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    mProgram.bind();

    glUniformMatrix4fv(mPMatIndex, 1, GL_FALSE, (GLfloat*)&mProjMat);
    glUniform4fv(mClrIndex, 1, pColor);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mAtlas->atlasTextureId());
    glUniform1i(mTexIndex, 0);

    bindResources(pWindowId);

    float loc_x = pPos[0];
    float loc_y = pPos[1];

    if (pFontSize < MIN_FONT_SIZE) {
        pFontSize = size_t(MIN_FONT_SIZE);
    } else if (pFontSize > MAX_FONT_SIZE) {
        pFontSize = size_t(MAX_FONT_SIZE);
    }

    glm::mat4 R =
        (pIsVertical ? glm::rotate(I, glm::radians(90.f), glm::vec3(0, 0, 1))
                     : I);

    auto& glyphList = mGlyphLists[pFontSize - size_t(MIN_FONT_SIZE)];

    for (size_t i = 0; i < std::strlen(pText); ++i) {
        int ccode = pText[i];

        if (ccode >= START_CHAR && ccode <= END_CHAR) {
            int idx = ccode - START_CHAR;

            Glyph* g = glyphList[idx];

            if (!pIsVertical) loc_x += g->mBearingX;

            glm::mat4 TR = glm::translate(I, glm::vec3(loc_x, loc_y, 0.0f)) * R;

            glUniformMatrix4fv(mMMatIndex, 1, GL_FALSE, (GLfloat*)&TR);

            glDrawArrays(GL_TRIANGLE_STRIP, GLint(g->mOffset), 4);

            if (pIsVertical) {
                loc_y += (g->mAdvanceX);
            } else {
                loc_x += (g->mAdvanceX - g->mBearingX);
            }
        }
    }

    unbindResources();

    mProgram.unbind();
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);

    CheckGL("End font_impl::render ");
}

}  // namespace opengl
}  // namespace forge
