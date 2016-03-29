/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/font.h>

#include <common.hpp>
#include <err_opengl.hpp>
#include <font_impl.hpp>
#include <shader_headers/font_vs.hpp>
#include <shader_headers/font_fs.hpp>

#include <cmath>
#include <cstring>
#include <sstream>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_STROKER_H

#undef __FTERRORS_H__
#define FT_ERRORDEF( e, v, s )  { e, s },
#define FT_ERROR_START_LIST     {
#define FT_ERROR_END_LIST       { 0, 0 } };
static const struct {
    int          code;
    const char*  message;
} FT_Errors[] =
#include FT_ERRORS_H

#ifndef OS_WIN
#include <fontconfig/fontconfig.h>
#endif

#ifdef OS_WIN
#include <windows.h>
#include <regex>
#endif

#define START_CHAR 32
#define END_CHAR   126

/* freetype library types */

namespace opengl
{

#ifdef NDEBUG
/* Relase Mode */
#define FT_THROW_ERROR(msg, error) \
    throw fg::Error("Freetype library", __LINE__, msg, error);

#else
/* Debug Mode */
#define FT_THROW_ERROR(msg, err)                                            \
    do {                                                                    \
        std::ostringstream ss;                                              \
        ss << "FT_Error (0x"<< std::hex << FT_Errors[err].code <<") : "     \
           << FT_Errors[err].message << std::endl;                          \
        throw fg::Error(ss.str().c_str(), __LINE__, msg, err);              \
    } while(0);

#endif

void font_impl::loadAtlasWithGlyphs(const size_t pFontSize)
{
    FT_Library  library;
    FT_Face     face;
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
    bError = FT_Set_Pixel_Sizes(face, 0, pFontSize);
    if (bError) {
        FT_Done_Face(face);
        FT_Done_FreeType(library);
        FT_THROW_ERROR("Freetype char size set failed", FG_ERR_FREETYPE_ERROR);
    }

    size_t missed = 0;

    /* retrieve the list of current font size */
    auto& currList = mGlyphLists[pFontSize-MIN_FONT_SIZE];

    for (size_t i=0; i<(END_CHAR-START_CHAR+1); ++i)
    {
        FT_ULong ccode = (FT_ULong)(START_CHAR + i);

        FT_UInt glyphIndex = FT_Get_Char_Index(face, ccode);

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

        FT_Glyph currGlyph;;

        bError = FT_Get_Glyph(face->glyph, &currGlyph);
        if (bError) {
            FT_Done_Face(face);
            FT_Done_FreeType(library);
            FT_THROW_ERROR("FT_Get_Glyph", FG_ERR_FREETYPE_ERROR);
        }

        ////FIXME Renable when outline strokes are working
        ///* use stroker to get outline */
        //FT_Stroker stroker;
        //bError = FT_Stroker_New(library, &stroker);
        //if (bError) {
        //    FT_Stroker_Done(stroker);
        //    FT_Done_Face(face);
        //    FT_Done_FreeType(library);
        //    FT_THROW_ERROR("FT_Stroker_New", fg::FG_ERR_FREETYPE_ERROR);
        //}

        //FT_Stroker_Set(stroker, 16, FT_STROKER_LINECAP_ROUND, FT_STROKER_LINEJOIN_ROUND, 0);

        ///* stroke the outline to current glyph */
        //bError = FT_Glyph_Stroke(&currGlyph, stroker, 1);
        //if (bError) {
        //    FT_Stroker_Done(stroker);
        //    FT_Done_Face(face);
        //    FT_Done_FreeType(library);
        //    FT_THROW_ERROR("FT_Glyph_Stroke", fg::FG_ERR_FREETYPE_ERROR);
        //}
        //FT_Stroker_Done(stroker);

        /* fixed channel depth of 1 */
        bError = FT_Glyph_To_Bitmap(&currGlyph, FT_RENDER_MODE_NORMAL, 0, 1);
        if (bError) {
            //FIXME Renable when outline strokes are working
            //FT_Stroker_Done(stroker);
            FT_Done_Face(face);
            FT_Done_FreeType(library);
            FT_THROW_ERROR("FT_Glyph_To_Bitmap", FG_ERR_FREETYPE_ERROR);
        }

        FT_BitmapGlyph bmpGlyph = (FT_BitmapGlyph) currGlyph;
        FT_Bitmap bmp = bmpGlyph->bitmap;

        int w = bmp.width + 1;
        int h = bmp.rows + 1;

        glm::vec4 region = mAtlas->getRegion(w, h);

        if (region.x<0 || region.y<0) {
            missed++;
            std::cerr<<"Texture atlas is full"<<std::endl;
            continue;
        }

        w = w-1; // reduce by one again to leave one pixel border
        h = h-1; // reduce by one again to leave one pixel border

        int x = region.x;
        int y = region.y;

        mAtlas->setRegion(x, y, w, h, bmp.buffer, bmp.pitch);

        Glyph* glyph = new Glyph();

        glyph->mWidth    = w;
        glyph->mHeight   = h;

        glyph->mBearingX = face->glyph->metrics.horiBearingX>>6;
        glyph->mBearingY = face->glyph->metrics.horiBearingY>>6;

        glyph->mAdvanceX = face->glyph->advance.x>>6;
        glyph->mAdvanceY = (face->glyph->metrics.height - face->glyph->metrics.horiBearingY)>>6;

        glyph->mS0       = x/(float)mAtlas->width();
        glyph->mT1       = y/(float)mAtlas->height();
        glyph->mS1       = (x + glyph->mWidth)/(float)mAtlas->width();
        glyph->mT0       = (y + glyph->mHeight)/(float)mAtlas->height();

        currList.push_back(glyph);

        FT_Done_Glyph(currGlyph);
    }

    /* cleanup freetype variables */
    FT_Done_Face(face);
    FT_Done_FreeType(library);
}

void font_impl::bindResources(int pWindowId)
{
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        size_t sz = 2*sizeof(float);
        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sz, 0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2*sz, reinterpret_cast<void*>(sz));
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }
    glBindVertexArray(mVAOMap[pWindowId]);
}

void font_impl::unbindResources() const
{
    glBindVertexArray(0);
}

void font_impl::destroyGLResources()
{
    if (mVBO)
        glDeleteBuffers(1, &mVBO);
    /* remove all glyph structures from heap */
    for (auto it: mGlyphLists) {
        /* for each font size glyph list */
        for (auto& m : it) {
            delete m; /* delete Glyph structure */
        }
        it.clear();
    }
    /* clear list */
    mGlyphLists.clear();
}

font_impl::font_impl()
    : mTTFfile(""), mIsFontLoaded(false), mAtlas(new FontAtlas(1024, 1024, 1)),
    mVBO(0), mProgram(0), mOrthoW(1), mOrthoH(1)
{
    mProgram   = initShaders(glsl::font_vs.c_str(), glsl::font_fs.c_str());
    mPMatIndex = glGetUniformLocation(mProgram, "projectionMatrix");
    mMMatIndex = glGetUniformLocation(mProgram, "modelViewMatrix");
    mTexIndex  = glGetUniformLocation(mProgram, "tex");
    mClrIndex  = glGetUniformLocation(mProgram, "textColor");

    mGlyphLists.resize(MAX_FONT_SIZE-MIN_FONT_SIZE+1, GlyphList());
}

font_impl::~font_impl()
{
    destroyGLResources();
    if (mProgram) glDeleteProgram(mProgram);
}

void font_impl::setOthro2D(int pWidth, int pHeight)
{
    mOrthoW  = pWidth;
    mOrthoH  = pHeight;
    mProjMat = glm::ortho(0.0f, float(mOrthoW), 0.0f, float(mOrthoH));
}

void font_impl::loadFont(const char* const pFile)
{
    CheckGL("Begin font_impl::loadFont");

    /* Check if font is already loaded. If yes, check if current font load
     * request is same as earlier. If so, return from the function, otherwise,
     * cleanup currently used resources and mark accordingly for subsequent
     * font loading.*/
    if (mIsFontLoaded) {
        if (pFile==mTTFfile)
            return;
        else {
            destroyGLResources();
            mIsFontLoaded = false;
        }
    }

    mTTFfile = pFile;
    /* Load different font sizes into font atlas */
    for (size_t s=MIN_FONT_SIZE; s<=MAX_FONT_SIZE; ++s) {
        loadAtlasWithGlyphs(s);
    }

    mAtlas->upload();

    /* push each glyphs vertex and texture data into VBO */
    std::vector<float> vdata;

    uint index = 0;

    for (size_t f=0; f<mGlyphLists.size(); ++f)
    {
        auto& list = mGlyphLists[f];

        for (size_t l=0; l<list.size(); ++l)
        {
            Glyph* g = list[l];

            std::vector<float> data(16, 0.0f);
            data[0] = 0.0f; data[1] = float(-g->mAdvanceY+g->mHeight);
            data[2] = g->mS0; data[3] = g->mT1;

            data[4] = 0.0f; data[5] = float(-g->mAdvanceY);
            data[6] = g->mS0; data[7] = g->mT0;

            data[8] = float(g->mWidth); data[9] = float(-g->mAdvanceY+g->mHeight);
            data[10] = g->mS1; data[11] = g->mT1;

            data[12] = float(g->mWidth); data[13] = float(-g->mAdvanceY);
            data[14] = g->mS1; data[15] = g->mT0;

            vdata.insert(vdata.end(), data.begin(), data.end());

            g->mOffset = index;
            index += 4;
        }
    }

    mVBO = createBuffer(GL_ARRAY_BUFFER, vdata.size(), vdata.data(), GL_STATIC_DRAW);

    mIsFontLoaded = true;

    CheckGL("End Font::loadFont");
}

void font_impl::loadSystemFont(const char* const pName)
{
    std::string ttf_file_path;

#ifndef OS_WIN
    // use fontconfig to get the file
    FcConfig* config = FcInitLoadConfigAndFonts();
    if (!config) {
        throw fg::Error("Fontconfig init failed",
                        __LINE__, __PRETTY_FUNCTION__,
                        FG_ERR_FONTCONFIG_ERROR);
    }
    // configure the search pattern,
    FcPattern* pat = FcNameParse((const FcChar8*)(pName));
    if (!pat) {
        throw fg::Error("Fontconfig name parse failed",
                        __LINE__, __PRETTY_FUNCTION__,
                        FG_ERR_FONTCONFIG_ERROR);
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
#else
    char buf[512];
    GetWindowsDirectory(buf, 512);

    std::regex fontRegex(std::string(pName), std::regex_constants::egrep | std::regex_constants::icase);
    std::vector<std::string> fontFiles;
    std::vector<std::string> matchedFontFiles;

    getFontFilePaths(fontFiles, std::string(buf)+"\\Fonts\\", std::string("ttf"));
    for (const auto &fontName : fontFiles) {
        if (std::regex_search(fontName, fontRegex)) {
            matchedFontFiles.push_back(fontName);
        }
    }
    /* out of all the possible matches, we choose the
       first possible match for given input font name parameter
    */
    if (matchedFontFiles.size()==0)
        FT_THROW_ERROR("loadSystemFont failed to find the given font name", fg::FG_ERR_FREETYPE_ERROR);

    ttf_file_path = buf;
    ttf_file_path += "\\Fonts\\";
    ttf_file_path += matchedFontFiles[0];
#endif

    loadFont(ttf_file_path.c_str());
}

void font_impl::render(int pWindowId,
                       const float pPos[], const float pColor[], const char* pText,
                       size_t pFontSize, bool pIsVertical)
{
    static const glm::mat4 I(1);

    CheckGL("Begin font_impl::render ");
    if(!mIsFontLoaded) {
        return;
    }

    glDepthMask(GL_FALSE);
    glDepthFunc(GL_ALWAYS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(mProgram);

    glUniformMatrix4fv(mPMatIndex, 1, GL_FALSE, (GLfloat*)&mProjMat);
    glUniform4fv(mClrIndex, 1, pColor);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mAtlas->atlasTextureId());
    glUniform1i(mTexIndex, 0);

    bindResources(pWindowId);

    float loc_x = pPos[0];
    float loc_y = pPos[1];

    if (pFontSize<MIN_FONT_SIZE) {
       pFontSize = MIN_FONT_SIZE;
    }
    else if (pFontSize>MAX_FONT_SIZE) {
        pFontSize = MAX_FONT_SIZE;
    }

    glm::mat4 R = (pIsVertical ? glm::rotate(I, glm::radians(90.f), glm::vec3(0,0,1)) : I);

    auto& glyphList = mGlyphLists[pFontSize - MIN_FONT_SIZE];

    for (size_t i=0; i<std::strlen(pText); ++i)
    {
        int ccode = pText[i];

        if (ccode>=START_CHAR && ccode<=END_CHAR) {

            int idx = ccode - START_CHAR;

            Glyph* g = glyphList[idx];

            if (!pIsVertical)
                loc_x += g->mBearingX;

            glm::mat4 TR = glm::translate(I, glm::vec3(loc_x, loc_y, 0.0f)) * R;

            glUniformMatrix4fv(mMMatIndex, 1, GL_FALSE, (GLfloat*)&TR);

            glDrawArrays(GL_TRIANGLE_STRIP, g->mOffset, 4);

            if (pIsVertical) {
                loc_y += (g->mAdvanceX);
            } else {
                loc_x += (g->mAdvanceX-g->mBearingX);
            }
        }
    }

    unbindResources();

    glUseProgram(0);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);

    CheckGL("End font_impl::render ");
}

}
