/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/font.h>
#include <fg/exception.h>
#include <common.hpp>
#include <cmath>
#include <algorithm>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <ft2build.h>
#include <freetype.h>
#include <ftglyph.h>
#include FT_FREETYPE_H

#ifndef WINDOWS_OS
#include <fontconfig/fontconfig.h>
#endif


static const char* gFontVertShader =
"#version 330\n"
"uniform mat4 projectionMatrix;\n"
"uniform mat4 modelViewMatrix;\n"
"layout (location = 0) in vec2 inPosition;\n"
"layout (location = 1) in vec2 inCoord;\n"
"out vec2 texCoord;\n"
"void main()\n"
"{\n"
"    gl_Position = projectionMatrix*modelViewMatrix*vec4(inPosition, 0.0, 1.0);\n"
"    texCoord = inCoord;\n"
"}\n";

static const char* gFontFragShader =
"#version 330\n"
"in vec2 texCoord;\n"
"out vec4 outputColor;\n"
"uniform sampler2D tex;\n"
"uniform vec4 textColor;\n"
"void main()\n"
"{\n"
"    vec4 texC = texture(tex, texCoord);\n"
"    vec4 alpha = vec4(1.0, 1.0, 1.0, texC.r);\n"
"    outputColor = alpha*textColor;\n"
"}\n";

/* freetype library types */
static FT_Library  gFTLib;
static FT_Face     gFTFace;

namespace fg
{

#define FT_THROW_ERROR(msg, err) \
    throw fg::Error("Freetype library", __LINE__, msg, err);

void Font::extractGlyph(int pCharacter)
{
    FT_Load_Glyph(gFTFace, FT_Get_Char_Index(gFTFace, pCharacter), FT_LOAD_DEFAULT);
    FT_Render_Glyph(gFTFace->glyph, FT_RENDER_MODE_NORMAL);
    FT_Bitmap& bitmap = gFTFace->glyph->bitmap;

    int pIndex = pCharacter - START_CHAR;

    int bmp_w = bitmap.width;
    int bmp_h = bitmap.rows;
    int w     = next_p2(bmp_w);
    int h     = next_p2(bmp_h);

    std::vector<unsigned char> glyphData(w*h, 0);
    for (int j=0; j<h; ++j) {
        for (int i=0; i<w; ++i) {
            glyphData[ (j*w+i) ] =
                (i<bmp_w && j<bmp_h ? bitmap.buffer[(bmp_h-1-j)*bmp_w+i] : 0);
        }
    }

    CheckGL("Before Character texture creation");
    // texture from it
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glGenTextures(1, &(mCharTextures[pIndex]));
    glBindTexture(GL_TEXTURE_2D, mCharTextures[pIndex]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE,
        (pCharacter==32 ? NULL : &glyphData.front()));
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    CheckGL("After Character texture creation");

    mAdvX[pIndex] = gFTFace->glyph->advance.x>>6;
    mBearingX[pIndex] = gFTFace->glyph->metrics.horiBearingX>>6;
    mCharWidth[pIndex] = gFTFace->glyph->metrics.width>>6;

    mAdvY[pIndex] = (gFTFace->glyph->metrics.height - gFTFace->glyph->metrics.horiBearingY)>>6;
    mBearingY[pIndex] = gFTFace->glyph->metrics.horiBearingY>>6;
    mCharHeight[pIndex] = gFTFace->glyph->metrics.height>>6;

    mNewLine = std::max(mNewLine, int(gFTFace->glyph->metrics.height>>6));

    // Rendering data, texture coordinates are always the same,
    // so now we waste a little memory
    float quad[8];
    float quad_texcoords[8] = { 0, 1, 0, 0, 1, 1, 1, 0};

    quad[0] = 0.0f; quad[1] = float(-mAdvY[pIndex]+h);
    quad[2] = 0.0f; quad[3] = float(-mAdvY[pIndex]);
    quad[4] = float(w); quad[5] = float(-mAdvY[pIndex]+h);
    quad[6] = float(w); quad[7] = float(-mAdvY[pIndex]);

    // texture coordinates are duplicated for each character
    // TODO work on removing this duplicates to reduce memory usage
    for (int i=0; i<4; ++i) {
        float* vert_ptr = quad+2*i;
        float* tex_ptr  = quad_texcoords+2*i;
        mVertexData.insert(mVertexData.end(), vert_ptr, vert_ptr+2);
        mVertexData.insert(mVertexData.end(), tex_ptr, tex_ptr+2);
    }
}

void Font::destroyGLResources()
{
    if (mIsFontLoaded) {
        if (mProgram) glDeleteProgram(mProgram);
        if (mVAO) glDeleteVertexArrays(1, &mVAO);
        if (mVBO) glDeleteBuffers(1, &mVBO);
        glDeleteTextures(NUM_CHARS, mCharTextures);
    }
}

Font::Font()
    : mIsFontLoaded(false), mTTFfile(""), mVAO(0), mVBO(0), mProgram(0), mSampler(0)
{
    mProgram = initShaders(gFontVertShader, gFontFragShader);

    memset(mCharTextures, 0, NUM_CHARS);

    glGenSamplers(1, &mSampler);
    glSamplerParameteri(mSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(mSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(mSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(mSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

Font::~Font()
{
    destroyGLResources();
    if (mProgram) glDeleteProgram(mProgram);
    if (mSampler) glDeleteSamplers(1, &mSampler);
}

void Font::loadFont(const char* const pFile, int pFontSize)
{
    if (mIsFontLoaded) {
        if (pFile==mTTFfile)
            return;
        else {
            destroyGLResources();
            mIsFontLoaded = false;
        }
    }
    mLoadedPixelSize = pFontSize;

    CheckGL("Begin Font::loadFont");
    // Initialize freetype font library
    bool bError = FT_Init_FreeType(&gFTLib);

    bError = FT_New_Face(gFTLib, pFile, 0, &gFTFace);
    if(bError) {
        FT_Done_FreeType(gFTLib);
        FT_THROW_ERROR("font face creation failed", FG_ERR_FREETYPE_ERROR);
    }

    bError = FT_Set_Pixel_Sizes(gFTFace, 0, pFontSize);
    if (bError) {
        FT_Done_Face(gFTFace);
        FT_Done_FreeType(gFTLib);
        FT_THROW_ERROR("set font size failed", FG_ERR_FREETYPE_ERROR);
    }

    // read font glyphs for only characters
    // from ' ' to '~'
    for (int i=START_CHAR; i<END_CHAR; ++i) extractGlyph(i);

    FT_Done_Face(gFTFace);
    FT_Done_FreeType(gFTLib);

    size_t size = sizeof(glm::vec2);

    mVBO = createBuffer<float>(mVertexData.size(), &mVertexData.front(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &mVAO);
    glBindVertexArray(mVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, size*2, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, size*2, (void*)(size));
    glBindVertexArray(0);

    mIsFontLoaded = true;
    mTTFfile = pFile;
    CheckGL("End Font::loadFont");
}

void Font::loadSystemFont(const char* const pName, int pFontSize)
{
    //TODO do error checking once it is working
    std::string ttf_file_path;

#ifndef WINDOWS_OS
    // use fontconfig to get the file
    FcConfig* config = FcInitLoadConfigAndFonts();
    if (!config) {
        FT_THROW_ERROR("fontconfig initilization failed", FG_ERR_FREETYPE_ERROR);
    }
    // configure the search pattern,
    FcPattern* pat = FcNameParse((const FcChar8*)(pName));
    if (!pat) {
        FT_THROW_ERROR("fontconfig pattern creation failed", FG_ERR_FREETYPE_ERROR);
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
    ttf_file_path = buf;
    ttf_file_path += "\\Fonts\\";
    ttf_file_path += pName;
    ttf_file_path += ".ttf";
#endif

    loadFont(ttf_file_path.c_str(), pFontSize);
}

void Font::setOthro2D(int pWidth, int pHeight)
{
    mHeight = pHeight;
    mWidth = pWidth;
}

void Font::render(const float pPos[], const float pColor[], std::string pText, int pFontSize, bool pIsVertical)
{
    if(!mIsFontLoaded)return;

    glDisable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(mProgram);
    GLuint pmat_loc = glGetUniformLocation(mProgram, "projectionMatrix");
    GLuint mvmat_loc = glGetUniformLocation(mProgram, "modelViewMatrix");
    GLuint tex_loc = glGetUniformLocation(mProgram, "tex");
    GLuint col_loc = glGetUniformLocation(mProgram, "textColor");

    glm::mat4 projMat = glm::ortho(0.0f, float(mWidth), 0.0f, float(mHeight));

    glUniformMatrix4fv(pmat_loc, 1, GL_FALSE, (GLfloat*)&projMat);
    glUniform4fv(col_loc, 1, pColor);

    int loc_x = pPos[0], loc_y = pPos[1];
    if(pFontSize == -1)pFontSize = mLoadedPixelSize;
    float scale_factor = float(pFontSize)/float(mLoadedPixelSize);

    for (std::string::iterator it=pText.begin(); it!=pText.end(); ++it) {
        char currChar = *it;

        if(currChar == '\n') {
            // if it is new line, move location to next line
            loc_x = pPos[0];
            loc_y -= mNewLine*pFontSize/mLoadedPixelSize;
        } else if (currChar <= '~' && currChar >= ' ') {
            // regular characters are rendered as textured quad
            int idx = int(currChar) - START_CHAR;
            loc_x += mBearingX[idx]*pFontSize/mLoadedPixelSize;

            glActiveTexture(GL_TEXTURE0);
            glUniform1i(tex_loc, 0);
            glBindTexture(GL_TEXTURE_2D, mCharTextures[idx]);
            glBindSampler(0, mSampler);

            /* rotate by 90 degress if we need
             * to render the characters vertically */
            glm::mat4 modelView = glm::translate(glm::mat4(1.0f),
                    glm::vec3(float(loc_x), float(loc_y), 0.0f));

            modelView = glm::scale(modelView, glm::vec3(scale_factor));
            glUniformMatrix4fv(mvmat_loc, 1, GL_FALSE, (GLfloat*)&modelView);

            // Draw letter
            glBindVertexArray(mVAO);
            glDrawArrays(GL_TRIANGLE_STRIP, idx*4, 4);
            CheckGL("Draw pCharacter");
            glBindVertexArray(0);

            loc_x += (mAdvX[idx]-mBearingX[idx])*pFontSize/mLoadedPixelSize;
        }
        /* if the text needs to be rendered vertically,
         * move the pen cursor to next line after each
         * character render mandatorily */
        if (pIsVertical) {
            loc_x = pPos[0];
            loc_y -= mNewLine*pFontSize/mLoadedPixelSize;
        }
    }

    glUseProgram(0);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
}

}
