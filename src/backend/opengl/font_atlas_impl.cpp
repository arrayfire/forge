/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/* =========================================================================
 * The following copyright block is related to freetype-gl project
 * from where the algorithm for Skyline-Bottom Left used in glyph bin packing
 * is used. Hence, the following copyright block is attached in this
 * source file where it is only pertained to.
 ===========================================================================
 */

/* =========================================================================
 * Freetype GL - A C OpenGL Freetype engine
 * Platform:    Any
 * WWW:         https://github.com/rougier/freetype-gl
 * -------------------------------------------------------------------------
 * Copyright 2011,2012 Nicolas P. Rougier. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY NICOLAS P. ROUGIER ''AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL NICOLAS P. ROUGIER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of Nicolas P. Rougier.
 *
 =========================================================================
 */

#include <common/err_handling.hpp>
#include <font_atlas_impl.hpp>
#include <gl_helpers.hpp>

#include <algorithm>
#include <cstring>

namespace forge {
namespace opengl {

static const int BORDER_GAP = 4;

int FontAtlas::fit(const size_t pIndex, const size_t pWidth,
                   const size_t pHeight) {
    auto node     = nodes[pIndex];
    int x         = int(node.x);
    int y         = int(node.y);
    int widthLeft = int(pWidth);
    int i         = int(pIndex);

    if ((x + pWidth) > (mWidth - BORDER_GAP)) { return -1; }

    y = int(node.y);

    while (widthLeft > 0) {
        auto node = nodes[i];
        if (node.y > y) { y = int(node.y); }
        if ((y + pHeight) > (mHeight - BORDER_GAP)) { return -1; }
        widthLeft -= int(node.z);
        ++i;
    }
    return y;
}

void FontAtlas::merge() {
    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        glm::vec3& node = nodes[i];
        auto next       = nodes[i + 1];

        if (node.y == next.y) {
            node.z += next.z;
            nodes.erase(nodes.begin() + (i + 1));
            --i;
        }
    }
}

FontAtlas::FontAtlas(const size_t pWidth, const size_t pHeight,
                     const size_t pDepth)
    : mWidth(pWidth), mHeight(pHeight), mDepth(pDepth), mUsed(0), mId(0) {
    CheckGL("Begin FontAtlas::FontAtlas");
    if (!((pDepth == 1) || (pDepth == 3) || (pDepth == 4))) {
        FG_ERROR("Font Atlas: Invalid depth argument", FG_ERR_INTERNAL);
    }

    glGenTextures(1, &mId);
    glBindTexture(GL_TEXTURE_2D, mId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // one pixel border around the whole atlas to
    // avoid any artefact when sampling texture
    nodes.push_back(glm::vec3(BORDER_GAP, BORDER_GAP, mWidth - BORDER_GAP - 1));

    mData.resize(mWidth * mHeight * mDepth, 0);
    CheckGL("End FontAtlas::FontAtlas");
}

FontAtlas::~FontAtlas() {
    nodes.clear();
    mData.clear();
    if (mId) { glDeleteTextures(1, &mId); }
}

size_t FontAtlas::width() const { return mWidth; }

size_t FontAtlas::height() const { return mHeight; }

size_t FontAtlas::depth() const { return mDepth; }

glm::vec4 FontAtlas::getRegion(const size_t pWidth, const size_t pHeight) {
    glm::vec4 region(0, 0, pWidth, pHeight);

    size_t best_height = UINT_MAX;
    size_t best_width  = UINT_MAX;

    int best_index = -1;
    int y;

    for (size_t i = 0; i < nodes.size(); ++i) {
        y = fit(i, pWidth, pHeight);

        if (y >= 0) {
            auto node = nodes[i];
            if (((y + pHeight) < best_height) ||
                (((y + pHeight) == best_height) && (node.z < best_width))) {
                best_height = y + pHeight;
                best_index  = int(i);
                best_width  = int(node.z);
                region.x    = node.x;
                region.y    = float(y);
            }
        }
    }

    if (best_index == -1) {
        region.x = -1;
        region.y = -1;
        region.z = 0;
        region.w = 0;
        return region;
    }

    glm::vec3 node(region.x, region.y + pHeight, pWidth);

    nodes.insert(nodes.begin() + best_index, node);

    for (size_t i = best_index + 1; i < nodes.size(); ++i) {
        glm::vec3& node = nodes[i];
        auto prev       = nodes[i - 1];

        if (node.x < (prev.x + prev.z)) {
            int shrink = int(prev.x + prev.z - node.x);
            node.x += shrink;
            node.z -= shrink;
            if (node.z <= 0) {
                nodes.erase(nodes.begin() + i);
                --i;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    merge();
    mUsed += pWidth * pHeight;

    return region;
}

bool FontAtlas::setRegion(const size_t pX, const size_t pY, const size_t pWidth,
                          const size_t pHeight, const unsigned char* pData,
                          const size_t pStride) {
    if (pX > 0 && pY > 0 && pX < (mWidth - BORDER_GAP) &&
        (pX + pWidth) <= (mWidth - BORDER_GAP) && pY < (mHeight - BORDER_GAP) &&
        (pY + pHeight) <= (mHeight - BORDER_GAP)) {
        size_t depth    = mDepth;
        size_t charsize = sizeof(unsigned char);

        for (size_t i = 0; i < pHeight; ++i) {
            std::memcpy(
                mData.data() + ((pY + i) * mWidth + pX) * charsize * depth,
                pData + (i * pStride) * charsize, pWidth * charsize * depth);
        }
        return true;
    }
    return false;
}

void FontAtlas::upload() {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, mId);

    if (mDepth == 4) {
#ifdef GL_UNSIGNED_INT_8_8_8_8_REV
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, GLsizei(mWidth),
                     GLsizei(mHeight), 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                     mData.data());
#else
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, GLsizei(mWidth),
                     GLsizei(mHeight), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     mData.data());
#endif
    } else if (mDepth == 3) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, GLsizei(mWidth),
                     GLsizei(mHeight), 0, GL_RGB, GL_UNSIGNED_BYTE,
                     mData.data());
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, GLsizei(mWidth),
                     GLsizei(mHeight), 0, GL_RED, GL_UNSIGNED_BYTE,
                     mData.data());
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
}

void FontAtlas::clear() {
    mUsed = 0;

    nodes.clear();
    nodes.push_back(glm::vec3(BORDER_GAP, BORDER_GAP, mWidth - BORDER_GAP - 1));

    std::fill(mData.begin(), mData.end(), 0);
}

GLuint FontAtlas::atlasTextureId() const { return mId; }

}  // namespace opengl
}  // namespace forge
