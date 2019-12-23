/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/**
 * @author Pradeep Garigipati (pradeep@arrayfire.com)
 *
 * A font atlas is used to pack several small regions into a single texture.
 *
 * It is an implementation of Skyline Bottom-Left algorithm described
 * in the article by Jukka Jylänki : "A  Thousand Ways to Pack the Bin -
 * A Practical Approach to Two-Dimensional Rectangle Bin Packing",
 * February 27, 2010. Following code also loosely follows C++ sources provided
 * by Jukka Jylänki at: http://clb.demon.fi/files/RectangleBinPack/ for the
 * implementation of the Skyline Bottom-Left algorithm.
 */

#pragma once

#include <common/defines.hpp>

namespace forge {
namespace opengl {

class FontAtlas {
   private:
    size_t mWidth;
    size_t mHeight;
    size_t mDepth;
    size_t mUsed;
    uint32_t mId;

    std::vector<unsigned char> mData;
    std::vector<glm::vec3> nodes;

    /* helper functions */
    int fit(const size_t pIndex, const size_t pWidth, const size_t pHeight);
    void merge();

   public:
    FontAtlas(const size_t pWidth, const size_t pHeight, const size_t pDepth);
    ~FontAtlas();

    size_t width() const;
    size_t height() const;
    size_t depth() const;

    glm::vec4 getRegion(const size_t pWidth, const size_t pHeight);
    bool setRegion(const size_t pX, const size_t pY, const size_t pWidth,
                   const size_t pHeight, const unsigned char* pData,
                   const size_t pStride);

    void upload();
    void clear();

    uint32_t atlasTextureId() const;
};

struct Glyph {
    size_t mWidth;
    size_t mHeight;

    int mBearingX;
    int mBearingY;

    float mAdvanceX;
    float mAdvanceY;

    /* normalized texture coordinate (x) of top-left corner */
    float mS0, mT0;

    /* First normalized texture coordinate (x) of bottom-right corner */
    float mS1, mT1;

    /* render quad vbo offset */
    size_t mOffset;
};

}  // namespace opengl
}  // namespace forge
