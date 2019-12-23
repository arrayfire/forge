/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/font.h>

#include <error.hpp>

#include <utility>

namespace forge {
Font::Font() {
    fg_font temp = 0;
    FG_THROW(fg_create_font(&temp));
    std::swap(mValue, temp);
}

Font::Font(const Font& other) {
    fg_font temp = 0;
    FG_THROW(fg_retain_font(&temp, other.get()));
    std::swap(mValue, temp);
}

Font::~Font() { fg_release_font(get()); }

void Font::loadFontFile(const char* const pFile) {
    FG_THROW(fg_load_font_file(get(), pFile));
}

void Font::loadSystemFont(const char* const pName) {
    FG_THROW(fg_load_system_font(get(), pName));
}

fg_font Font::get() const { return mValue; }
}  // namespace forge
