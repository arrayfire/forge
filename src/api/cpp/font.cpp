/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/font.h>

#include <handle.hpp>
#include <font.hpp>

namespace forge
{

Font::Font()
{
    mValue = getHandle(new common::Font());
}

Font::Font(const Font& other)
{
    mValue = getHandle(new common::Font(other.get()));
}

Font::~Font()
{
    delete getFont(mValue);
}

void Font::loadFontFile(const char* const pFile)
{
    getFont(mValue)->loadFont(pFile);
}

void Font::loadSystemFont(const char* const pName)
{
    getFont(mValue)->loadSystemFont(pName);
}

fg_font Font::get() const
{
    return getFont(mValue);
}

}
