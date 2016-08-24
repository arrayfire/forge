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
    try {
        mValue = getHandle(new common::Font());
    } CATCH_INTERNAL_TO_EXTERNAL
}

Font::Font(const Font& other)
{
    try {
        mValue = getHandle(new common::Font(other.get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Font::~Font()
{
    delete getFont(mValue);
}

void Font::loadFontFile(const char* const pFile)
{
    try {
        getFont(mValue)->loadFont(pFile);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Font::loadSystemFont(const char* const pName)
{
    try {
        getFont(mValue)->loadSystemFont(pName);
    } CATCH_INTERNAL_TO_EXTERNAL
}

fg_font Font::get() const
{
    try {
        return getFont(mValue);
    } CATCH_INTERNAL_TO_EXTERNAL
}

}
