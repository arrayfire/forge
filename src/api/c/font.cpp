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
#include <Font.hpp>

fg_err fg_create_font(fg_font* pFont)
{
    try {
        *pFont = getHandle(new common::Font());
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_destroy_font(fg_font pFont)
{
    try {
        delete getFont(pFont);
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_load_font_file(fg_font pFont, const char* const pFileFullPath)
{
    try {
        getFont(pFont)->loadFont(pFileFullPath);
    }
    CATCHALL

    return FG_SUCCESS;
}

fg_err fg_load_system_font(fg_font pFont, const char* const pFontName)
{
    try {
        getFont(pFont)->loadSystemFont(pFontName);
    }
    CATCHALL

    return FG_SUCCESS;
}
