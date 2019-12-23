/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/font.hpp>
#include <common/handle.hpp>
#include <fg/font.h>

using namespace forge;

using forge::common::getFont;

fg_err fg_create_font(fg_font* pFont) {
    try {
        *pFont = getHandle(new common::Font());
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_font(fg_font* pOut, fg_font pIn) {
    try {
        common::Font* temp = new common::Font(getFont(pIn));
        *pOut              = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_font(fg_font pFont) {
    try {
        delete getFont(pFont);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_load_font_file(fg_font pFont, const char* const pFileFullPath) {
    try {
        getFont(pFont)->loadFont(pFileFullPath);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_load_system_font(fg_font pFont, const char* const pFontName) {
    try {
        getFont(pFont)->loadSystemFont(pFontName);
    }
    CATCHALL

    return FG_ERR_NONE;
}
