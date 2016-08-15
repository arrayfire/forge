/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fg/defines.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup font_functions
 *  @{
 */

/**
   Create a Font object

   \param[out] pFont will point to the font object created after this function returns

   \return \ref fg_err error code
 */
FGAPI fg_err fg_create_font(fg_font* pFont);

/**
   Destroy font object

   \param[in] pFont is the font handle

   \return \ref fg_err error code
 */
FGAPI fg_err fg_destroy_font(fg_font pFont);

/**
   Load a given font file

   \param[in] pFont is the font handle
   \param[in] pFileFullPath True Type Font file path

   \return \ref fg_err error code
 */
FGAPI fg_err fg_load_font_file(fg_font pFont, const char* const pFileFullPath);

/**
   Load a system font based on the name

   \param[in] pFont is the font handle
   \param[in] pFontName True Type Font name

   \return \ref fg_err error code
 */
FGAPI fg_err fg_load_system_font(fg_font pFont, const char* const pFontName);

/** @} */

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

namespace forge
{

/**
   \class Font

   \brief Font object is essentially a resource handler for the specific font you want to use
 */
class Font {
    private:
        fg_font mValue;

    public:
        /**
           Creates Font object
         */
        FGAPI Font();

        /**
           Copy constructor for Font

           \param[in] other is the Font object of which we make a copy of, this is not a deep copy.
         */
        FGAPI Font(const Font& other);

        /**
           Font Destructor
         */
        FGAPI ~Font();

        /**
           Load a given font file

           \param[in] pFile True Type Font file path
         */
        FGAPI void loadFontFile(const char* const pFile);

        /**
           Load a system font based on the name

           \param[in] pName True Type Font name
         */
        FGAPI void loadSystemFont(const char* const pName);

        /**
           Get handle for internal implementation of Font object
         */
        FGAPI fg_font get() const;
};

}

#endif
