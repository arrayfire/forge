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

namespace internal
{
class _Font;
}

namespace fg
{

/**
   \class Font

   \brief Font object is essentially a resource handler for the specific font you want to use
 */
class Font {
    private:
        internal::_Font* value;

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
        FGAPI internal::_Font* get() const;
};

}
