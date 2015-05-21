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
#include <memory>

namespace fg
{

class Font {
    private:
        std::shared_ptr<internal::_Font> value;

    public:
        FGAPI Font();

        FGAPI void loadFont(const char* const pFile, int pFontSize);
        FGAPI void loadSystemFont(const char* const pName, int pFontSize);

        FGAPI void setOthro2D(int pWidth, int pHeight);
        FGAPI internal::_Font* get() const;

        FGAPI void render(const float pPos[2], const float pColor[4], 
                          const char* pText, int pFontSize=-1, bool pIsVertical=false);
};

}