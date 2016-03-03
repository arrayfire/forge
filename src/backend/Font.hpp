/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <backend.hpp>
#include <font.hpp>

#include <memory>

namespace common
{

class Font {
    private:
        std::shared_ptr<detail::font_impl> fnt;

    public:
        Font() : fnt(std::make_shared<detail::font_impl>()) {}

        const std::shared_ptr<detail::font_impl>& impl() const {
            return fnt;
        }

        inline void setOthro2D(int pWidth, int pHeight) {
            fnt->setOthro2D(pWidth, pHeight);
        }

        inline void loadFont(const char* const pFile) {
            fnt->loadFont(pFile);
        }

        inline void loadSystemFont(const char* const pName) {
            fnt->loadSystemFont(pName);
        }
};

}
