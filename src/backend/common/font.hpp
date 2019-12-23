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
#include <fg/defines.h>
#include <font_impl.hpp>

#include <memory>

namespace forge {
namespace common {

class Font {
   private:
    std::shared_ptr<detail::font_impl> mFont;

   public:
    Font() : mFont(std::make_shared<detail::font_impl>()) {}

    Font(const fg_font pOther) {
        mFont = reinterpret_cast<Font*>(pOther)->impl();
    }

    const std::shared_ptr<detail::font_impl>& impl() const { return mFont; }

    inline void setOthro2D(int pWidth, int pHeight) {
        mFont->setOthro2D(pWidth, pHeight);
    }

    inline void loadFont(const char* const pFile) { mFont->loadFont(pFile); }

    inline void loadSystemFont(const char* const pName) {
        mFont->loadSystemFont(pName);
    }
};

}  // namespace common
}  // namespace forge
