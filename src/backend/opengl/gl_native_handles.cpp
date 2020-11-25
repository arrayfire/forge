/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gl_native_handles.hpp>

#if defined(OS_WIN)
#include <windows.h>
#elif defined(OS_MAC)
#include <OpenGL/OpenGL.h>
#else
#include <GL/glx.h>
#endif

namespace forge {
namespace opengl {

ContextHandle getCurrentContextHandle() {
    auto id = ContextHandle{0};

#if defined(OS_WIN)
    const auto context = wglGetCurrentContext();
#elif defined(OS_LNX)
    const auto context = glXGetCurrentContext();
#else
    const auto context = CGLGetCurrentContext();
#endif
    id = reinterpret_cast<ContextHandle>(context);

    return id;
}

DisplayHandle getCurrentDisplayHandle() {
    auto id = DisplayHandle{0};

#if defined(OS_WIN)
    const auto display = wglGetCurrentDC();
#elif defined(OS_LNX)
    const auto display = glXGetCurrentDisplay();
#else
    const DisplayHandle display = 0;
#endif
    id = reinterpret_cast<DisplayHandle>(display);

    return id;
}

}  // namespace opengl
}  // namespace forge
