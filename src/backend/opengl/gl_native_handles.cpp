/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <gl_native_handles.hpp>

#ifdef OS_WIN
#include <windows.h>
#elif OS_MAC
#include <OpenGL/OpenGL.h>
#else
#include <GL/glx.h>
#endif

namespace opengl
{

ContextHandle getCurrentContextHandle()
{
    auto id = ContextHandle{0};

#ifdef SYSTEM_WINDOWS
    const auto context = wglGetCurrentContext();
#elif SYSTEM_DARWIN
    const auto context = CGLGetCurrentContext();
#else
    const auto context = glXGetCurrentContext();
#endif
    id = reinterpret_cast<ContextHandle>(context);

    return id;
}

DisplayHandle getCurrentDisplayHandle()
{
    auto id = DisplayHandle{0};

#if defined(OD_WIN)
    const auto display = wglGetCurrentDC();
#elif defined(OS_LNX)
    const auto display = glXGetCurrentDisplay();
#endif
    id = reinterpret_cast<DisplayHandle>(display);

    return id;
}

}
