/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <fg/defines.h>

typedef struct
{
    ContextHandle   cxt;
    DisplayHandle   dsp;

    GLFWwindow*     pWindow;
    GLEWContext*    pGLEWContext;
    int             uiWidth;
    int             uiHeight;
    int             uiID;
    FGenum          type;
    fg_color_mode   mode;
} fg_window_struct;

typedef fg_window_struct* fg_window_handle;

#ifdef __cplusplus
namespace fg
{

class FGAPI Window {
    private:
        fg_window_handle mHandle;

    public:
        Window();
        Window(const uint pWidth, const uint pHeight, const char* pTitle,
               WindowColorMode pMode, FGenum pChannelType);
        ~Window();

        uint width() const;
        uint height() const;
        WindowColorMode colorMode() const;
        FGenum channelType() const;
        fg_window_handle get() const;
};

void makeWindowCurrent(const Window& pWindow);

}
#endif

#ifdef __cplusplus
extern "C" {
#endif
    FGAPI fg_err fg_create_window(fg_window_handle *out, const unsigned width, const unsigned height,
                                  const char *title, fg_color_mode mode, FGenum type);

    FGAPI fg_err fg_make_window_current(const fg_window_handle in);

    FGAPI fg_err fg_destroy_window(const fg_window_handle in);
#ifdef __cplusplus
}
#endif
