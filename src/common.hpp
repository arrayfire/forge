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

#ifdef OS_WIN
    #define GLFW_EXPOSE_NATIVE_WIN32
    #define GLFW_EXPOSE_NATIVE_WGL
#endif

#ifdef OS_LNX
    #define GLFW_EXPOSE_NATIVE_X11
    #define GLFW_EXPOSE_NATIVE_GLX
#endif

#include <GLFW/glfw3.h>
#ifndef OS_MAC
#include <GLFW/glfw3native.h>
#endif
#include <fg/exception.h>
#include <err_common.hpp>

static const float GRAY[]  = {0.0f   , 0.0f   , 0.0f   , 1.0f};
static const float WHITE[] = {1.0f   , 1.0f   , 1.0f   , 1.0f};
static const float BLUE[]  = {0.0588f, 0.1137f, 0.2745f, 1.0f};
static const float RED[]   = {1.0f   , 0.0f   , 0.0f   , 1.0f};

namespace internal
{

/* Basic renderable class
 *
 * Any object that is renderable to a window should inherit from this
 * class.
 */
class AbstractRenderable {
    public:
        /* render is a pure virtual function.
         * @pX X coordinate at which the currently bound viewport begins.
         * @pX Y coordinate at which the currently bound viewport begins.
         * @pViewPortWidth Width of the currently bound viewport.
         * @pViewPortHeight Height of the currently bound viewport.
         *
         * Any concrete class that inherits AbstractRenderable class needs to
         * implement this method to render their OpenGL objects to
         * the currently bound viewport.
         *
         * @return nothing.
         */
        virtual void render(const void* pWnd,
                int pX, int pY, int pViewPortWidth, int pViewPortHeight) = 0;

        /* virtual function to set colormap, a derviced class might
         * use it or ignore it if it doesnt have a need for color maps */
        virtual void setColorMapUBOParams(GLuint ubo, GLuint size) {
        }
};

}

GLenum FGTypeToGLenum(fg::FGType val);

fg::FGType GLenumToFGType(GLenum val);

GLenum FGModeToGLColor(fg::ColorMode mode);

fg::ColorMode GLModeToFGColor(GLenum mode);

char* loadFile(const char *fname, GLint &fSize);

GLuint initShaders(const char* vshader_code, const char* fshader_code);

template<typename T>
GLuint createBuffer(GLenum target, size_t size, const T* data, GLenum usage)
{
    GLuint ret_val = 0;
    glGenBuffers(1, &ret_val);
    glBindBuffer(target, ret_val);
    glBufferData(target, size*sizeof(T), data, usage);
    glBindBuffer(target, 0);
    return ret_val;
}

int next_p2(int value);

float clampTo01(float a);
