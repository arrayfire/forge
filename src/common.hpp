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
#include <fg/exception.h>
#include <err_common.hpp>

static const float GRAY[] = {0.0, 0.0, 0.0, 1.0};
static const float WHITE[] = {1.0, 1.0, 1.0, 1.0};
static const float BLUE[4] = { 0.0588f, 0.1137f, 0.2745f, 1 };

GLenum FGMode_to_GLColor(fg::ColorMode mode);

fg::ColorMode GLMode_to_FGColor(GLenum mode);

char* loadFile(const char *fname, GLint &fSize);

GLuint initShaders(const char* vshader_code, const char* fshader_code);

template<typename T>
GLuint createBuffer(size_t size, const T* data, GLenum usage)
{
    GLuint ret_val = 0;
    glGenBuffers(1,&ret_val);
    glBindBuffer(GL_ARRAY_BUFFER,ret_val);
    glBufferData(GL_ARRAY_BUFFER,size*sizeof(T),data,usage);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    return ret_val;
}

int next_p2(int value);

float clampTo01(float a);
