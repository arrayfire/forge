/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#include <err_common.hpp>

namespace backend
{

// Required to be defined for GLEW MX to work,
// along with the GLEW_MX define in the perprocessor!
GLEWContext* glewGetContext();

void MakeContextCurrent();

void MakeContextCurrent(fg_window_handle wh);

GLenum mode_to_glColor(fg_color_mode mode);

char* loadFile(const char *fname, GLint &fSize);

GLuint initShaders(const char* vshader_code, const char* fshader_code);

template<typename T>
GLuint createBuffer(int size, const T* data, GLenum usage)
{
    GLuint ret_val = 0;
    glGenBuffers(1,&ret_val);
    glBindBuffer(GL_ARRAY_BUFFER,ret_val);
    glBufferData(GL_ARRAY_BUFFER,size*sizeof(T),data,usage);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    return ret_val;
}

}
