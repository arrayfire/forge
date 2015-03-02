/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <type_util.hpp>

const char *getName(GLenum type)
{
    switch(type) {
    case GL_FLOAT:              return "float";
    case GL_INT:                return "int";
    case GL_UNSIGNED_INT:       return "unsigned int";
    case GL_BYTE:               return "char";
    case GL_UNSIGNED_BYTE:      return "unsigned char";
    default:                    return "unknown type";
    }
}
