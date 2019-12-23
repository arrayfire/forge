/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_handling.hpp>
#include <gl_helpers.hpp>
#include <shader_program.hpp>

#include <iostream>

using namespace std;

typedef struct {
    uint32_t vertex;
    uint32_t fragment;
    uint32_t geometry;
} Shaders;

#define FG_COMPILE_LINK_ERROR(pArg, PTYPE)                                     \
    do {                                                                       \
        int infoLogLen   = 0;                                                  \
        int charsWritten = 0;                                                  \
        GLchar* infoLog;                                                       \
                                                                               \
        glGet##PTYPE##iv(pArg, GL_INFO_LOG_LENGTH, &infoLogLen);               \
                                                                               \
        if (infoLogLen > 1) {                                                  \
            infoLog = new GLchar[infoLogLen];                                  \
            glGet##PTYPE##InfoLog(pArg, infoLogLen, &charsWritten, infoLog);   \
            std::cerr << "InfoLog:" << std::endl << infoLog << std::endl;      \
            delete[] infoLog;                                                  \
            FG_ERROR("OpenGL " #PTYPE " Compilation Failed", FG_ERR_GL_ERROR); \
        }                                                                      \
    } while (0)

void attachAndLinkProgram(uint32_t pProgram, Shaders pShaders) {
    glAttachShader(pProgram, pShaders.vertex);
    glAttachShader(pProgram, pShaders.fragment);
    if (pShaders.geometry > 0) { glAttachShader(pProgram, pShaders.geometry); }

    glLinkProgram(pProgram);
    int32_t linked;
    glGetProgramiv(pProgram, GL_LINK_STATUS, &linked);
    if (!linked) {
        std::cerr << "Program did not link." << std::endl;
        FG_COMPILE_LINK_ERROR(pProgram, Program);
    }
}

Shaders loadShaders(const char* pVertexShaderSrc,
                    const char* pFragmentShaderSrc,
                    const char* pGeometryShaderSrc) {
    uint32_t f, v;

    v = glCreateShader(GL_VERTEX_SHADER);
    if (!v) {
        std::cerr << "Vertex shader creation failed." << std::endl;
        FG_COMPILE_LINK_ERROR(v, Shader);
    }
    f = glCreateShader(GL_FRAGMENT_SHADER);
    if (!f) {
        std::cerr << "Fragment shader creation failed." << std::endl;
        FG_COMPILE_LINK_ERROR(f, Shader);
    }

    // load shaders & get length of each
    glShaderSource(v, 1, &pVertexShaderSrc, NULL);
    glShaderSource(f, 1, &pFragmentShaderSrc, NULL);

    int32_t compiled;

    glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        std::cerr << "Vertex shader not compiled." << std::endl;
        FG_COMPILE_LINK_ERROR(v, Shader);
    }

    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        std::cerr << "Fragment shader not compiled." << std::endl;
        FG_COMPILE_LINK_ERROR(f, Shader);
    }

    uint32_t g = 0;
    /* compile geometry shader if source provided */
    if (pGeometryShaderSrc) {
        g = glCreateShader(GL_GEOMETRY_SHADER);
        if (!g) {
            std::cerr << "Geometry shader not compiled." << std::endl;
            FG_COMPILE_LINK_ERROR(g, Shader);
        }
        glShaderSource(g, 1, &pGeometryShaderSrc, NULL);
        glCompileShader(g);
        glGetShaderiv(g, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            std::cerr << "Geometry shader not compiled." << std::endl;
            FG_COMPILE_LINK_ERROR(g, Shader);
        }
    }

    Shaders out;
    out.vertex   = v;
    out.fragment = f;
    out.geometry = g;

    return out;
}

namespace forge {
namespace opengl {

ShaderProgram::ShaderProgram(const char* pVertShaderSrc,
                             const char* pFragShaderSrc,
                             const char* pGeomShaderSrc)
    : mVertex(0), mFragment(0), mGeometry(0), mProgram(0) {
    Shaders shrds = loadShaders(pVertShaderSrc, pFragShaderSrc, pGeomShaderSrc);
    mProgram      = glCreateProgram();
    attachAndLinkProgram(mProgram, shrds);
    mVertex   = shrds.vertex;
    mFragment = shrds.fragment;
    mGeometry = shrds.geometry;
}

ShaderProgram::~ShaderProgram() {
    if (mVertex) glDeleteShader(mVertex);
    if (mFragment) glDeleteShader(mFragment);
    if (mGeometry) glDeleteShader(mGeometry);
    if (mProgram) glDeleteProgram(mProgram);
}

uint32_t ShaderProgram::getProgramId() const { return mProgram; }

uint32_t ShaderProgram::getUniformLocation(const char* pAttributeName) {
    return glGetUniformLocation(mProgram, pAttributeName);
}

uint32_t ShaderProgram::getUniformBlockIndex(const char* pAttributeName) {
    return glGetUniformBlockIndex(mProgram, pAttributeName);
}

uint32_t ShaderProgram::getAttributeLocation(const char* pAttributeName) {
    return glGetAttribLocation(mProgram, pAttributeName);
}

void ShaderProgram::bind() { glUseProgram(mProgram); }

void ShaderProgram::unbind() { glUseProgram(0); }

}  // namespace opengl
}  // namespace forge
