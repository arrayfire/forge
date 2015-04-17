/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/exception.h>
#include <common.hpp>

#include <iostream>
#include <fstream>

using namespace fg;

static fg::Window* current = NULL;

typedef struct {
    GLuint vertex;
    GLuint fragment;
} shaders_t;

GLEWContext* glewGetContext()
{
    return current->glewContext();
}

void MakeContextCurrent(fg::Window* pWindow)
{
    CheckGL("Begin MakeContextCurrent");
    if (pWindow != NULL)
    {
        glfwMakeContextCurrent((GLFWwindow*)pWindow->window());
        current = pWindow;
    }
    CheckGL("End MakeContextCurrent");
}

void MakeContextCurrent()
{
    MakeContextCurrent(current);
}

GLenum FGMode_to_GLColor(ColorMode mode)
{
    GLenum color = GL_RGBA;
    switch(mode) {
        case FG_RED : color = GL_RED;  break;
        case FG_RGB : color = GL_RGB;  break;
        case FG_RGBA: color = GL_RGBA; break;
    }
    return color;
}

ColorMode GLMode_to_FGColor(GLenum mode)
{
    ColorMode color = FG_RGBA;
    switch(mode) {
        case GL_RED : color = FG_RED;  break;
        case GL_RGB : color = FG_RGB;  break;
        case GL_RGBA: color = FG_RGBA; break;
    }
    return color;
}

char* loadFile(const char * fname, GLint &fSize)
{
    std::ifstream file(fname,std::ios::in|std::ios::binary|std::ios::ate);
    if (file.is_open())
    {
        unsigned int size = (unsigned int)file.tellg();
        fSize = size;
        char *memblock = new char [size];
        file.seekg (0, std::ios::beg);
        file.read (memblock, size);
        file.close();
        std::cerr << "file " << fname << " loaded" << std::endl;
        return memblock;
    }

    char buffer[64];
    sprintf(buffer, "Unable to open file %s", fname);

    throw fg::Error("loadFile", __LINE__, buffer, FG_ERR_GL_ERROR);
}

void printShaderInfoLog(GLint shader)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 1)
    {
        infoLog = new GLchar[infoLogLen];
        glGetShaderInfoLog(shader,infoLogLen, &charsWritten, infoLog);
        std::cerr << "InfoLog:" << std::endl << infoLog << std::endl;
        delete [] infoLog;
        throw fg::Error("printShaderInfoLog", __LINE__, "OpenGL Shader compilation failed", FG_ERR_GL_ERROR);
    }
}

void printLinkInfoLog(GLint prog)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 1)
    {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetProgramInfoLog(prog,infoLogLen, &charsWritten, infoLog);
        std::cerr << "InfoLog:" << std::endl << infoLog << std::endl;
        delete [] infoLog;
        throw fg::Error("printLinkInfoLog", __LINE__, "OpenGL Shader linking failed", FG_ERR_GL_ERROR);
    }
}

void attachAndLinkProgram(GLuint program, shaders_t shaders)
{
    glAttachShader(program, shaders.vertex);
    glAttachShader(program, shaders.fragment);

    glLinkProgram(program);
    GLint linked;
    glGetProgramiv(program,GL_LINK_STATUS, &linked);
    if (!linked)
    {
        std::cerr << "Program did not link." << std::endl;
        throw fg::Error("attachAndLinkProgram", __LINE__, "OpenGL program linking failed", FG_ERR_GL_ERROR);
    }
    printLinkInfoLog(program);
}

shaders_t loadShaders(const char * vert_code, const char * frag_code)
{
    GLuint f, v;

    v = glCreateShader(GL_VERTEX_SHADER);
    f = glCreateShader(GL_FRAGMENT_SHADER);

    // load shaders & get length of each
    glShaderSource(v, 1, &vert_code, NULL);
    glShaderSource(f, 1, &frag_code, NULL);

    GLint compiled;

    glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        std::cerr << "Vertex shader not compiled." << std::endl;
        printShaderInfoLog(v);
    }

    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        std::cerr << "Fragment shader not compiled." << std::endl;
        printShaderInfoLog(f);
    }

    shaders_t out; out.vertex = v; out.fragment = f;

    return out;
}

GLuint initShaders(const char* vshader_code, const char* fshader_code)
{
    shaders_t shaders = loadShaders(vshader_code, fshader_code);
    GLuint shader_program = glCreateProgram();
    attachAndLinkProgram(shader_program, shaders);
    return shader_program;
}
