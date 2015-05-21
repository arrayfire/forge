/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/image.h>
#include <image.hpp>
#include <common.hpp>

static const char* vertex_shader_code =
"#version 330\n"
"layout(location = 0) in vec3 pos;\n"
"layout(location = 1) in vec2 tex;\n"
"uniform mat4 matrix;\n"
"out vec2 texcoord;\n"
"void main() {\n"
"    texcoord = tex;\n"
"    gl_Position = matrix * vec4(pos,1.0);\n"
"}\n";

static const char* fragment_shader_code =
"#version 330\n"
"uniform sampler2D tex;\n"
"uniform bool isGrayScale;\n"
"in vec2 texcoord;\n"
"out vec4 fragColor;\n"
"void main()\n"
"{\n"
"    vec4 tcolor = texture(tex, texcoord);\n"
"    if(isGrayScale)\n"
"        fragColor = vec4(tcolor.r, tcolor.r, tcolor.r, 1);\n"
"    else\n"
"        fragColor = tcolor;\n"
"}\n";

static GLuint gCanvasVAO = 0;

namespace internal
{

_Image::_Image(unsigned pWidth, unsigned pHeight, fg::ColorMode pFormat, GLenum pDataType)
: mWidth(pWidth), mHeight(pHeight), mFormat(pFormat), mDataType(pDataType)
{
    CheckGL("Begin Image::Image");
    mGLformat = FGMode_to_GLColor(mFormat);

    // Initialize OpenGL Items
    glGenTextures(1, &(mTex));
    glBindTexture(GL_TEXTURE_2D, mTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, mGLformat, mWidth, mHeight, 0, mGLformat, mDataType, NULL);

    CheckGL("Before PBO Initialization");
    glGenBuffers(1, &mPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPBO);
    size_t typeSize = 0;
    switch(mDataType) {
        case GL_FLOAT:          typeSize = sizeof(float);           break;
        case GL_INT:            typeSize = sizeof(int  );           break;
        case GL_UNSIGNED_INT:   typeSize = sizeof(unsigned int);    break;
        case GL_BYTE:           typeSize = sizeof(char );           break;
        case GL_UNSIGNED_BYTE:  typeSize = sizeof(unsigned char);   break;
    }
    mPBOsize = mWidth * mHeight * mFormat * typeSize;
    glBufferData(GL_PIXEL_UNPACK_BUFFER, mPBOsize, NULL, GL_STREAM_COPY);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CheckGL("After PBO Initialization");

    mProgram = initShaders(vertex_shader_code, fragment_shader_code);

    if (gCanvasVAO==0) {
        const float vertices[12] = {
            -1.0f,-1.0f,0.0,
            1.0f,-1.0f,0.0,
            1.0f, 1.0f,0.0,
            -1.0f, 1.0f,0.0};

        const float texcords[8] = {0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0};

        const unsigned indices[6] = {0,1,2,0,2,3};

        GLuint vbo  = createBuffer(12, vertices, GL_STATIC_DRAW);
        GLuint tbo  = createBuffer(8, texcords, GL_STATIC_DRAW);
        GLuint ibo;
        glGenBuffers(1,&ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned) * 6, indices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
        // bind vao
        glGenVertexArrays(1, &gCanvasVAO);
        glBindVertexArray(gCanvasVAO);
        // attach vbo
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(0);
        // attach tbo
        glBindBuffer(GL_ARRAY_BUFFER,tbo);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        glEnableVertexAttribArray(1);
        // attach ibo
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBindVertexArray(0);
    }
    CheckGL("End Image::Image");
}

_Image::~_Image()
{
    glDeleteBuffers(1, &mPBO);
    glDeleteTextures(1, &mTex);
    glDeleteProgram(mProgram);
}

unsigned _Image::width() const { return mWidth; }

unsigned _Image::height() const { return mHeight; }

fg::ColorMode _Image::pixelFormat() const { return mFormat; }

GLenum _Image::channelType() const { return mDataType; }

GLuint _Image::pbo() const { return mPBO; }

size_t _Image::size() const { return mPBOsize; }

void _Image::render() const
{
    static const float matrix[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f};

    glUseProgram(mProgram);
    // get uniform locations
    int mat_loc = glGetUniformLocation(mProgram,"matrix");
    int tex_loc = glGetUniformLocation(mProgram,"tex");
    int chn_loc = glGetUniformLocation(mProgram,"isGrayScale");

    glUniform1i(chn_loc, mFormat==1);
    // load texture from PBO
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(tex_loc, 0);
    glBindTexture(GL_TEXTURE_2D, mTex);
    // bind PBO to load data into texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPBO);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, mGLformat, mDataType, 0);

    glUniformMatrix4fv(mat_loc, 1, GL_FALSE, matrix);

    // Draw to screen
    glBindVertexArray(gCanvasVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // Unbind textures
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // ubind the shader program
    glUseProgram(0);
}

}

namespace fg
{

Image::Image(unsigned pWidth, unsigned pHeight, fg::ColorMode pFormat, GLenum pDataType) {
    value = std::make_shared<internal::_Image>(pWidth, pHeight, pFormat, pDataType);
}

unsigned Image::width() const {
    return value.get()->width();
}

unsigned Image::height() const {
    return value.get()->height();
}

ColorMode Image::pixelFormat() const {
    return value.get()->pixelFormat();
}

GLenum Image::channelType() const {
    return value.get()->channelType();
}

GLuint Image::pbo() const {
    return value.get()->pbo();
}

size_t Image::size() const {
    return value.get()->size();
}

internal::_Image* Image::get() const {
    return value.get();
}

void Image::render() const {
    value.get()->render();
}

}