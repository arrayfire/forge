/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#include <fg/image.h>
#include <fg/exception.h>
#include <common.hpp>
#include <err_common.hpp>

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
"in vec2 texcoord;\n"
"out vec4 fragColor;\n"
"void main()\n"
"{\n"
"    fragColor = texture2D(tex,texcoord);\n"
"}\n";

static GLuint gCanvasVAO = 0;

namespace fg
{

Image::Image(unsigned pWidth, unsigned pHeight, ColorMode pFormat, GLenum pDataType)
: mWidth(pWidth), mHeight(pHeight), mFormat(pFormat), mDataType(pDataType)
{
    MakeContextCurrent();
    mGLformat = FGMode_to_GLColor(mFormat);

    // Initialize OpenGL Items
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &(mTex));
    glBindTexture(GL_TEXTURE_2D, mTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, mGLformat, mWidth, mHeight, 0, mGLformat, mDataType, NULL);

    CheckGL("Before PBO Initialization");
    glGenBuffers(1, &mPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mPBO);
    size_t typeSize = 0;
    switch(mDataType) {
        case GL_FLOAT:          typeSize = sizeof(float);     break;
        case GL_INT:            typeSize = sizeof(int  );     break;
        case GL_UNSIGNED_INT:   typeSize = sizeof(uint );     break;
        case GL_BYTE:           typeSize = sizeof(char );     break;
        case GL_UNSIGNED_BYTE:  typeSize = sizeof(uchar);     break;
    }
    mPBOsize = mWidth * mHeight * mFormat * typeSize;
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, mPBOsize, NULL, GL_STREAM_COPY);

    glBindTexture(GL_TEXTURE_2D, 0);
    CheckGL("After PBO Initialization");

    mProgram = initShaders(vertex_shader_code, fragment_shader_code);

    if (gCanvasVAO==0) {
        const float vertices[12] = {
            -1.0f,-1.0f,0.0,
            1.0f,-1.0f,0.0,
            1.0f, 1.0f,0.0,
            -1.0f, 1.0f,0.0};

        const float texcords[8] = {0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0};

        const uint indices[6] = {0,1,2,0,2,3};

        GLuint vbo  = createBuffer(12, vertices, GL_STATIC_DRAW);
        GLuint tbo  = createBuffer(8, texcords, GL_STATIC_DRAW);
        GLuint ibo;
        glGenBuffers(1,&ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint)*6, indices, GL_STATIC_DRAW);
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
}

Image::~Image()
{
    MakeContextCurrent();
    glDeleteBuffers(1, &mPBO);
    glDeleteTextures(1, &mTex);
    glDeleteProgram(mProgram);
}

unsigned Image::width() const { return mWidth; }

unsigned Image::height() const { return mHeight; }

ColorMode Image::pixelFormat() const { return mFormat; }

GLenum Image::channelType() const { return mDataType; }

GLuint Image::pbo() const { return mPBO; }

size_t Image::size() const { return mPBOsize; }

void Image::render() const
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

    // load texture from PBO
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(tex_loc, 0);
    glBindTexture(GL_TEXTURE_2D, mTex);
    // bind PBO to load data into texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mPBO);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, mGLformat, mDataType, 0);

    glUniformMatrix4fv(mat_loc, 1, GL_FALSE, matrix);

    // Draw to screen
    glBindVertexArray(gCanvasVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // Unbind textures
    glBindTexture(GL_TEXTURE_2D, 0);

    // ubind the shader program
    glUseProgram(0);
}


void drawImage(Window* pWindow, const Image& pImage)
{
    CheckGL("Begin drawImage");
    MakeContextCurrent(pWindow);

    int wind_width, wind_height;
    glfwGetWindowSize(pWindow->window(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);

    // clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.2, 0.2, 0.2, 1.0);

    pImage.render();

    glfwSwapBuffers(pWindow->window());
    glfwPollEvents();
    ForceCheckGL("End drawImage");
}

void drawImages(Window* pWindow, int pRows, int pCols, const unsigned int pNumImages, const std::vector<Image>& pHandles)
{
    CheckGL("Begin drawImages");
    MakeContextCurrent(pWindow);

    int wind_width, wind_height;
    glfwGetWindowSize(pWindow->window(), &wind_width, &wind_height);
    glViewport(0, 0, wind_width, wind_height);

    // calculate cell width and height
    uint wid_step = wind_width/ pCols;
    uint hei_step = wind_height/ pRows;

    // clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.2, 0.2, 0.2, 1.0);

    for (int c=0; c<pCols; ++c) {
        for (int r=0; r<pRows; ++r) {
            uint idx = c*pRows + pRows-1-r;
            if (idx<pNumImages) {

                int x_off = c * wid_step;
                int y_off = r * hei_step;

                // set viewport to render sub image
                glViewport(x_off, y_off, wid_step, hei_step);

                pHandles[idx].render();
            }
        }
    }

    glfwSwapBuffers(pWindow->window());
    glfwPollEvents();
    CheckGL("End drawImages");
}

}
