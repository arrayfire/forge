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
#include <mutex>
#include <map>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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
"const int size = 259;\n"
"uniform float cmaplen;\n"
"layout(std140) uniform ColorMap\n"
"{\n"
"    vec4 ch[size];\n"
"};\n"
"uniform sampler2D tex;\n"
"uniform bool isGrayScale;\n"
"in vec2 texcoord;\n"
"out vec4 fragColor;\n"
"void main()\n"
"{\n"
"    vec4 tcolor = texture(tex, texcoord);\n"
"    vec4 clrs = vec4(1, 0, 0, 1);\n"
"    if(isGrayScale)\n"
"        clrs = vec4(tcolor.r, tcolor.r, tcolor.r, 1);\n"
"    else\n"
"        clrs = tcolor;\n"
"    vec4 fidx  = (cmaplen-1) * clrs;\n"
"    ivec4 idx  = ivec4(fidx.x, fidx.y, fidx.z, fidx.w);\n"
"    float r_ch = ch[idx.x].r;\n"
"    float g_ch = ch[idx.y].g;\n"
"    float b_ch = ch[idx.z].b;\n"
"    fragColor = vec4(r_ch, g_ch , b_ch, 1);\n"
"}\n";

GLuint imageQuadVAO(int pWindowId)
{
    static std::map<int, GLuint> mVAOMap;

    if (mVAOMap.find(pWindowId)==mVAOMap.end()) {
        static const float vertices[12] = {-1.0f,-1.0f,0.0,
                                    1.0f,-1.0f,0.0,
                                    1.0f, 1.0f,0.0,
                                    -1.0f, 1.0f,0.0};
        static const float texcords[8]  = {0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0};
        static const unsigned indices[6]= {0,1,2,0,2,3};
        GLuint vbo  = createBuffer(GL_ARRAY_BUFFER, 12, vertices, GL_STATIC_DRAW);
        GLuint tbo  = createBuffer(GL_ARRAY_BUFFER, 8, texcords, GL_STATIC_DRAW);
        GLuint ibo  = createBuffer(GL_ELEMENT_ARRAY_BUFFER, 6, indices, GL_STATIC_DRAW);

        GLuint vao = 0;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        // attach vbo
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
        // attach tbo
        glBindBuffer(GL_ARRAY_BUFFER, tbo);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        // attach ibo
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    return mVAOMap[pWindowId];
}

namespace internal
{

void image_impl::bindResources(int pWindowId)
{
    glBindVertexArray(imageQuadVAO(pWindowId));
}

void image_impl::unbindResources() const
{
    glBindVertexArray(0);
}

image_impl::image_impl(unsigned pWidth, unsigned pHeight,
                       fg::ChannelFormat pFormat, fg::dtype pDataType)
    : mWidth(pWidth), mHeight(pHeight),
      mFormat(pFormat), mGLformat(gl_ctype(mFormat)), mGLiformat(gl_ictype(mFormat)),
      mDataType(pDataType), mGLType(gl_dtype(mDataType))
{
    CheckGL("Begin Image::Image");

    // Initialize OpenGL Items
    glGenTextures(1, &(mTex));
    glBindTexture(GL_TEXTURE_2D, mTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, mGLiformat, mWidth, mHeight, 0, mGLformat, mGLType, NULL);

    CheckGL("Before PBO Initialization");
    glGenBuffers(1, &mPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPBO);
    size_t typeSize = 0;
    switch(mGLType) {
        case GL_INT:            typeSize = sizeof(int  );           break;
        case GL_UNSIGNED_INT:   typeSize = sizeof(unsigned int);    break;
        case GL_BYTE:           typeSize = sizeof(char );           break;
        case GL_UNSIGNED_BYTE:  typeSize = sizeof(unsigned char);   break;
        default: typeSize = sizeof(float); break;
    }
    mPBOsize = mWidth * mHeight * mFormat * typeSize;
    glBufferData(GL_PIXEL_UNPACK_BUFFER, mPBOsize, NULL, GL_STREAM_COPY);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    CheckGL("After PBO Initialization");

    mProgram = initShaders(vertex_shader_code, fragment_shader_code);

    CheckGL("End Image::Image");
}

image_impl::~image_impl()
{
    glDeleteBuffers(1, &mPBO);
    glDeleteTextures(1, &mTex);
    glDeleteProgram(mProgram);
}

void image_impl::setColorMapUBOParams(GLuint ubo, GLuint size)
{
    mColorMapUBO = ubo;
    mUBOSize = size;
}

void image_impl::keepAspectRatio(const bool keep)
{
    mKeepARatio = keep;
}

unsigned image_impl::width() const { return mWidth; }

unsigned image_impl::height() const { return mHeight; }

fg::ChannelFormat image_impl::pixelFormat() const { return mFormat; }

fg::dtype image_impl::channelType() const { return mDataType; }

unsigned image_impl::pbo() const { return mPBO; }

unsigned image_impl::size() const { return (unsigned)mPBOsize; }

void image_impl::render(int pWindowId, int pX, int pY, int pViewPortWidth, int pViewPortHeight)
{
    float xscale = 1.f;
    float yscale = 1.f;
    if (mKeepARatio) {
        if (mWidth > mHeight) {
            float trgtH = pViewPortWidth * float(mHeight)/float(mWidth);
            float trgtW = trgtH * float(mWidth)/float(mHeight);
            xscale = trgtW/pViewPortWidth;
            yscale = trgtH/pViewPortHeight;
        } else {
            float trgtW = pViewPortHeight * float(mWidth)/float(mHeight);
            float trgtH = trgtW * float(mHeight)/float(mWidth);
            xscale = trgtW/pViewPortWidth;
            yscale = trgtH/pViewPortHeight;
        }
    }
    glm::mat4 strans = glm::scale(glm::mat4(1.0f), glm::vec3(xscale, yscale, 1));

    glUseProgram(mProgram);
    // get uniform locations
    int mat_loc = glGetUniformLocation(mProgram, "matrix");
    int tex_loc = glGetUniformLocation(mProgram, "tex");
    int chn_loc = glGetUniformLocation(mProgram, "isGrayScale");
    int cml_loc = glGetUniformLocation(mProgram, "cmaplen");
    int ubo_idx = glGetUniformBlockIndex(mProgram, "ColorMap");

    glUniform1i(chn_loc, mFormat==1);
    // load texture from PBO
    glActiveTexture(GL_TEXTURE0);
    glUniform1i(tex_loc, 0);
    glBindTexture(GL_TEXTURE_2D, mTex);
    // bind PBO to load data into texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mPBO);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, mGLformat, mGLType, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glUniformMatrix4fv(mat_loc, 1, GL_FALSE, glm::value_ptr(strans));

    glUniform1f(cml_loc, (GLfloat)mUBOSize);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, mColorMapUBO);
    glUniformBlockBinding(mProgram, ubo_idx, 0);

    CheckGL("Before render");

    // Draw to screen
    bindResources(pWindowId);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    unbindResources();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // ubind the shader program
    glUseProgram(0);
    CheckGL("After render");
}

}

namespace fg
{

Image::Image(unsigned pWidth, unsigned pHeight, fg::ChannelFormat pFormat, fg::dtype pDataType) {
    value = new internal::_Image(pWidth, pHeight, pFormat, pDataType);
}

Image::Image(const Image& other) {
    value = new internal::_Image(*other.get());
}

Image::~Image() {
    delete value;
}

unsigned Image::width() const {
    return value->width();
}

unsigned Image::height() const {
    return value->height();
}

ChannelFormat Image::pixelFormat() const {
    return value->pixelFormat();
}

fg::dtype Image::channelType() const {
    return value->channelType();
}

GLuint Image::pbo() const {
    return value->pbo();
}

unsigned Image::size() const {
    return (unsigned)value->size();
}

internal::_Image* Image::get() const {
    return value;
}

}
