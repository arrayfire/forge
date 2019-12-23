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
#include <window_impl.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <sstream>
#include <string>

using namespace std;

namespace forge {
namespace opengl {

GLenum dtype2gl(const dtype pValue) {
    switch (pValue) {
        case s8: return GL_BYTE;
        case u8: return GL_UNSIGNED_BYTE;
        case s32: return GL_INT;
        case u32: return GL_UNSIGNED_INT;
        case s16: return GL_SHORT;
        case u16: return GL_UNSIGNED_SHORT;
        default: return GL_FLOAT;
    }
}

GLenum ctype2gl(const ChannelFormat pMode) {
    switch (pMode) {
        case FG_GRAYSCALE: return GL_RED;
        case FG_RG: return GL_RG;
        case FG_RGB: return GL_RGB;
        case FG_BGR: return GL_BGR;
        case FG_BGRA: return GL_BGRA;
        default: return GL_RGBA;
    }
}

GLenum ictype2gl(const ChannelFormat pMode) {
    if (pMode == FG_GRAYSCALE)
        return GL_RED;
    else if (pMode == FG_RG)
        return GL_RG;
    else if (pMode == FG_RGB || pMode == FG_BGR)
        return GL_RGB;

    return GL_RGBA;
}

void glErrorCheck(const char* pMsg, const char* pFile, int pLine) {
// Skipped in release mode
#ifndef NDEBUG
    auto errorCheck = [](const char* pMsg, const char* pFile, int pLine) {
        GLenum x = glGetError();
        if (x != GL_NO_ERROR) {
            std::stringstream ss;
            ss << "GL Error at: " << pFile << ":" << pLine
               << " Message: " << pMsg << " Error Code: " << glGetString(x)
               << std::endl;
            FG_ERROR(ss.str().c_str(), FG_ERR_GL_ERROR);
        }
    };
    errorCheck(pMsg, pFile, pLine);
#endif
}

GLuint screenQuadVBO(const int pWindowId) {
    // FIXME: VBOs can be shared, but for simplicity
    // right now just created one VBO each window,
    // ignoring shared contexts
    static std::map<int, GLuint> svboMap;

    if (svboMap.find(pWindowId) == svboMap.end()) {
        static const float vertices[8] = {-1.0f, -1.0f, 1.0f,  -1.0f,
                                          1.0f,  1.0f,  -1.0f, 1.0f};
        svboMap[pWindowId] =
            createBuffer(GL_ARRAY_BUFFER, 8, vertices, GL_STATIC_DRAW);
    }

    return svboMap[pWindowId];
}

GLuint screenQuadVAO(const int pWindowId) {
    static std::map<int, GLuint> svaoMap;

    if (svaoMap.find(pWindowId) == svaoMap.end()) {
        static const float texcords[8]   = {0.0, 1.0, 1.0, 1.0,
                                          1.0, 0.0, 0.0, 0.0};
        static const uint32_t indices[6] = {0, 1, 2, 0, 2, 3};

        GLuint tbo = createBuffer(GL_ARRAY_BUFFER, 8, texcords, GL_STATIC_DRAW);
        GLuint ibo =
            createBuffer(GL_ELEMENT_ARRAY_BUFFER, 6, indices, GL_STATIC_DRAW);
        GLuint vao = 0;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach vbo
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, screenQuadVBO(pWindowId));
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        // attach tbo
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, tbo);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
        // attach ibo
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        svaoMap[pWindowId] = vao;
    }

    return svaoMap[pWindowId];
}

}  // namespace opengl
}  // namespace forge
