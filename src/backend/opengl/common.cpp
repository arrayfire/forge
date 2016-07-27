/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common.hpp>
#include <window_impl.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace gl;
using namespace fg;
using namespace std;

#define PI 3.14159

typedef struct {
    GLuint vertex;
    GLuint fragment;
    GLuint geometry;
} Shaders;

GLenum dtype2gl(const fg::dtype pValue)
{
    switch(pValue) {
        case s8:  return GL_BYTE;
        case u8:  return GL_UNSIGNED_BYTE;
        case s32: return GL_INT;
        case u32: return GL_UNSIGNED_INT;
        case s16: return GL_SHORT;
        case u16: return GL_UNSIGNED_SHORT;
        default:  return GL_FLOAT;
    }
}

GLenum ctype2gl(const ChannelFormat pMode)
{
    switch(pMode) {
        case FG_GRAYSCALE: return GL_RED;
        case FG_RG  : return GL_RG;
        case FG_RGB : return GL_RGB;
        case FG_BGR : return GL_BGR;
        case FG_BGRA: return GL_BGRA;
        default     : return GL_RGBA;
    }
}

GLenum ictype2gl(const ChannelFormat pMode)
{
    if (pMode==FG_GRAYSCALE)
        return GL_RED;
    else if (pMode==FG_RG)
        return GL_RG;
    else if (pMode==FG_RGB || pMode==FG_BGR)
        return GL_RGB;

    return GL_RGBA;
}

void printShaderInfoLog(GLint pShader)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetShaderiv(pShader, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 1) {
        infoLog = new GLchar[infoLogLen];
        glGetShaderInfoLog(pShader, infoLogLen, &charsWritten, infoLog);
        std::cerr << "InfoLog:" << std::endl << infoLog << std::endl;
        delete [] infoLog;
        throw fg::Error("printShaderInfoLog", __LINE__,
                "OpenGL Shader compilation failed", FG_ERR_GL_ERROR);
    }
}

void printLinkInfoLog(GLint pProgram)
{
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar *infoLog;

    glGetProgramiv(pProgram, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (infoLogLen > 1) {
        infoLog = new GLchar[infoLogLen];
        // error check for fail to allocate memory omitted
        glGetProgramInfoLog(pProgram, infoLogLen, &charsWritten, infoLog);
        std::cerr << "InfoLog:" << std::endl << infoLog << std::endl;
        delete [] infoLog;
        throw fg::Error("printLinkInfoLog", __LINE__,
                "OpenGL Shader linking failed", FG_ERR_GL_ERROR);
    }
}

void attachAndLinkProgram(GLuint pProgram, Shaders pShaders)
{
    glAttachShader(pProgram, pShaders.vertex);
    glAttachShader(pProgram, pShaders.fragment);
    if (pShaders.geometry>0) {
        glAttachShader(pProgram, pShaders.geometry);
    }

    glLinkProgram(pProgram);
    GLint linked;
    glGetProgramiv(pProgram,GL_LINK_STATUS, &linked);
    if (!linked) {
        std::cerr << "Program did not link." << std::endl;
        throw fg::Error("attachAndLinkProgram", __LINE__,
                "OpenGL program linking failed", FG_ERR_GL_ERROR);
    }
    printLinkInfoLog(pProgram);
}

Shaders loadShaders(const char* pVertexShaderSrc,
                    const char* pFragmentShaderSrc,
                    const char* pGeometryShaderSrc)
{
    GLuint f, v;

    v = glCreateShader(GL_VERTEX_SHADER);
    f = glCreateShader(GL_FRAGMENT_SHADER);

    // load shaders & get length of each
    glShaderSource(v, 1, &pVertexShaderSrc, NULL);
    glShaderSource(f, 1, &pFragmentShaderSrc, NULL);

    GLint compiled;

    glCompileShader(v);
    glGetShaderiv(v, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        std::cerr << "Vertex shader not compiled." << std::endl;
        printShaderInfoLog(v);
    }

    glCompileShader(f);
    glGetShaderiv(f, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        std::cerr << "Fragment shader not compiled." << std::endl;
        printShaderInfoLog(f);
    }

    GLuint g = 0;
    /* compile geometry shader if source provided */
    if (pGeometryShaderSrc) {
        g = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(g, 1, &pGeometryShaderSrc, NULL);
        glCompileShader(g);
        glGetShaderiv(g, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            std::cerr << "Geometry shader not compiled." << std::endl;
            printShaderInfoLog(g);
        }
    }

    Shaders out; out.vertex = v; out.fragment = f; out.geometry = g;

    return out;
}

GLuint initShaders(const char* pVertShaderSrc, const char* pFragShaderSrc, const char* pGeomShaderSrc)
{
    Shaders shrds = loadShaders(pVertShaderSrc, pFragShaderSrc, pGeomShaderSrc);
    GLuint shaderProgram = glCreateProgram();
    attachAndLinkProgram(shaderProgram, shrds);
    return shaderProgram;
}

float clampTo01(const float pValue)
{
    return (pValue < 0.0f ? 0.0f : (pValue>1.0f ? 1.0f : pValue));
}

#ifdef OS_WIN
#include <windows.h>
#include <strsafe.h>

void getFontFilePaths(std::vector<std::string>& pFiles,
                      const std::string& pDir,
                      const std::string& pExt)
{
   WIN32_FIND_DATA ffd;
   LARGE_INTEGER filesize;
   TCHAR szDir[MAX_PATH];
   size_t length_of_arg;
   DWORD dwError=0;
   HANDLE hFind = INVALID_HANDLE_VALUE;

   // Check that the input path plus 3 is not longer than MAX_PATH.
   // Three characters are for the "\*" plus NULL appended below.
   StringCchLength(pDir.c_str(), MAX_PATH, &length_of_arg);

   if (length_of_arg > (MAX_PATH - 3)) {
       throw fg::Error("getImageFilePaths", __LINE__,
           "WIN API call: Directory path is too long",
           FG_ERR_FILE_NOT_FOUND);
   }

   //printf("\nTarget directory is %s\n\n", pDir.c_str());
   // Prepare string for use with FindFile functions.  First, copy the
   // string to a buffer, then append '\*' to the directory name.
   StringCchCopy(szDir, MAX_PATH, pDir.c_str());
   std::string wildcard = "\\*" + pExt;
   StringCchCat(szDir, MAX_PATH, wildcard.c_str());

   // Find the first file in the directory.
   hFind = FindFirstFile(szDir, &ffd);
   if (INVALID_HANDLE_VALUE == hFind) {
       throw fg::Error("getImageFilePaths", __LINE__,
           "WIN API call: file fetch in DIR failed",
           FG_ERR_FILE_NOT_FOUND);
   }

   // List all the files in the directory with some info about them.
   do {
      if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
         // It is a directory, skip the entry
         //_tprintf(TEXT("  %s   <DIR>\n"), ffd.cFileName);
      } else {
         filesize.LowPart = ffd.nFileSizeLow;
         filesize.HighPart = ffd.nFileSizeHigh;
         //_tprintf(TEXT("  %s   %ld bytes\n"), ffd.cFileName, filesize.QuadPart);
         pFiles.push_back(std::string(ffd.cFileName));
      }
   } while (FindNextFile(hFind, &ffd) != 0);

   dwError = GetLastError();
   if (dwError != ERROR_NO_MORE_FILES) {
       throw fg::Error("getImageFilePaths", __LINE__,
           "WIN API call: files fetch returned no files",
           FG_ERR_FILE_NOT_FOUND);
   }

   FindClose(hFind);
}
#endif

std::string toString(const float pVal, const int pPrecision)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(pPrecision) << pVal;
    return out.str();
}

GLuint screenQuadVBO(const int pWindowId)
{
    //FIXME: VBOs can be shared, but for simplicity
    // right now just created one VBO each window,
    // ignoring shared contexts
    static std::map<int, GLuint> svboMap;

    if (svboMap.find(pWindowId)==svboMap.end()) {
        static const float vertices[8] = {
            -1.0f,-1.0f,
             1.0f,-1.0f,
             1.0f, 1.0f,
            -1.0f, 1.0f
        };
        svboMap[pWindowId] = createBuffer(GL_ARRAY_BUFFER, 8, vertices, GL_STATIC_DRAW);
    }

    return svboMap[pWindowId];
}

GLuint screenQuadVAO(const int pWindowId)
{
    static std::map<int, GLuint> svaoMap;

    if (svaoMap.find(pWindowId)==svaoMap.end()) {
        static const float texcords[8]  = {0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0};
        static const uint indices[6]    = {0,1,2,0,2,3};

        GLuint tbo  = createBuffer(GL_ARRAY_BUFFER, 8, texcords, GL_STATIC_DRAW);
        GLuint ibo  = createBuffer(GL_ELEMENT_ARRAY_BUFFER, 6, indices, GL_STATIC_DRAW);

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

std::ostream& operator<<(std::ostream& pOut, const glm::mat4& pMat)
{
    const float* ptr = (const float*)glm::value_ptr(pMat);
    pOut << "\n" << std::fixed;
    pOut << ptr[0]  << "\t" << ptr[1]  << "\t" << ptr[2]  << "\t" << ptr[3] << "\n";
    pOut << ptr[4]  << "\t" << ptr[5]  << "\t" << ptr[6]  << "\t" << ptr[7] << "\n";
    pOut << ptr[8]  << "\t" << ptr[9]  << "\t" << ptr[10] << "\t" << ptr[11] << "\n";
    pOut << ptr[12] << "\t" << ptr[13] << "\t" << ptr[14] << "\t" << ptr[15] << "\n";
    pOut << "\n";
    return pOut;
}

glm::vec3 trackballPoint(const float pX, const float pY,
                         const float pWidth, const float pHeight)
{
    glm::vec3 P = glm::vec3(1.0*pX/pWidth*2 - 1.0, 1.0*pY/pHeight*2 - 1.0, 0);

    P.y = -P.y;
    float OP_squared = P.x * P.x + P.y * P.y;
    if (OP_squared <= 1*1) {
        P.z = sqrt(1*1 - OP_squared);
    } else {
        P.z = 0;
        P = glm::normalize(P);
    }
    return P;
}
