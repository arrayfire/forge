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

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include <vector>
#include <iterator>

static const float BLACK[]   = {0.0f    , 0.0f    , 0.0f    , 1.0f};
static const float GRAY[]    = {0.85f   , 0.85f   , 0.85f   , 1.0f};
static const float WHITE[]   = {1.0f    , 1.0f    , 1.0f    , 1.0f};
static const float AF_BLUE[] = {0.0588f , 0.1137f , 0.2745f , 1.0f};
static const glm::mat4 IDENTITY(1.0f);

/* clamp the float to [0-1] range
 *
 * @pValue is the value to be clamped
 */
float clampTo01(const float pValue);

/* Convert forge type enum to OpenGL enum for GL_* type
 *
 * @pValue is the forge type enum
 *
 * @return GL_* typedef for data type
 */
GLenum dtype2gl(const fg::dtype pValue);

/* Convert forge channel format enum to OpenGL enum to indicate color component layout
 *
 * @pValue is the forge type enum
 *
 * @return OpenGL enum indicating color component layout
 */
GLenum ctype2gl(const fg::ChannelFormat pMode);

/* Convert forge channel format enum to OpenGL enum to indicate color component layout
 *
 * This function is used to group color component layout formats based
 * on number of components.
 *
 * @pValue is the forge type enum
 *
 * @return OpenGL enum indicating color component layout
 */
GLenum ictype2gl(const fg::ChannelFormat pMode);

/* Compile OpenGL GLSL vertex and fragment shader sources
 *
 * @pVertShaderSrc is the vertex shader source code string
 * @pFragShaderSrc is the vertex shader source code string
 * @pGeomShaderSrc is the vertex shader source code string
 *
 * @return GLSL program unique identifier for given shader duo
 */
GLuint initShaders(const char* pVertShaderSrc, const char* pFragShaderSrc, const char* pGeomShaderSrc=NULL);

/* Create OpenGL buffer object
 *
 * @pTarget should be either GL_ARRAY_BUFFER or GL_ELEMENT_ARRAY_BUFFER
 * @pSize is the size of the data in bytes
 * @pPtr is the pointer to host data. This can be NULL
 * @pUsage should be either GL_STATIC_DRAW or GL_DYNAMIC_DRAW
 *
 * @return OpenGL buffer object identifier
 */
template<typename T>
GLuint createBuffer(GLenum pTarget, size_t pSize, const T* pPtr, GLenum pUsage)
{
    GLuint retVal = 0;
    glGenBuffers(1, &retVal);
    glBindBuffer(pTarget, retVal);
    glBufferData(pTarget, pSize*sizeof(T), pPtr, pUsage);
    glBindBuffer(pTarget, 0);
    return retVal;
}

#ifdef OS_WIN
/* Get the paths to font files in Windows system directory
 *
 * @pFiles is the output vector to which font file paths are appended to.
 * @pDir is the directory from which font files are looked up
 * @pExt is the target font file extension we are looking for.
 */
void getFontFilePaths(std::vector<std::string>& pFiles,
                      const std::string& pDir,
                      const std::string& pExt);
#endif

/* Convert float value to string with given precision
 *
 * @pVal is the float value whose string representation is requested.
 * @pPrecision is the precision of the float used while converting to string.
 *
 * @return is the string representation of input float value.
 */
std::string toString(const float pVal, const int pPrecision = 2);

/* Get a vertex buffer object for quad that spans the screen
 */
GLuint screenQuadVBO(const int pWindowId);

/* Get a vertex array object that uses screenQuadVBO
 *
 * This vertex array object when bound and rendered, basically
 * draws a rectangle over the entire screen with standard
 * texture coordinates. Use of this vao would be as follows
 *
 *     `glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);`
 */
GLuint screenQuadVAO(const int pWindowId);

/* Print glm::mat4 to std::cout stream */
std::ostream& operator<<(std::ostream&, const glm::mat4&);

/* get the point of the surface of track ball */
glm::vec3 trackballPoint(const float pX, const float pY,
                         const float pWidth, const float pHeight);

namespace opengl
{

typedef unsigned int    uint;
typedef unsigned short  ushort;
typedef unsigned char   uchar;

/* Basic renderable class
 *
 * Any object that is renderable to a window should inherit from this
 * class.
 */
class AbstractRenderable {
    protected:
        /* OpenGL buffer objects */
        GLuint      mVBO;
        GLuint      mCBO;
        GLuint      mABO;
        size_t      mVBOSize;
        size_t      mCBOSize;
        size_t      mABOSize;
        GLfloat     mColor[4];
        GLfloat     mRange[6];
        std::string mLegend;
        bool        mIsPVCOn;
        bool        mIsPVAOn;

    public:
        /* Getter functions for OpenGL buffer objects
         * identifiers and their size in bytes
         *
         *  vbo is for vertices
         *  cbo is for colors of those vertices
         *  abo is for alpha values for those vertices
         */
        GLuint vbo() const { return mVBO; }
        GLuint cbo() { mIsPVCOn = true; return mCBO; }
        GLuint abo() { mIsPVAOn = true; return mABO; }
        size_t vboSize() const { return mVBOSize; }
        size_t cboSize() const { return mCBOSize; }
        size_t aboSize() const { return mABOSize; }

        /* Set color for rendering
         */
        void setColor(const float pRed, const float pGreen,
                      const float pBlue, const float pAlpha) {
            mColor[0] = clampTo01(pRed);
            mColor[1] = clampTo01(pGreen);
            mColor[2] = clampTo01(pBlue);
            mColor[3] = clampTo01(pAlpha);
        }

        /* Get renderable solid color
         */
        void getColor(float& pRed, float& pGreen, float& pBlue, float& pAlpha) {
            pRed    = mColor[0];
            pGreen  = mColor[1];
            pBlue   = mColor[2];
            pAlpha  = mColor[3];
        }

        /* Set legend for rendering
         */
        void setLegend(const char* pLegend) {
            mLegend = std::string(pLegend);
        }

        /* Get legend string
         */
        const std::string& legend() const {
            return mLegend;
        }

        /* Set 3d world coordinate ranges
         *
         * This method is mostly used for charts and related renderables
         */
        void setRanges(const float pMinX, const float pMaxX,
                       const float pMinY, const float pMaxY,
                       const float pMinZ, const float pMaxZ) {
            mRange[0] = pMinX; mRange[1] = pMaxX;
            mRange[2] = pMinY; mRange[3] = pMaxY;
            mRange[4] = pMinZ; mRange[5] = pMaxZ;
        }

        /* virtual function to set colormap, a derviced class might
         * use it or ignore it if it doesnt have a need for color maps.
         */
        virtual void setColorMapUBOParams(const GLuint pUBO, const GLuint pSize) {
        }

        /* render is a pure virtual function.
         *
         * @pWindowId is the window identifier
         * @pX is the X coordinate at which the currently bound viewport begins.
         * @pX is the Y coordinate at which the currently bound viewport begins.
         * @pViewPortWidth is the width of the currently bound viewport.
         * @pViewPortHeight is the height of the currently bound viewport.
         *
         * Any concrete class that inherits AbstractRenderable class needs to
         * implement this method to render their OpenGL objects to
         * the currently bound viewport of the Window bearing identifier pWindowId.
         *
         * @return nothing.
         */
        virtual void render(const int pWindowId,
                            const int pX, const int pY, const int pVPW, const int pVPH,
                            const glm::mat4 &pView) = 0;
};

}
