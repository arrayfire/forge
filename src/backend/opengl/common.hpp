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
#pragma warning (push, 0)
#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>
#pragma warning (pop)
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include <vector>
#include <iterator>
#include <unordered_map>
#include <utility>

#include <boost/functional/hash.hpp>

using CellIndex = std::tuple<int, int, int>;
using MatrixHashMap = std::unordered_map<CellIndex, glm::mat4>;

namespace std
{

template<> struct hash<CellIndex>
{
    std::size_t operator()(const CellIndex & key) const
    {
        size_t seed = 0;
        boost::hash_combine(seed, std::get<0>(key));
        boost::hash_combine(seed, std::get<1>(key));
        boost::hash_combine(seed, std::get<2>(key));
        return seed;
    }
};

}

typedef unsigned int    uint;
typedef unsigned short  ushort;
typedef unsigned char   uchar;

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
gl::GLenum dtype2gl(const forge::dtype pValue);

/* Convert forge channel format enum to OpenGL enum to indicate color component layout
 *
 * @pValue is the forge type enum
 *
 * @return OpenGL enum indicating color component layout
 */
gl::GLenum ctype2gl(const forge::ChannelFormat pMode);

/* Convert forge channel format enum to OpenGL enum to indicate color component layout
 *
 * This function is used to group color component layout formats based
 * on number of components.
 *
 * @pValue is the forge type enum
 *
 * @return OpenGL enum indicating color component layout
 */
gl::GLenum ictype2gl(const forge::ChannelFormat pMode);

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
gl::GLuint createBuffer(gl::GLenum pTarget, size_t pSize, const T* pPtr, gl::GLenum pUsage)
{
    gl::GLuint retVal = 0;
    gl::glGenBuffers(1, &retVal);
    gl::glBindBuffer(pTarget, retVal);
    gl::glBufferData(pTarget, pSize*sizeof(T), pPtr, pUsage);
    gl::glBindBuffer(pTarget, 0);
    return retVal;
}

#if defined(OS_WIN)
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
 * @pFormat is the c-style printf format for floats
 *
 * @return is the string representation of input float value.
 */
std::string toString(const float pVal, const std::string pFormat);

/* Get a vertex buffer object for quad that spans the screen
 */
gl::GLuint screenQuadVBO(const int pWindowId);

/* Get a vertex array object that uses screenQuadVBO
 *
 * This vertex array object when bound and rendered, basically
 * draws a rectangle over the entire screen with standard
 * texture coordinates. Use of this vao would be as follows
 *
 *     `glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);`
 */
gl::GLuint screenQuadVAO(const int pWindowId);

/* Print glm::mat4 to std::cout stream */
std::ostream& operator<<(std::ostream&, const glm::mat4&);

/* get the point of the surface of track ball */
glm::vec3 trackballPoint(const float pX, const float pY,
                         const float pWidth, const float pHeight);

namespace forge
{
namespace opengl
{

typedef unsigned int    uint;
typedef unsigned short  ushort;
typedef unsigned char   uchar;

class ShaderProgram {
    private:
        gl::GLuint mVertex;
        gl::GLuint mFragment;
        gl::GLuint mGeometry;
        gl::GLuint mProgram;

    public:
        ShaderProgram(const char* pVertShaderSrc,
                      const char* pFragShaderSrc,
                      const char* pGeomShaderSrc=NULL);

        ~ShaderProgram();

        gl::GLuint getProgramId() const;

        gl::GLuint getUniformLocation(const char* pAttributeName);
        gl::GLuint getUniformBlockIndex(const char* pAttributeName);
        gl::GLuint getAttributeLocation(const char* pAttributeName);

        void bind();
        void unbind();
};

/* Basic renderable class
 *
 * Any object that is renderable to a window should inherit from this
 * class.
 */
class AbstractRenderable {
    protected:
        /* OpenGL buffer objects */
        gl::GLuint  mVBO;
        gl::GLuint  mCBO;
        gl::GLuint  mABO;
        size_t      mVBOSize;
        size_t      mCBOSize;
        size_t      mABOSize;
        gl::GLfloat mColor[4];
        gl::GLfloat mRange[6];
        std::string mLegend;
        bool        mIsPVCOn;
        bool        mIsPVAOn;

        AbstractRenderable():
            mVBO(0), mCBO(0), mABO(0),
            mVBOSize(0), mCBOSize(0), mABOSize(0),
            mIsPVCOn(0), mIsPVAOn(0)
        {
            mColor[0] = 0;
            mColor[1] = 0;
            mColor[2] = 0;
            mColor[3] = 0;

            mRange[0] = 0;
            mRange[1] = 0;
            mRange[2] = 0;
            mRange[3] = 0;
            mRange[4] = 0;
            mRange[5] = 0;
        }

    public:
        /* Getter functions for OpenGL buffer objects
         * identifiers and their size in bytes
         *
         *  vbo is for vertices
         *  cbo is for colors of those vertices
         *  abo is for alpha values for those vertices
         */
        gl::GLuint vbo() const { return mVBO; }
        gl::GLuint cbo() { mIsPVCOn = true; return mCBO; }
        gl::GLuint abo() { mIsPVAOn = true; return mABO; }
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
        virtual void setColorMapUBOParams(const gl::GLuint pUBO, const gl::GLuint pSize) {
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
                            const glm::mat4 &pView, const glm::mat4 &pOrient) = 0;
};

}
}
