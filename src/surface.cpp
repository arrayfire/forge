/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/surface.h>
#include <surface.hpp>
#include <common.hpp>
#include <shader_headers/marker_fs.hpp>
#include <shader_headers/plot3_vs.hpp>
#include <shader_headers/plot3_fs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>

using namespace std;

void generateGridIndices(unsigned short rows, unsigned short cols, unsigned short *indices)
{
    unsigned short idx = 0;
    for(unsigned short r = 0; r < rows-1; ++r){
        for(unsigned short c = 0; c < cols*2; ++c){
            unsigned short i = c + (r * (cols*2));

            if(c == cols * 2 - 1) {
                *indices++ = idx;
            }else{
                *indices++ = idx;
                if(i%2 == 0){
                    idx += cols;
                } else {
                    idx -= (r%2 == 0) ? (cols-1) : (cols+1);
                }
            }
        }
    }
}

namespace internal
{

void surface_impl::bindResources(const int pWindowId)
{
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach plot vertices
        glEnableVertexAttribArray(mSurfPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(mSurfPointIndex, 3, mDataType, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(mSurfColorIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mCBO);
        glVertexAttribPointer(mSurfColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(mSurfAlphaIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mABO);
        glVertexAttribPointer(mSurfAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        //attach indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void surface_impl::unbindResources() const
{
    glBindVertexArray(0);
}

void surface_impl::computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput)
{
    // identity matrix
    static const glm::mat4 I(1.0f);

    float range_x = mRange[1] - mRange[0];
    float range_y = mRange[3] - mRange[2];
    float range_z = mRange[5] - mRange[4];
    // set scale to zero if input is constant array
    // otherwise compute scale factor by standard equation
    float graph_scale_x = std::abs(range_x) < 1.0e-3 ? 0.0f : 2/(range_x);
    float graph_scale_y = std::abs(range_y) < 1.0e-3 ? 0.0f : 2/(range_y);
    float graph_scale_z = std::abs(range_z) < 1.0e-3 ? 0.0f : 2/(range_z);

    float coor_offset_x = (-mRange[0] * graph_scale_x);
    float coor_offset_y = (-mRange[2] * graph_scale_y);
    float coor_offset_z = (-mRange[4] * graph_scale_z);

    glm::mat4 rMat = glm::rotate(I, -glm::radians(90.f), glm::vec3(1,0,0));
    glm::mat4 tMat = glm::translate(I,
            glm::vec3(-1 + coor_offset_x  , -1 + coor_offset_y, -1 + coor_offset_z));
    glm::mat4 sMat = glm::scale(I,
            glm::vec3(1.0f * graph_scale_x, -1.0f * graph_scale_y, 1.0f * graph_scale_z));

    glm::mat4 model = rMat * tMat * sMat;

    pOut = pInput * model;
}

void surface_impl::renderGraph(const int pWindowId, const glm::mat4& transform)
{
    CheckGL("Begin surface_impl::renderGraph");

    glUseProgram(mSurfProgram);

    glUniformMatrix4fv(mSurfMatIndex, 1, GL_FALSE, glm::value_ptr(transform));
    glUniform2fv(mSurfRangeIndex, 3, mRange);
    glUniform1i(mSurfPVCIndex, mIsPVCOn);

    bindResources(pWindowId);
    glDrawElements(GL_TRIANGLE_STRIP, mIBOSize, GL_UNSIGNED_SHORT, (void*)0 );
    unbindResources();
    glUseProgram(0);

    if(mMarkerType != fg::FG_NONE) {
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(mMarkerProgram);

        glUniformMatrix4fv(mMarkerMatIndex, 1, GL_FALSE, glm::value_ptr(transform));
        glUniform1i(mMarkerPVCIndex, mIsPVCOn);
        glUniform1i(mMarkerTypeIndex, mMarkerType);
        glUniform4fv(mMarkerColIndex, 1, mColor);

        bindResources(pWindowId);
        glDrawElements(GL_POINTS, mIBOSize, GL_UNSIGNED_SHORT, (void*)0);
        unbindResources();

        glUseProgram(0);
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
    CheckGL("End surface_impl::renderGraph");
}


surface_impl::surface_impl(unsigned pNumXPoints, unsigned pNumYPoints,
                           fg::dtype pDataType, fg::MarkerType pMarkerType)
    : mNumXPoints(pNumXPoints),mNumYPoints(pNumYPoints), mDataType(dtype2gl(pDataType)),
      mIsPVCOn(false), mMarkerType(pMarkerType), mIBO(0), mIBOSize(0), mMarkerProgram(-1),
      mSurfProgram(-1), mMarkerMatIndex(-1), mMarkerPointIndex(-1), mMarkerColorIndex(-1),
      mMarkerAlphaIndex(-1), mMarkerPVCIndex(-1), mMarkerTypeIndex(-1), mMarkerColIndex(-1),
      mSurfMatIndex(-1), mSurfRangeIndex(-1), mSurfPointIndex(-1), mSurfColorIndex(-1),
      mSurfAlphaIndex(-1), mSurfPVCIndex(-1)
{
    CheckGL("Begin surface_impl::surface_impl");
    setColor(0.9, 0.5, 0.6, 1.0);
    setLegend(std::string(""));

    mMarkerProgram   = initShaders(glsl::plot3_vs.c_str(), glsl::marker_fs.c_str());
    mMarkerMatIndex  = glGetUniformLocation(mMarkerProgram, "transform");
    mMarkerPVCIndex  = glGetUniformLocation(mMarkerProgram, "isPVCOn");
    mMarkerTypeIndex = glGetUniformLocation(mMarkerProgram, "marker_type");
    mMarkerColIndex  = glGetUniformLocation(mMarkerProgram, "marker_color");
    mMarkerPointIndex= glGetAttribLocation (mMarkerProgram, "point");
    mMarkerColorIndex= glGetAttribLocation (mMarkerProgram, "color");
    mMarkerAlphaIndex= glGetAttribLocation (mMarkerProgram, "alpha");

    mSurfProgram    = initShaders(glsl::plot3_vs.c_str(), glsl::plot3_fs.c_str());
    mSurfMatIndex   = glGetUniformLocation(mSurfProgram, "transform");
    mSurfRangeIndex = glGetUniformLocation(mSurfProgram, "minmaxs");
    mSurfPVCIndex   = glGetUniformLocation(mSurfProgram, "isPVCOn");
    mSurfPointIndex = glGetAttribLocation (mSurfProgram, "point");
    mSurfColorIndex = glGetAttribLocation (mSurfProgram, "color");
    mSurfAlphaIndex = glGetAttribLocation (mSurfProgram, "alpha");

    unsigned totalPoints = mNumXPoints * mNumYPoints;

    mVBOSize = 3*totalPoints;
    mCBOSize = 3*totalPoints;
    mABOSize = totalPoints;
#define SURF_CREATE_BUFFERS(type) \
    mVBO = createBuffer<type>(GL_ARRAY_BUFFER, mVBOSize, NULL, GL_DYNAMIC_DRAW);  \
    mCBO = createBuffer<float>(GL_ARRAY_BUFFER, mCBOSize, NULL, GL_DYNAMIC_DRAW); \
    mABO = createBuffer<float>(GL_ARRAY_BUFFER, mABOSize, NULL, GL_DYNAMIC_DRAW); \
    mVBOSize *= sizeof(type);   \
    mCBOSize *= sizeof(float);  \
    mABOSize *= sizeof(float);

    switch(mDataType) {
        case GL_FLOAT          : SURF_CREATE_BUFFERS(float) ; break;
        case GL_INT            : SURF_CREATE_BUFFERS(int)   ; break;
        case GL_UNSIGNED_INT   : SURF_CREATE_BUFFERS(uint)  ; break;
        case GL_SHORT          : SURF_CREATE_BUFFERS(short) ; break;
        case GL_UNSIGNED_SHORT : SURF_CREATE_BUFFERS(ushort); break;
        case GL_UNSIGNED_BYTE  : SURF_CREATE_BUFFERS(float) ; break;
        default: fg::TypeError("surface_impl::surface_impl", __LINE__, 1, pDataType);
    }

#undef SURF_CREATE_BUFFERS

    mIBOSize = (2 * mNumYPoints) * (mNumXPoints - 1);
    std::vector<ushort> indices(mIBOSize);
    generateGridIndices(mNumXPoints, mNumYPoints, indices.data());
    mIBO = createBuffer<ushort>(GL_ELEMENT_ARRAY_BUFFER, mIBOSize, indices.data(), GL_STATIC_DRAW);

    CheckGL("End surface_impl::surface_impl");
}

surface_impl::~surface_impl()
{
    CheckGL("Begin Plot::~Plot");
    for (auto it = mVAOMap.begin(); it!=mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mVBO);
    glDeleteBuffers(1, &mCBO);
    glDeleteBuffers(1, &mABO);
    glDeleteBuffers(1, &mIBO);
    glDeleteProgram(mMarkerProgram);
    glDeleteProgram(mSurfProgram);
    CheckGL("End Plot::~Plot");
}

void surface_impl::render(const int pWindowId,
                          const int pX, const int pY, const int pVPW, const int pVPH,
                          const glm::mat4 &pModel)
{
    CheckGL("Begin surface_impl::render");
    glm::mat4 mvp(1.0);
    computeTransformMat(mvp, pModel);
    renderGraph(pWindowId, mvp);
    CheckGL("End surface_impl::render");
}

void scatter3_impl::renderGraph(const int pWindowId, const glm::mat4& transform)
{
    if(mMarkerType != fg::FG_NONE) {
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(mMarkerProgram);

        glUniformMatrix4fv(mMarkerMatIndex, 1, GL_FALSE, glm::value_ptr(transform));
        glUniform1i(mMarkerPVCIndex, mIsPVCOn);
        glUniform1i(mMarkerTypeIndex, mMarkerType);
        glUniform4fv(mMarkerColIndex, 1, mColor);

        bindResources(pWindowId);
        glDrawElements(GL_POINTS, mIBOSize, GL_UNSIGNED_SHORT, (void*)0);
        unbindResources();

        glUseProgram(0);
        glDisable(GL_PROGRAM_POINT_SIZE);
    }
}

}

namespace fg
{

Surface::Surface(unsigned pNumXPoints, unsigned pNumYPoints, dtype pDataType, PlotType pPlotType, MarkerType pMarkerType)
{
    mValue = new internal::_Surface(pNumXPoints, pNumYPoints, pDataType, pPlotType, pMarkerType);
}

Surface::Surface(const Surface& other)
{
    mValue = new internal::_Surface(*other.get());
}

Surface::~Surface()
{
    delete mValue;
}

void Surface::setColor(const Color pColor)
{
    float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
    float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
    float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
    float a = (((int) pColor       ) & 0xFF ) / 255.f;
    mValue->setColor(r, g, b, a);
}

void Surface::setColor(const float pRed, const float pGreen,
                    const float pBlue, const float pAlpha)
{
    mValue->setColor(pRed, pGreen, pBlue, pAlpha);
}

void Surface::setLegend(const std::string& pLegend)
{
    mValue->setLegend(pLegend);
}

uint Surface::vertices() const
{
    return mValue->vbo();
}

uint Surface::colors() const
{
    return mValue->cbo();
}

uint Surface::alphas() const
{
    return mValue->abo();
}

uint Surface::verticesSize() const
{
    return (uint)mValue->vboSize();
}

uint Surface::colorsSize() const
{
    return (uint)mValue->cboSize();
}

uint Surface::alphasSize() const
{
    return (uint)mValue->aboSize();
}

internal::_Surface* Surface::get() const
{
    return mValue;
}

}
