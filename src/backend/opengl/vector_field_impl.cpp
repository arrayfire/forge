/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_opengl.hpp>
#include <vector_field_impl.hpp>
#include <shader_headers/vector_field2d_vs.hpp>
#include <shader_headers/vector_field2d_gs.hpp>
#include <shader_headers/vector_field_vs.hpp>
#include <shader_headers/vector_field_gs.hpp>
#include <shader_headers/histogram_fs.hpp>

#include <cmath>

using namespace std;

// identity matrix
static const glm::mat4 I(1.0f);

namespace opengl
{

void vector_field_impl::bindResources(const int pWindowId)
{
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach vertices
        glEnableVertexAttribArray(mFieldPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(mFieldPointIndex, mDimension, mGLType, GL_FALSE, 0, 0);
        // attach colors
        glEnableVertexAttribArray(mFieldColorIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mCBO);
        glVertexAttribPointer(mFieldColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        // attach alphas
        glEnableVertexAttribArray(mFieldAlphaIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mABO);
        glVertexAttribPointer(mFieldAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        // attach field directions
        glEnableVertexAttribArray(mFieldDirectionIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mDBO);
        glVertexAttribPointer(mFieldDirectionIndex, mDimension, GL_FLOAT, GL_FALSE, 0, 0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void vector_field_impl::unbindResources() const
{
    glBindVertexArray(0);
}

void vector_field_impl::computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput,
                                            const int pX, const int pY,
                                            const int pVPW, const int pVPH)
{
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

    glm::mat4 model= rMat * tMat * sMat;

    pOut = pInput * model;
    glScissor(pX, pY, pVPW, pVPH);
}

vector_field_impl::vector_field_impl(const uint pNumPoints, const fg::dtype pDataType, const int pD)
    : mDimension(pD), mNumPoints(pNumPoints), mDataType(pDataType), mGLType(dtype2gl(mDataType)),
    mFieldProgram(-1), mDBO(-1), mDBOSize(0), mFieldMatIndex(-1), mFieldPointIndex(-1),
    mFieldColorIndex(-1), mFieldAlphaIndex(-1), mFieldDirectionIndex(-1),
    mFieldPVCOnIndex(-1), mFieldPVAOnIndex(-1), mFieldUColorIndex(-1)
{
    CheckGL("Begin vector_field_impl::vector_field_impl");
    mIsPVCOn = false;
    mIsPVAOn = false;

    setColor(0, 1, 0, 1);
    mLegend  = std::string("");

    // FIXME
    if (mDimension==2) {
        mFieldProgram = initShaders(glsl::vector_field2d_vs.c_str(), glsl::histogram_fs.c_str(),
                                    glsl::vector_field2d_gs.c_str());
        mVBOSize = 2*mNumPoints;
        mDBOSize = 2*mNumPoints;
    } else {
        mFieldProgram     = initShaders(glsl::vector_field_vs.c_str(), glsl::histogram_fs.c_str(),
                                        glsl::vector_field_gs.c_str());
        mVBOSize = 3*mNumPoints;
        mDBOSize = 3*mNumPoints;
    }

    mCBOSize = 3*mNumPoints;
    mABOSize = mNumPoints;

    mFieldMatIndex    = glGetUniformLocation(mFieldProgram, "transform");
    mFieldPointIndex  = glGetAttribLocation (mFieldProgram, "point");
    mFieldColorIndex  = glGetAttribLocation (mFieldProgram, "color");
    mFieldAlphaIndex  = glGetAttribLocation (mFieldProgram, "alpha");
    mFieldDirectionIndex  = glGetAttribLocation (mFieldProgram, "direction");

    mFieldPVCOnIndex  = glGetUniformLocation(mFieldProgram, "isPVCOn");
    mFieldPVAOnIndex  = glGetUniformLocation(mFieldProgram, "isPVAOn");
    mFieldUColorIndex = glGetUniformLocation(mFieldProgram, "barColor");

#define PLOT_CREATE_BUFFERS(type)   \
        mVBO = createBuffer<type>(GL_ARRAY_BUFFER, mVBOSize, NULL, GL_DYNAMIC_DRAW);    \
        mCBO = createBuffer<float>(GL_ARRAY_BUFFER, mCBOSize, NULL, GL_DYNAMIC_DRAW);   \
        mABO = createBuffer<float>(GL_ARRAY_BUFFER, mABOSize, NULL, GL_DYNAMIC_DRAW);   \
        mDBO = createBuffer<float>(GL_ARRAY_BUFFER, mDBOSize, NULL, GL_DYNAMIC_DRAW);   \
        mVBOSize *= sizeof(type);   \
        mCBOSize *= sizeof(float);  \
        mABOSize *= sizeof(float);  \
        mDBOSize *= sizeof(float);

        switch(mGLType) {
            case GL_FLOAT          : PLOT_CREATE_BUFFERS(float) ; break;
            case GL_INT            : PLOT_CREATE_BUFFERS(int)   ; break;
            case GL_UNSIGNED_INT   : PLOT_CREATE_BUFFERS(uint)  ; break;
            case GL_SHORT          : PLOT_CREATE_BUFFERS(short) ; break;
            case GL_UNSIGNED_SHORT : PLOT_CREATE_BUFFERS(ushort); break;
            case GL_UNSIGNED_BYTE  : PLOT_CREATE_BUFFERS(float) ; break;
            default: fg::TypeError("vector_field_impl::vector_field_impl", __LINE__, 1, mDataType);
        }
#undef PLOT_CREATE_BUFFERS
        CheckGL("End vector_field_impl::vector_field_impl");
}

vector_field_impl::~vector_field_impl()
{
    CheckGL("Begin vector_field_impl::~vector_field_impl");
    for (auto it = mVAOMap.begin(); it!=mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mVBO);
    glDeleteBuffers(1, &mCBO);
    glDeleteBuffers(1, &mABO);
    glDeleteBuffers(1, &mDBO);
    glDeleteProgram(mFieldProgram);
    CheckGL("End vector_field_impl::~vector_field_impl");
}

GLuint vector_field_impl::directions()
{
    return mDBO;
}

size_t vector_field_impl::directionsSize() const
{
    return mDBOSize;
}

void vector_field_impl::render(const int pWindowId,
                       const int pX, const int pY, const int pVPW, const int pVPH,
                       const glm::mat4& pTransform)
{
    CheckGL("Begin vector_field_impl::render");
    glEnable(GL_SCISSOR_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glm::mat4 mvp(1.0);
    this->computeTransformMat(mvp, pTransform, pX, pY, pVPW, pVPH);

    glUseProgram(mFieldProgram);

    glUniformMatrix4fv(mFieldMatIndex, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform1i(mFieldPVCOnIndex, mIsPVCOn);
    glUniform1i(mFieldPVAOnIndex, mIsPVAOn);
    glUniform4fv(mFieldUColorIndex, 1, mColor);

    //FIXME remove debug polygon mode setting
    vector_field_impl::bindResources(pWindowId);
    glDrawArrays(GL_POINTS, 0, mNumPoints);
    vector_field_impl::unbindResources();

    glUseProgram(0);

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glDisable(GL_SCISSOR_TEST);
    CheckGL("End vector_field_impl::render");
}

void vector_field2d_impl::computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput,
                                      const int pX, const int pY,
                                      const int pVPW, const int pVPH)
{
    float range_x = mRange[1] - mRange[0];
    float range_y = mRange[3] - mRange[2];
    // set scale to zero if input is constant array
    // otherwise compute scale factor by standard equation
    float graph_scale_x = std::abs(range_x) < 1.0e-3 ? 0.0f : 2/(range_x);
    float graph_scale_y = std::abs(range_y) < 1.0e-3 ? 0.0f : 2/(range_y);

    float coor_offset_x = (-mRange[0] * graph_scale_x);
    float coor_offset_y = (-mRange[2] * graph_scale_y);

    //FIXME: Using hard constants for now, find a way to get chart values
    const float lMargin = 68;
    const float rMargin = 8;
    const float tMargin = 8;
    const float bMargin = 44;
    const float tickSize = 10;

    float viewWidth    = pVPW - (lMargin + rMargin + tickSize/2);
    float viewHeight   = pVPH - (bMargin + tMargin + tickSize/2);
    float view_scale_x = viewWidth/pVPW;
    float view_scale_y = viewHeight/pVPH;

    coor_offset_x *= view_scale_x;
    coor_offset_y *= view_scale_y;

    float view_offset_x = (2.0f * (lMargin + tickSize/2 )/ pVPW ) ;
    float view_offset_y = (2.0f * (bMargin + tickSize )/ pVPH ) ;

    glm::mat4 tMat = glm::translate(I,
            glm::vec3(-1 + view_offset_x + coor_offset_x  , -1 + view_offset_y + coor_offset_y, 0));
    pOut = glm::scale(tMat,
            glm::vec3(graph_scale_x * view_scale_x , graph_scale_y * view_scale_y ,1));

    pOut = pInput * pOut;

    glScissor(pX + lMargin + tickSize/2, pY+bMargin + tickSize/2,
              pVPW - lMargin - rMargin - tickSize/2,
              pVPH - bMargin - tMargin - tickSize/2);
}

}
