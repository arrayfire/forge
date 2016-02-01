/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot.h>
#include <plot.hpp>

#include <cmath>

using namespace std;

// identity matrix
static const glm::mat4 I(1.0f);

namespace internal
{

void plot_impl::bindResources(const int pWindowId)
{
    if (mVAOMap.find(pWindowId) == mVAOMap.end()) {
        GLuint vao = 0;
        /* create a vertex array object
         * with appropriate bindings */
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        // attach vertices
        glEnableVertexAttribArray(mPlotPointIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mVBO);
        glVertexAttribPointer(mPlotPointIndex, mDimension, mGLType, GL_FALSE, 0, 0);
        // attach colors
        glEnableVertexAttribArray(mPlotColorIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mCBO);
        glVertexAttribPointer(mPlotColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
        // attach alphas
        glEnableVertexAttribArray(mPlotAlphaIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mABO);
        glVertexAttribPointer(mPlotAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        // attach radii
        glEnableVertexAttribArray(mMarkerRadiiIndex);
        glBindBuffer(GL_ARRAY_BUFFER, mRBO);
        glVertexAttribPointer(mMarkerRadiiIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glBindVertexArray(0);
        /* store the vertex array object corresponding to
         * the window instance in the map */
        mVAOMap[pWindowId] = vao;
    }

    glBindVertexArray(mVAOMap[pWindowId]);
}

void plot_impl::unbindResources() const
{
    glBindVertexArray(0);
}

void plot_impl::computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput,
                                    const int pX, const int pY,
                                    const int pVPW, const int pVPH)
{
    float range_x = mRange[1] - mRange[0];
    float range_y = mRange[3] - mRange[2];
    float range_z       = mRange[5] - mRange[4];
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

void plot_impl::bindDimSpecificUniforms()
{
    glUniform2fv(mPlotRangeIndex, 3, mRange);
}

plot_impl::plot_impl(const uint pNumPoints, const fg::dtype pDataType,
                     const fg::PlotType pPlotType, const fg::MarkerType pMarkerType, const int pD)
    : mDimension(pD), mMarkerSize(10), mNumPoints(pNumPoints), mDataType(pDataType),
    mGLType(dtype2gl(mDataType)), mMarkerType(pMarkerType), mPlotType(pPlotType), mIsPVROn(false),
    mPlotProgram(-1), mMarkerProgram(-1), mRBO(-1), mPlotMatIndex(-1), mPlotPVCOnIndex(-1),
    mPlotPVAOnIndex(-1), mPlotUColorIndex(-1), mPlotRangeIndex(-1), mPlotPointIndex(-1),
    mPlotColorIndex(-1), mPlotAlphaIndex(-1), mMarkerPVCOnIndex(-1), mMarkerPVAOnIndex(-1),
    mMarkerTypeIndex(-1), mMarkerColIndex(-1), mMarkerMatIndex(-1), mMarkerPointIndex(-1),
    mMarkerColorIndex(-1), mMarkerAlphaIndex(-1), mMarkerRadiiIndex(-1)
{
    CheckGL("Begin plot_impl::plot_impl");
    mIsPVCOn = false;
    mIsPVAOn = false;

    setColor(0, 1, 0, 1);
    setLegend(std::string(""));

    if (mDimension==2) {
        mPlotProgram     = initShaders(glsl::marker2d_vs.c_str(), glsl::histogram_fs.c_str());
        mMarkerProgram   = initShaders(glsl::marker2d_vs.c_str(), glsl::marker_fs.c_str());
        mPlotUColorIndex = glGetUniformLocation(mPlotProgram, "barColor");
        mVBOSize = 2*mNumPoints;
    } else {
        mPlotProgram     = initShaders(glsl::plot3_vs.c_str(), glsl::plot3_fs.c_str());
        mMarkerProgram   = initShaders(glsl::plot3_vs.c_str(), glsl::marker_fs.c_str());
        mPlotRangeIndex  = glGetUniformLocation(mPlotProgram, "minmaxs");
        mVBOSize = 3*mNumPoints;
    }

    mCBOSize = 3*mNumPoints;
    mABOSize = mNumPoints;
    mRBOSize = mNumPoints;

    mPlotMatIndex    = glGetUniformLocation(mPlotProgram, "transform");
    mPlotPVCOnIndex  = glGetUniformLocation(mPlotProgram, "isPVCOn");
    mPlotPVAOnIndex  = glGetUniformLocation(mPlotProgram, "isPVAOn");
    mPlotPointIndex  = glGetAttribLocation (mPlotProgram, "point");
    mPlotColorIndex  = glGetAttribLocation (mPlotProgram, "color");
    mPlotAlphaIndex  = glGetAttribLocation (mPlotProgram, "alpha");

    mMarkerMatIndex   = glGetUniformLocation(mMarkerProgram, "transform");
    mMarkerPVCOnIndex = glGetUniformLocation(mMarkerProgram, "isPVCOn");
    mMarkerPVAOnIndex = glGetUniformLocation(mMarkerProgram, "isPVAOn");
    mMarkerPVROnIndex = glGetUniformLocation(mMarkerProgram, "isPVROn");
    mMarkerTypeIndex  = glGetUniformLocation(mMarkerProgram, "marker_type");
    mMarkerColIndex   = glGetUniformLocation(mMarkerProgram, "marker_color");
    mMarkerPSizeIndex = glGetUniformLocation(mMarkerProgram, "psize");
    mMarkerPointIndex = glGetAttribLocation (mMarkerProgram, "point");
    mMarkerColorIndex = glGetAttribLocation (mMarkerProgram, "color");
    mMarkerAlphaIndex = glGetAttribLocation (mMarkerProgram, "alpha");
    mMarkerRadiiIndex = glGetAttribLocation (mMarkerProgram, "pointsize");

#define PLOT_CREATE_BUFFERS(type)   \
        mVBO = createBuffer<type>(GL_ARRAY_BUFFER, mVBOSize, NULL, GL_DYNAMIC_DRAW);    \
        mCBO = createBuffer<float>(GL_ARRAY_BUFFER, mCBOSize, NULL, GL_DYNAMIC_DRAW);   \
        mABO = createBuffer<float>(GL_ARRAY_BUFFER, mABOSize, NULL, GL_DYNAMIC_DRAW);   \
        mRBO = createBuffer<float>(GL_ARRAY_BUFFER, mRBOSize, NULL, GL_DYNAMIC_DRAW);   \
        mVBOSize *= sizeof(type);   \
        mCBOSize *= sizeof(float);  \
        mABOSize *= sizeof(float);  \
        mRBOSize *= sizeof(float);

        switch(mGLType) {
            case GL_FLOAT          : PLOT_CREATE_BUFFERS(float) ; break;
            case GL_INT            : PLOT_CREATE_BUFFERS(int)   ; break;
            case GL_UNSIGNED_INT   : PLOT_CREATE_BUFFERS(uint)  ; break;
            case GL_SHORT          : PLOT_CREATE_BUFFERS(short) ; break;
            case GL_UNSIGNED_SHORT : PLOT_CREATE_BUFFERS(ushort); break;
            case GL_UNSIGNED_BYTE  : PLOT_CREATE_BUFFERS(float) ; break;
            default: fg::TypeError("plot_impl::plot_impl", __LINE__, 1, mDataType);
        }
#undef PLOT_CREATE_BUFFERS
        CheckGL("End plot_impl::plot_impl");
}

plot_impl::~plot_impl()
{
    CheckGL("Begin plot_impl::~plot_impl");
    for (auto it = mVAOMap.begin(); it!=mVAOMap.end(); ++it) {
        GLuint vao = it->second;
        glDeleteVertexArrays(1, &vao);
    }
    glDeleteBuffers(1, &mVBO);
    glDeleteBuffers(1, &mCBO);
    glDeleteBuffers(1, &mABO);
    glDeleteProgram(mPlotProgram);
    glDeleteProgram(mMarkerProgram);
    CheckGL("End plot_impl::~plot_impl");
}

void plot_impl::setMarkerSize(const float pMarkerSize)
{
    mMarkerSize = pMarkerSize;
}

GLuint plot_impl::markers()
{
    mIsPVROn = true;
    return mRBO;
}

size_t plot_impl::markersSizes() const
{
    return mRBOSize;
}

void plot_impl::render(const int pWindowId,
                       const int pX, const int pY, const int pVPW, const int pVPH,
                       const glm::mat4& pTransform)
{
    CheckGL("Begin plot_impl::render");
    glEnable(GL_SCISSOR_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    glm::mat4 mvp(1.0);
    this->computeTransformMat(mvp, pTransform, pX, pY, pVPW, pVPH);

    if (mPlotType == fg::FG_LINE) {
        glUseProgram(mPlotProgram);

        this->bindDimSpecificUniforms();
        glUniformMatrix4fv(mPlotMatIndex, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1i(mPlotPVCOnIndex, mIsPVCOn);
        glUniform1i(mPlotPVAOnIndex, mIsPVAOn);

        plot_impl::bindResources(pWindowId);
        glDrawArrays(GL_LINE_STRIP, 0, mNumPoints);
        plot_impl::unbindResources();

        glUseProgram(0);
    }

    if (mMarkerType != fg::FG_NONE) {
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(mMarkerProgram);

        glUniformMatrix4fv(mMarkerMatIndex, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1i(mMarkerPVCOnIndex, mIsPVCOn);
        glUniform1i(mMarkerPVAOnIndex, mIsPVAOn);
        glUniform1i(mMarkerPVROnIndex, mIsPVROn);
        glUniform1i(mMarkerTypeIndex, mMarkerType);
        glUniform4fv(mMarkerColIndex, 1, mColor);
        glUniform1f(mMarkerPSizeIndex, mMarkerSize);

        plot_impl::bindResources(pWindowId);
        glDrawArrays(GL_POINTS, 0, mNumPoints);
        plot_impl::unbindResources();

        glUseProgram(0);
        glDisable(GL_PROGRAM_POINT_SIZE);
    }

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    CheckGL("End plot_impl::render");
}

void plot2d_impl::computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput,
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
    const float bMargin = 32;
    const float tickSize = 10;

    float viewWidth    = pVPW - (lMargin + rMargin + tickSize/2);
    float viewHeight   = pVPH - (bMargin + tMargin + tickSize );
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

void plot2d_impl::bindDimSpecificUniforms()
{
    glUniform4fv(mPlotUColorIndex, 1, mColor);
}

}

namespace fg
{

Plot::Plot(const uint pNumPoints, const dtype pDataType, const ChartType pChartType,
           const PlotType pPlotType, const MarkerType pMarkerType)
{
    mValue = new internal::_Plot(pNumPoints, pDataType, pPlotType, pMarkerType, pChartType);
}

Plot::Plot(const Plot& pOther)
{
    mValue = new internal::_Plot(*pOther.get());
}

Plot::~Plot()
{
    delete mValue;
}

void Plot::setColor(const Color pColor)
{
    float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
    float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
    float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
    float a = (((int) pColor       ) & 0xFF ) / 255.f;
    mValue->setColor(r, g, b, a);
}

void Plot::setColor(const float pRed, const float pGreen,
                    const float pBlue, const float pAlpha)
{
    mValue->setColor(pRed, pGreen, pBlue, pAlpha);
}

void Plot::setLegend(const std::string& pLegend)
{
    mValue->setLegend(pLegend);
}

void Plot::setMarkerSize(const float pMarkerSize)
{
    mValue->setMarkerSize(pMarkerSize);
}

uint Plot::vertices() const
{
    return mValue->vbo();
}

uint Plot::colors() const
{
    return mValue->cbo();
}

uint Plot::alphas() const
{
    return mValue->abo();
}

uint Plot::markers() const
{
    return mValue->mbo();
}

uint Plot::verticesSize() const
{
    return (uint)mValue->vboSize();
}

uint Plot::colorsSize() const
{
    return (uint)mValue->cboSize();
}

uint Plot::alphasSize() const
{
    return (uint)mValue->aboSize();
}

uint Plot::markersSize() const
{
    return (uint)mValue->mboSize();
}

internal::_Plot* Plot::get() const
{
    return mValue;
}

}
