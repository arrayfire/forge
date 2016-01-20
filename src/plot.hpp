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
#include <common.hpp>
#include <shader_headers/marker2d_vs.hpp>
#include <shader_headers/marker_fs.hpp>
#include <shader_headers/histogram_fs.hpp>
#include <shader_headers/plot3_vs.hpp>
#include <shader_headers/plot3_fs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include <map>

namespace internal
{

template<fg::ChartType PLOT_TYPE>
class plot_impl : public AbstractRenderable {
    protected:
        /* plot points characteristics */
        GLuint    mNumPoints;
        fg::dtype mDataType;
        GLenum    mGLType;
        bool      mIsPVCOn;
        fg::MarkerType mMarkerType;
        fg::PlotType   mPlotType;
        /* OpenGL Objects */
        GLuint    mPlotProgram;
        GLuint    mMarkerProgram;
        /* shaderd variable index locations */
        GLuint    mPlotMatIndex;
        GLuint    mPlotPVCOnIndex;
        GLuint    mPlotUColorIndex;
        GLuint    mPlotRangeIndex;
        GLuint    mPlotPointIndex;
        GLuint    mPlotColorIndex;
        GLuint    mPlotAlphaIndex;

        GLuint    mMarkerPVCOnIndex;
        GLuint    mMarkerTypeIndex;
        GLuint    mMarkerColIndex;
        GLuint    mMarkerMatIndex;
        GLuint    mMarkerPointIndex;
        GLuint    mMarkerColorIndex;
        GLuint    mMarkerAlphaIndex;

        std::map<int, GLuint> mVAOMap;

        /* bind and unbind helper functions
         * for rendering resources */
        void bindResources(const int pWindowId)
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
                if (PLOT_TYPE==fg::FG_2D)
                    glVertexAttribPointer(mPlotPointIndex, 2, mGLType, GL_FALSE, 0, 0);
                else if (PLOT_TYPE==fg::FG_3D)
                    glVertexAttribPointer(mPlotPointIndex, 3, mGLType, GL_FALSE, 0, 0);
                // attach colors
                glEnableVertexAttribArray(mPlotColorIndex);
                glBindBuffer(GL_ARRAY_BUFFER, mCBO);
                glVertexAttribPointer(mPlotColorIndex, 3, GL_FLOAT, GL_FALSE, 0, 0);
                // attach alphas
                glEnableVertexAttribArray(mPlotAlphaIndex);
                glBindBuffer(GL_ARRAY_BUFFER, mABO);
                glVertexAttribPointer(mPlotAlphaIndex, 1, GL_FLOAT, GL_FALSE, 0, 0);
                glBindVertexArray(0);
                /* store the vertex array object corresponding to
                 * the window instance in the map */
                mVAOMap[pWindowId] = vao;
            }

            glBindVertexArray(mVAOMap[pWindowId]);
        }

        void unbindResources() const
        {
            glBindVertexArray(0);
        }

        void computeTransformMat(glm::mat4& pOut, const glm::mat4 pInput,
                                 const int pX, const int pY,
                                 const int pVPW, const int pVPH)
        {
            // identity matrix
            static const glm::mat4 I(1.0f);

            float range_x = mRange[1] - mRange[0];
            float range_y = mRange[3] - mRange[2];
            // set scale to zero if input is constant array
            // otherwise compute scale factor by standard equation
            float graph_scale_x = std::abs(range_x) < 1.0e-3 ? 0.0f : 2/(range_x);
            float graph_scale_y = std::abs(range_y) < 1.0e-3 ? 0.0f : 2/(range_y);

            float coor_offset_x = (-mRange[0] * graph_scale_x);
            float coor_offset_y = (-mRange[2] * graph_scale_y);

            if (PLOT_TYPE == fg::FG_3D) {
                float range_z       = mRange[5] - mRange[4];
                float graph_scale_z = std::abs(range_z) < 1.0e-3 ? 0.0f : 2/(range_z);
                float coor_offset_z = (-mRange[4] * graph_scale_z);

                glm::mat4 rMat = glm::rotate(I, -glm::radians(90.f), glm::vec3(1,0,0));
                glm::mat4 tMat = glm::translate(I,
                        glm::vec3(-1 + coor_offset_x  , -1 + coor_offset_y, -1 + coor_offset_z));
                glm::mat4 sMat = glm::scale(I,
                        glm::vec3(1.0f * graph_scale_x, -1.0f * graph_scale_y, 1.0f * graph_scale_z));

                glm::mat4 model= rMat * tMat * sMat;

                pOut = pInput * model;
                glScissor(pX, pY, pVPW, pVPH);
            } else if (PLOT_TYPE == fg::FG_2D) {
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

                glScissor(pX + lMargin + tickSize/2, pY+bMargin + tickSize/2,
                          pVPW - lMargin - rMargin - tickSize/2,
                          pVPH - bMargin - tMargin - tickSize/2);
            }
        }

    public:
        plot_impl(const uint pNumPoints, const fg::dtype pDataType,
                  const fg::PlotType pPlotType, const fg::MarkerType pMarkerType)
            : mNumPoints(pNumPoints), mDataType(pDataType), mGLType(dtype2gl(mDataType)),
            mIsPVCOn(false), mMarkerType(pMarkerType), mPlotType(pPlotType),
            mPlotProgram(-1), mMarkerProgram(-1), mPlotMatIndex(-1), mPlotPVCOnIndex(-1),
            mPlotUColorIndex(-1), mPlotRangeIndex(-1), mPlotPointIndex(-1), mPlotColorIndex(-1),
            mPlotAlphaIndex(-1), mMarkerPVCOnIndex(-1), mMarkerTypeIndex(-1),
            mMarkerColIndex(-1), mMarkerMatIndex(-1), mMarkerPointIndex(-1),
            mMarkerColorIndex(-1), mMarkerAlphaIndex(-1)
        {
            CheckGL("Begin plot_impl::plot_impl");

            setColor(0, 1, 0, 1);
            setLegend(std::string(""));

            if (PLOT_TYPE==fg::FG_2D) {
                mPlotProgram     = initShaders(glsl::marker2d_vs.c_str(), glsl::histogram_fs.c_str());
                mMarkerProgram   = initShaders(glsl::marker2d_vs.c_str(), glsl::marker_fs.c_str());
                mPlotUColorIndex = glGetUniformLocation(mPlotProgram, "barColor");
                mVBOSize = 2*mNumPoints;
            } else  if (PLOT_TYPE==fg::FG_3D) {
                mPlotProgram     = initShaders(glsl::plot3_vs.c_str(), glsl::plot3_fs.c_str());
                mMarkerProgram   = initShaders(glsl::plot3_vs.c_str(), glsl::marker_fs.c_str());
                mPlotRangeIndex  = glGetUniformLocation(mPlotProgram, "minmaxs");
                mVBOSize = 3*mNumPoints;
            }
            mCBOSize = 3*mNumPoints;
            mABOSize = mNumPoints;

            mPlotMatIndex    = glGetUniformLocation(mPlotProgram, "transform");
            mPlotPVCOnIndex  = glGetUniformLocation(mPlotProgram, "isPVCOn");
            mPlotPointIndex  = glGetAttribLocation (mPlotProgram, "point");
            mPlotColorIndex  = glGetAttribLocation (mPlotProgram, "color");
            mPlotAlphaIndex  = glGetAttribLocation (mPlotProgram, "alpha");

            mMarkerMatIndex   = glGetUniformLocation(mMarkerProgram, "transform");
            mMarkerPVCOnIndex = glGetUniformLocation(mMarkerProgram, "isPVCOn");
            mMarkerTypeIndex  = glGetUniformLocation(mMarkerProgram, "marker_type");
            mMarkerColIndex   = glGetUniformLocation(mMarkerProgram, "marker_color");
            mMarkerPointIndex = glGetAttribLocation (mMarkerProgram, "point");
            mMarkerColorIndex = glGetAttribLocation (mMarkerProgram, "color");
            mMarkerAlphaIndex = glGetAttribLocation (mMarkerProgram, "alpha");

#define PLOT_CREATE_BUFFERS(type)   \
            mVBO = createBuffer<type>(GL_ARRAY_BUFFER, mVBOSize, NULL, GL_DYNAMIC_DRAW);    \
            mCBO = createBuffer<float>(GL_ARRAY_BUFFER, mCBOSize, NULL, GL_DYNAMIC_DRAW);   \
            mABO = createBuffer<float>(GL_ARRAY_BUFFER, mABOSize, NULL, GL_DYNAMIC_DRAW);   \
            mVBOSize *= sizeof(type);   \
            mCBOSize *= sizeof(float);  \
            mABOSize *= sizeof(float);

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

        ~plot_impl()
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

        void render(const int pWindowId,
                    const int pX, const int pY, const int pVPW, const int pVPH,
                    const glm::mat4& pTransform)
        {
            CheckGL("Begin plot_impl::render");
            glEnable(GL_SCISSOR_TEST);

            glm::mat4 mvp(1.0);
            computeTransformMat(mvp, pTransform, pX, pY, pVPW, pVPH);

            if (mPlotType == fg::FG_LINE) {
                glUseProgram(mPlotProgram);

                if (PLOT_TYPE==fg::FG_3D) {
                    glUniform2fv(mPlotRangeIndex, 3, mRange);
                }
                if (PLOT_TYPE==fg::FG_2D) {
                    glUniform4fv(mPlotUColorIndex, 1, mColor);
                }
                glUniformMatrix4fv(mPlotMatIndex, 1, GL_FALSE, glm::value_ptr(mvp));
                glUniform1i(mPlotPVCOnIndex, mIsPVCOn);

                plot_impl::bindResources(pWindowId);
                glDrawArrays(GL_LINE_STRIP, 0, mNumPoints);
                plot_impl::unbindResources();

                glUseProgram(0);
            }

            if (mMarkerType != fg::FG_NONE) {
                glEnable(GL_PROGRAM_POINT_SIZE);
                glPointSize(10);
                glUseProgram(mMarkerProgram);

                glUniformMatrix4fv(mMarkerMatIndex, 1, GL_FALSE, glm::value_ptr(mvp));
                glUniform1i(mMarkerPVCOnIndex, mIsPVCOn);
                glUniform1i(mMarkerTypeIndex, mMarkerType);
                glUniform4fv(mMarkerColIndex, 1, mColor);

                plot_impl::bindResources(pWindowId);
                glDrawArrays(GL_POINTS, 0, mNumPoints);
                plot_impl::unbindResources();

                glUseProgram(0);
                glDisable(GL_PROGRAM_POINT_SIZE);
                glPointSize(1);
            }

            glDisable(GL_SCISSOR_TEST);
            CheckGL("End plot_impl::render");
        }
};

class _Plot {
    private:
        std::shared_ptr<AbstractRenderable> mPlot;

    public:
        _Plot(const uint pNumPoints, const fg::dtype pDataType,
              const fg::PlotType pPlotType, const fg::MarkerType pMarkerType,
              const fg::ChartType pChartType) {
            if (pChartType == fg::FG_2D) {
                mPlot = std::make_shared< plot_impl<fg::FG_2D> >(pNumPoints, pDataType,
                        pPlotType, pMarkerType);
            } else {
                mPlot = std::make_shared< plot_impl<fg::FG_3D> >(pNumPoints, pDataType,
                        pPlotType, pMarkerType);
            }
        }

        inline const std::shared_ptr<AbstractRenderable>& impl() const {
            return mPlot;
        }

        inline void setColor(const float pRed, const float pGreen,
                             const float pBlue, const float pAlpha) {
            mPlot->setColor(pRed, pGreen, pBlue, pAlpha);
        }

        inline void setLegend(const std::string pLegend) {
            mPlot->setLegend(pLegend);
        }

        inline GLuint vbo() const {
            return mPlot->vbo();
        }

        inline GLuint cbo() const {
            return mPlot->cbo();
        }

        inline GLuint abo() const {
            return mPlot->abo();
        }

        inline size_t vboSize() const {
            return mPlot->vboSize();
        }

        inline size_t cboSize() const {
            return mPlot->cboSize();
        }

        inline size_t aboSize() const {
            return mPlot->aboSize();
        }

        inline void render(const int pWindowId,
                           const int pX, const int pY, const int pVPW, const int pVPH,
                           const glm::mat4& pTransform) const {
            mPlot->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
        }
};

}
