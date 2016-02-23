/*******************************************************
* Copyright (c) 2015-2019, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#pragma once

#include <common.hpp>
#include <glm/glm.hpp>

#include <map>
#include <memory>
#include <vector>
#include <string>

namespace internal
{

class AbstractChart : public AbstractRenderable {
    protected:
        /* internal class attributes for
        * drawing ticks on axes for plots*/
        std::vector<float> mTickTextX;
        std::vector<float> mTickTextY;
        std::vector<float> mTickTextZ;
        std::vector<std::string> mXText;
        std::vector<std::string> mYText;
        std::vector<std::string> mZText;
        int   mTickCount;  /* should be an odd number always */
        int   mTickSize;
        int   mLeftMargin;
        int   mRightMargin;
        int   mTopMargin;
        int   mBottomMargin;
        /* chart axes ranges and titles */
        float mXMax;
        float mXMin;
        float mYMax;
        float mYMin;
        float mZMax;
        float mZMin;
        std::string mXTitle;
        std::string mYTitle;
        std::string mZTitle;
        /* OpenGL Objects */
        GLuint mDecorVBO;
        GLuint mBorderProgram;
        GLuint mSpriteProgram;
        /* shader uniform variable locations */
        GLuint mBorderAttribPointIndex;
        GLuint mBorderUniformColorIndex;
        GLuint mBorderUniformMatIndex;
        GLuint mSpriteUniformMatIndex;
        GLuint mSpriteUniformTickcolorIndex;
        GLuint mSpriteUniformTickaxisIndex;
        /* Chart legend position*/
        float mLegendX;
        float mLegendY;
        /* VAO map to store a vertex array object
         * for each valid window context */
        std::map<int, GLuint> mVAOMap;
        /* list of renderables to be displayed on the chart*/
        std::vector< std::shared_ptr<AbstractRenderable> > mRenderables;

        /* rendering helper functions */
        void renderTickLabels(const int pWindowId, const uint pW, const uint pH,
                              const std::vector<std::string> &pTexts,
                              const glm::mat4 &pTransformation, const int pCoordsOffset,
                              const bool pUseZoffset=true) const;

        /* virtual functions that has to be implemented by
         * dervied class: chart2d_impl, chart3d_impl */
        virtual void bindResources(const int pWindowId) = 0;
        virtual void unbindResources() const = 0;
        virtual void pushTicktextCoords(const float pX, const float pY, const float pZ=0.0) = 0;
        virtual void generateChartData() = 0;
        virtual void generateTickLabels() = 0;

    public:
        AbstractChart(const int pLeftMargin, const int pRightMargin,
                      const int pTopMargin, const int pBottomMargin);
        virtual ~AbstractChart();

        void setAxesTitles(const char* pXTitle,
                           const char* pYTitle,
                           const char* pZTitle);

        void setAxesLimits(const float pXmin, const float pXmax,
                           const float pYmin, const float pYmax,
                           const float pZmin, const float pZmax);

        void setLegendPosition(const float pX, const float pY);

        float xmax() const;
        float xmin() const;
        float ymax() const;
        float ymin() const;
        float zmax() const;
        float zmin() const;

        void addRenderable(const std::shared_ptr<AbstractRenderable> pRenderable);
};

class chart2d_impl : public AbstractChart {
    private:
        /* rendering helper functions that are derived
         * from AbstractRenderable base class
         * */
        void bindResources(const int pWindowId);
        void unbindResources() const;
        void pushTicktextCoords(const float x, const float y, const float z=0.0);
        void generateChartData();
        void generateTickLabels();

    public:
        chart2d_impl();

        virtual ~chart2d_impl() {}

        void render(const int pWindowId,
                    const int pX, const int pY, const int pVPW, const int pVPH,
                    const glm::mat4& pTransform);
};

class chart3d_impl : public AbstractChart {
    private:
        /* rendering helper functions that are derived
         * from AbstractRenderable base class
         * */
        void bindResources(const int pWindowId);
        void unbindResources() const;
        void pushTicktextCoords(const float x, const float y, const float z=0.0);
        void generateChartData();
        void generateTickLabels();

    public:
        chart3d_impl();

        virtual ~chart3d_impl() {}

        void render(const int pWindowId,
                    const int pX, const int pY, const int pVPW, const int pVPH,
                    const glm::mat4& pTransform);
};

class _Chart {
    private:
        std::shared_ptr<AbstractChart> mChart;

    public:
        _Chart(const fg::ChartType cType) {
            if (cType == fg::FG_2D) {
                mChart = std::make_shared<chart2d_impl>();
            } else if (cType == fg::FG_3D) {
                mChart = std::make_shared<chart3d_impl>();
            } else {
                throw fg::ArgumentError("_Chart::_Chart",
                                        __LINE__, 0,
                                        "Invalid chart type");
            }
        }

        inline const std::shared_ptr<AbstractChart>& impl() const {
            return mChart;
        }

        inline void setAxesTitles(const char* pX,
                                  const char* pY,
                                  const char* pZ) {
            mChart->setAxesTitles(pX, pY, pZ);
        }

        inline void setAxesLimits(const float pXmin, const float pXmax,
                                  const float pYmin, const float pYmax,
                                  const float pZmin, const float pZmax) {
            mChart->setAxesLimits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax);
        }

        inline void setLegendPosition(const uint pX, const uint pY) {
            mChart->setLegendPosition(pX, pY);
        }

        inline void addRenderable(const std::shared_ptr<AbstractRenderable> pRenderable) {
            mChart->addRenderable(pRenderable);
        }

        inline void render(const int pWindowId,
                           const int pX, const int pY, const int pVPW, const int pVPH,
                           const glm::mat4 &pTransform) const {
            mChart->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
        }
};

}
