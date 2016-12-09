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

#include <backend.hpp>
#include <chart_impl.hpp>

#include <glm/glm.hpp>

#include <memory>

namespace forge
{
namespace common
{

class Chart {
    private:
        forge::ChartType mChartType;
        std::shared_ptr<detail::AbstractChart> mChart;

    public:
        Chart(const forge::ChartType cType)
            : mChartType(cType) {

            ARG_ASSERT(0, cType == FG_CHART_2D || cType == FG_CHART_3D);

            if (cType == FG_CHART_2D) {
                mChart = std::make_shared<detail::chart2d_impl>();
            } else if (cType == FG_CHART_3D) {
                mChart = std::make_shared<detail::chart3d_impl>();
            }
        }

        Chart(const fg_chart pOther) {
            mChart = reinterpret_cast<Chart*>(pOther)->impl();
        }

        inline forge::ChartType chartType() const {
            return mChartType;
        }

        inline const std::shared_ptr<detail::AbstractChart>& impl() const {
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

        inline void setAxesLabelFormat(const char* pXFormat,
                                       const char* pYFormat,
                                       const char* pZFormat) {
            mChart->setAxesLabelFormat(pXFormat, pYFormat, pZFormat);
        }

        inline void getAxesLimits(float* pXmin, float* pXmax,
                                  float* pYmin, float* pYmax,
                                  float* pZmin, float* pZmax) {
            mChart->getAxesLimits(pXmin, pXmax, pYmin, pYmax, pZmin, pZmax);
        }

        inline void setLegendPosition(const unsigned pX, const unsigned pY) {
            mChart->setLegendPosition(pX, pY);
        }

        inline void addRenderable(const std::shared_ptr<detail::AbstractRenderable> pRenderable) {
            mChart->addRenderable(pRenderable);
        }

        inline void render(const int pWindowId,
                           const int pX, const int pY, const int pVPW, const int pVPH,
                           const glm::mat4 &pView, const glm::mat4 &pOrient) const {
            mChart->render(pWindowId, pX, pY, pVPW, pVPH, pView, pOrient);
        }
};

}
}
