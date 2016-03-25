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
#include <chart.hpp>

#include <glm/glm.hpp>

#include <memory>

namespace common
{

class Chart {
    private:
        fg::ChartType mChartType;
        std::shared_ptr<detail::AbstractChart> mChart;

    public:
        Chart(const fg::ChartType cType)
            : mChartType(cType) {
            if (cType == FG_CHART_2D) {
                mChart = std::make_shared<detail::chart2d_impl>();
            } else if (cType == FG_CHART_3D) {
                mChart = std::make_shared<detail::chart3d_impl>();
            } else {
                throw fg::ArgumentError("Chart::Chart",
                                        __LINE__, 0,
                                        "Invalid chart type");
            }
        }

        Chart(const fg_chart pOther) {
            mChart = reinterpret_cast<Chart*>(pOther)->impl();
        }

        inline fg::ChartType chartType() const {
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

        inline void setLegendPosition(const uint pX, const uint pY) {
            mChart->setLegendPosition(pX, pY);
        }

        inline void addRenderable(const std::shared_ptr<detail::AbstractRenderable> pRenderable) {
            mChart->addRenderable(pRenderable);
        }

        inline void render(const int pWindowId,
                           const int pX, const int pY, const int pVPW, const int pVPH,
                           const glm::mat4 &pTransform) const {
            mChart->render(pWindowId, pX, pY, pVPW, pVPH, pTransform);
        }
};

}
