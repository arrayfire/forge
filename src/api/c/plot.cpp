/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/chart_renderables.hpp>
#include <common/handle.hpp>
#include <fg/plot.h>

using namespace forge;

using forge::common::getPlot;

fg_err fg_create_plot(fg_plot* pPlot, const unsigned pNPoints,
                      const fg_dtype pType, const fg_chart_type pChartType,
                      const fg_plot_type pPlotType,
                      const fg_marker_type pMarkerType) {
    try {
        ARG_ASSERT(1, (pNPoints > 0));

        *pPlot = getHandle(new common::Plot(
            pNPoints, (forge::dtype)pType, pPlotType, pMarkerType, pChartType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_plot(fg_plot* pOut, fg_plot pIn) {
    try {
        ARG_ASSERT(1, (pIn != 0));

        common::Plot* temp = new common::Plot(getPlot(pIn));
        *pOut              = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_plot(fg_plot pPlot) {
    try {
        ARG_ASSERT(0, (pPlot != 0));

        delete getPlot(pPlot);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_plot_color(fg_plot pPlot, const float pRed, const float pGreen,
                         const float pBlue, const float pAlpha) {
    try {
        ARG_ASSERT(0, (pPlot != 0));

        getPlot(pPlot)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_plot_legend(fg_plot pPlot, const char* pLegend) {
    try {
        ARG_ASSERT(0, (pPlot != 0));
        ARG_ASSERT(1, (pLegend != 0));

        getPlot(pPlot)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_plot_marker_size(fg_plot pPlot, const float pMarkerSize) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        getPlot(pPlot)->setMarkerSize(pMarkerSize);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_vertex_buffer(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = getPlot(pPlot)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_color_buffer(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = getPlot(pPlot)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_alpha_buffer(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = getPlot(pPlot)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_radii_buffer(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = getPlot(pPlot)->mbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_vertex_buffer_size(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = (unsigned)getPlot(pPlot)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_color_buffer_size(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = (unsigned)getPlot(pPlot)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_alpha_buffer_size(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = (unsigned)getPlot(pPlot)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_radii_buffer_size(unsigned* pOut, const fg_plot pPlot) {
    try {
        ARG_ASSERT(1, (pPlot != 0));

        *pOut = (unsigned)getPlot(pPlot)->mboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}
