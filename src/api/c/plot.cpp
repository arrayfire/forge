/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/plot.h>

#include <handle.hpp>
#include <chart_renderables.hpp>

using namespace forge;

fg_err fg_create_plot(fg_plot *pPlot,
                      const unsigned pNPoints, const fg_dtype pType,
                      const fg_chart_type pChartType,
                      const fg_plot_type pPlotType,
                      const fg_marker_type pMarkerType)
{
    try {
        *pPlot = getHandle(new common::Plot(pNPoints, (forge::dtype)pType, pPlotType,
                                            pMarkerType, pChartType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_destroy_plot(fg_plot pPlot)
{
    try {
        delete getPlot(pPlot);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_plot_color(fg_plot pPlot,
                         const float pRed, const float pGreen,
                         const float pBlue, const float pAlpha)
{
    try {
        getPlot(pPlot)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_plot_legend(fg_plot pPlot, const char* pLegend)
{
    try {
        getPlot(pPlot)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_plot_marker_size(fg_plot pPlot, const float pMarkerSize)
{
    try {
        getPlot(pPlot)->setMarkerSize(pMarkerSize);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_vertex_buffer(unsigned* pOut, const fg_plot pPlot)
{
    try {
        *pOut = getPlot(pPlot)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_color_buffer(unsigned* pOut, const fg_plot pPlot)
{
    try {
        *pOut = getPlot(pPlot)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_alpha_buffer(unsigned* pOut, const fg_plot pPlot)
{
    try {
        *pOut = getPlot(pPlot)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_radii_buffer(unsigned* pOut, const fg_plot pPlot)
{
    try {
        *pOut = getPlot(pPlot)->mbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_vertex_buffer_size(unsigned* pOut, const fg_plot pPlot)
{
    try {
        *pOut = (unsigned)getPlot(pPlot)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_color_buffer_size(unsigned* pOut, const fg_plot pPlot)
{
    try {
        *pOut = (unsigned)getPlot(pPlot)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_alpha_buffer_size(unsigned* pOut, const fg_plot pPlot)
{
    try {
        *pOut = (unsigned)getPlot(pPlot)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_plot_radii_buffer_size(unsigned* pOut, const fg_plot pPlot)
{
    try {
        *pOut = (unsigned)getPlot(pPlot)->mboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}
