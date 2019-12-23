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
#include <fg/histogram.h>

using namespace forge;
using forge::common::getHistogram;

fg_err fg_create_histogram(fg_histogram* pHistogram, const unsigned pNBins,
                           const fg_dtype pType) {
    try {
        ARG_ASSERT(1, (pNBins > 0));

        *pHistogram =
            getHandle(new common::Histogram(pNBins, (forge::dtype)pType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_histogram(fg_histogram* pOut, fg_histogram pIn) {
    try {
        ARG_ASSERT(1, (pIn != 0));

        common::Histogram* temp = new common::Histogram(getHistogram(pIn));
        *pOut                   = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_histogram(fg_histogram pHistogram) {
    try {
        ARG_ASSERT(0, (pHistogram != 0));

        delete getHistogram(pHistogram);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_histogram_color(fg_histogram pHistogram, const float pRed,
                              const float pGreen, const float pBlue,
                              const float pAlpha) {
    try {
        ARG_ASSERT(0, (pHistogram != 0));

        getHistogram(pHistogram)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_histogram_legend(fg_histogram pHistogram, const char* pLegend) {
    try {
        ARG_ASSERT(0, (pHistogram != 0));
        ARG_ASSERT(1, (pLegend != 0));

        getHistogram(pHistogram)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_vertex_buffer(unsigned* pOut,
                                      const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = getHistogram(pHistogram)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_color_buffer(unsigned* pOut,
                                     const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = getHistogram(pHistogram)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_alpha_buffer(unsigned* pOut,
                                     const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = getHistogram(pHistogram)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_vertex_buffer_size(unsigned* pOut,
                                           const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = (unsigned)getHistogram(pHistogram)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_color_buffer_size(unsigned* pOut,
                                          const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = (unsigned)getHistogram(pHistogram)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_histogram_alpha_buffer_size(unsigned* pOut,
                                          const fg_histogram pHistogram) {
    try {
        ARG_ASSERT(1, (pHistogram != 0));

        *pOut = (unsigned)getHistogram(pHistogram)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}
