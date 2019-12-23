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
#include <fg/vector_field.h>

using namespace forge;

using forge::common::getVectorField;

fg_err fg_create_vector_field(fg_vector_field* pField, const unsigned pNPoints,
                              const fg_dtype pType,
                              const fg_chart_type pChartType) {
    try {
        ARG_ASSERT(1, (pNPoints > 0));

        *pField = getHandle(
            new common::VectorField(pNPoints, (forge::dtype)pType, pChartType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_retain_vector_field(fg_vector_field* pOut, fg_vector_field pIn) {
    try {
        ARG_ASSERT(1, (pIn != 0));

        common::VectorField* temp =
            new common::VectorField(getVectorField(pIn));
        *pOut = getHandle(temp);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_release_vector_field(fg_vector_field pField) {
    try {
        ARG_ASSERT(0, (pField != 0));

        delete getVectorField(pField);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_vector_field_color(fg_vector_field pField, const float pRed,
                                 const float pGreen, const float pBlue,
                                 const float pAlpha) {
    try {
        ARG_ASSERT(0, (pField != 0));

        getVectorField(pField)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_vector_field_legend(fg_vector_field pField, const char* pLegend) {
    try {
        ARG_ASSERT(0, (pField != 0));
        ARG_ASSERT(1, (pLegend != 0));

        getVectorField(pField)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_vertex_buffer(unsigned* pOut,
                                         const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = getVectorField(pField)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_color_buffer(unsigned* pOut,
                                        const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = getVectorField(pField)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_alpha_buffer(unsigned* pOut,
                                        const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = getVectorField(pField)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_direction_buffer(unsigned* pOut,
                                            const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = getVectorField(pField)->dbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_vertex_buffer_size(unsigned* pOut,
                                              const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = (unsigned)getVectorField(pField)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_color_buffer_size(unsigned* pOut,
                                             const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = (unsigned)getVectorField(pField)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_alpha_buffer_size(unsigned* pOut,
                                             const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = (unsigned)getVectorField(pField)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_direction_buffer_size(unsigned* pOut,
                                                 const fg_vector_field pField) {
    try {
        ARG_ASSERT(1, (pField != 0));

        *pOut = (unsigned)getVectorField(pField)->dboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}
