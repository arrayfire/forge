/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/vector_field.h>

#include <handle.hpp>
#include <chart_renderables.hpp>

fg_err fg_create_vector_field(fg_vector_field *pField,
                              const uint pNPoints,
                              const fg_dtype pType,
                              const fg_chart_type pChartType)
{
    try {
        *pField = getHandle(new common::VectorField(pNPoints, (fg::dtype)pType, pChartType));
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_destroy_vector_field(fg_vector_field pField)
{
    try {
        delete getVectorField(pField);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_vector_field_color(fg_vector_field pField,
                                 const float pRed, const float pGreen,
                                 const float pBlue, const float pAlpha)
{
    try {
        getVectorField(pField)->setColor(pRed, pGreen, pBlue, pAlpha);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_set_vector_field_legend(fg_vector_field pField, const char* pLegend)
{
    try {
        getVectorField(pField)->setLegend(pLegend);
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_vbo(uint* pOut, const fg_vector_field pField)
{
    try {
        *pOut = getVectorField(pField)->vbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_cbo(uint* pOut, const fg_vector_field pField)
{
    try {
        *pOut = getVectorField(pField)->cbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_abo(uint* pOut, const fg_vector_field pField)
{
    try {
        *pOut = getVectorField(pField)->abo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_dbo(uint* pOut, const fg_vector_field pField)
{
    try {
        *pOut = getVectorField(pField)->dbo();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_vbo_size(uint* pOut, const fg_vector_field pField)
{
    try {
        *pOut = (uint)getVectorField(pField)->vboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_cbo_size(uint* pOut, const fg_vector_field pField)
{
    try {
        *pOut = (uint)getVectorField(pField)->cboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_abo_size(uint* pOut, const fg_vector_field pField)
{
    try {
        *pOut = (uint)getVectorField(pField)->aboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}

fg_err fg_get_vector_field_dbo_size(uint* pOut, const fg_vector_field pField)
{
    try {
        *pOut = (uint)getVectorField(pField)->dboSize();
    }
    CATCHALL

    return FG_ERR_NONE;
}
