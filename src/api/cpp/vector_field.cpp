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

namespace forge
{

VectorField::VectorField(const uint pNumPoints, const dtype pDataType, const ChartType pChartType)
{
    try {
        mValue = getHandle(new common::VectorField(pNumPoints, pDataType, pChartType));
    } CATCH_INTERNAL_TO_EXTERNAL
}

VectorField::VectorField(const VectorField& pOther)
{
    try {
        mValue = getHandle(new common::VectorField(pOther.get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

VectorField::~VectorField()
{
    try {
        delete getVectorField(mValue);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void VectorField::setColor(const Color pColor)
{
    try {
        float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
        float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
        float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
        float a = (((int) pColor       ) & 0xFF ) / 255.f;
        getVectorField(mValue)->setColor(r, g, b, a);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void VectorField::setColor(const float pRed, const float pGreen,
                           const float pBlue, const float pAlpha)
{
    try {
        getVectorField(mValue)->setColor(pRed, pGreen, pBlue, pAlpha);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void VectorField::setLegend(const char* pLegend)
{
    try {
        getVectorField(mValue)->setLegend(pLegend);
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint VectorField::vertices() const
{
    try {
        return getVectorField(mValue)->vbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint VectorField::colors() const
{
    try {
        return getVectorField(mValue)->cbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint VectorField::alphas() const
{
    try {
        return getVectorField(mValue)->abo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint VectorField::directions() const
{
    try {
        return getVectorField(mValue)->dbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint VectorField::verticesSize() const
{
    try {
        return (uint)getVectorField(mValue)->vboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint VectorField::colorsSize() const
{
    try {
        return (uint)getVectorField(mValue)->cboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint VectorField::alphasSize() const
{
    try {
        return (uint)getVectorField(mValue)->aboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint VectorField::directionsSize() const
{
    try {
        return (uint)getVectorField(mValue)->dboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

fg_vector_field VectorField::get() const
{
    try {
        return mValue;
    } CATCH_INTERNAL_TO_EXTERNAL
}

}
