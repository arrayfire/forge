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

VectorField::VectorField(const unsigned pNumPoints, const dtype pDataType, const ChartType pChartType)
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
    delete getVectorField(mValue);
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

unsigned VectorField::vertices() const
{
    try {
        return getVectorField(mValue)->vbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned VectorField::colors() const
{
    try {
        return getVectorField(mValue)->cbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned VectorField::alphas() const
{
    try {
        return getVectorField(mValue)->abo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned VectorField::directions() const
{
    try {
        return getVectorField(mValue)->dbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned VectorField::verticesSize() const
{
    try {
        return (unsigned)getVectorField(mValue)->vboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned VectorField::colorsSize() const
{
    try {
        return (unsigned)getVectorField(mValue)->cboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned VectorField::alphasSize() const
{
    try {
        return (unsigned)getVectorField(mValue)->aboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

unsigned VectorField::directionsSize() const
{
    try {
        return (unsigned)getVectorField(mValue)->dboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

fg_vector_field VectorField::get() const
{
    try {
        return mValue;
    } CATCH_INTERNAL_TO_EXTERNAL
}

}
