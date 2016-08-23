/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fg/histogram.h>

#include <handle.hpp>
#include <chart_renderables.hpp>

namespace forge
{

Histogram::Histogram(const uint pNBins, const dtype pDataType)
{
    try {
        mValue = getHandle(new common::Histogram(pNBins, pDataType));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Histogram::Histogram(const Histogram& pOther)
{
    try {
        mValue = getHandle(new common::Histogram(pOther.get()));
    } CATCH_INTERNAL_TO_EXTERNAL
}

Histogram::~Histogram()
{
    try {
        delete getHistogram(mValue);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Histogram::setColor(const Color pColor)
{
    try {
        float r = (((int) pColor >> 24 ) & 0xFF ) / 255.f;
        float g = (((int) pColor >> 16 ) & 0xFF ) / 255.f;
        float b = (((int) pColor >> 8  ) & 0xFF ) / 255.f;
        float a = (((int) pColor       ) & 0xFF ) / 255.f;
        getHistogram(mValue)->setColor(r, g, b, a);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Histogram::setColor(const float pRed, const float pGreen,
                         const float pBlue, const float pAlpha)
{
    try {
        getHistogram(mValue)->setColor(pRed, pGreen, pBlue, pAlpha);
    } CATCH_INTERNAL_TO_EXTERNAL
}

void Histogram::setLegend(const char* pLegend)
{
    try {
        getHistogram(mValue)->setLegend(pLegend);
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint Histogram::vertices() const
{
    try {
        return getHistogram(mValue)->vbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint Histogram::colors() const
{
    try {
        return getHistogram(mValue)->cbo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint Histogram::alphas() const
{
    try {
        return getHistogram(mValue)->abo();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint Histogram::verticesSize() const
{
    try {
        return (uint)getHistogram(mValue)->vboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint Histogram::colorsSize() const
{
    try {
        return (uint)getHistogram(mValue)->cboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

uint Histogram::alphasSize() const
{
    try {
        return (uint)getHistogram(mValue)->aboSize();
    } CATCH_INTERNAL_TO_EXTERNAL
}

fg_histogram Histogram::get() const
{
    try {
        return mValue;
    } CATCH_INTERNAL_TO_EXTERNAL
}

}
