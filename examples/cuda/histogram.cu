/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <forge.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuComplex.h>
#include <CUDACopy.hpp>
#include <cstdio>

const unsigned IMGW = 256;
const unsigned IMGH = 256;
const unsigned DIMX = 1000;
const unsigned DIMY = 800;
const unsigned IMG_SIZE = IMGW * IMGH * 4;
const unsigned WIN_ROWS = 1;
const unsigned WIN_COLS = 2;
const unsigned NBINS = 256;
const float PERSISTENCE = 0.5f;

void genBaseNoise(float* baseNoise);
void genPerlinNoise(unsigned char* perlinNoise, float* baseNoise);
void genHistogram(unsigned char* perlinNoise, int* histOut, float* histColors);

curandState_t* state;

__global__
void setupRandomKernel(curandState *states, unsigned long long seed)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tid, 0, &states[tid]);
}

int main(void)
{
    float* baseNoise;
    unsigned char* perlinNoise;

    int *histOut;
    float *histColors;

    FORGE_CUDA_CHECK(cudaMalloc((void **)&state, NBINS*sizeof(curandState_t)));
    setupRandomKernel <<< 1, NBINS >>> (state, 314567);

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "Histogram Demo");
    wnd.makeCurrent();

    /*
     * Split the window into grid regions
     */
    wnd.grid(WIN_ROWS, WIN_COLS);

    fg::Image img(IMGW, IMGH, FG_RGBA, u8);

    fg::Chart chart(FG_2D);
    /* set x axis limits to maximum and minimum values of data
     * and y axis limits to range [0, number of pixels ideally]
     * but practically total number of pixels as y range will skew
     * the histogram graph vertically. Therefore setting it to
     * 25% of total number of pixels */
    chart.setAxesLimits(0, 1, 0, IMGW*IMGH/(float)(NBINS/4.0));

    /*
     * Create histogram object specifying number of bins
     */
    fg::Histogram hist = chart.histogram(NBINS, u8);
    /*
     * Set histogram colors
     */
    hist.setColor(FG_YELLOW);

    // IMG_SIZE/4 pixels of each 4 bytes(float)
    FORGE_CUDA_CHECK(cudaMalloc((void**)&baseNoise, IMG_SIZE));
    // IMG_SIZE/4 pixels of each 1 bytes(unsigned char)
    FORGE_CUDA_CHECK(cudaMalloc((void**)&perlinNoise, IMG_SIZE));
    FORGE_CUDA_CHECK(cudaMalloc((void**)&histOut, NBINS * sizeof(int)));
    FORGE_CUDA_CHECK(cudaMalloc((void**)&histColors, 3*NBINS * sizeof(float)));

    genBaseNoise(baseNoise);
    genPerlinNoise(perlinNoise, baseNoise);
    genHistogram(perlinNoise, histOut, histColors);

    fg::copy(img, perlinNoise);
    fg::copy(hist.vertices(), histOut);
    fg::copy(hist.colors(), histColors);

    do {
        wnd.draw(0, 0, img,  "Dynamic Perlin Noise" );
        wnd.draw(1, 0, chart, "Histogram of Noisy Image");
        wnd.swapBuffers();

        // limit histogram update frequency
        //if(fmod(persistance, 0.5f) < 0.01) {
            genBaseNoise(baseNoise);
            genPerlinNoise(perlinNoise, baseNoise);
            genHistogram(perlinNoise, histOut, histColors);

            fg::copy(img, perlinNoise);
            fg::copy(hist.vertices(), histOut);
            fg::copy(hist.colors(), histColors);
        //}
    } while(!wnd.close());

    FORGE_CUDA_CHECK(cudaFree(perlinNoise));
    FORGE_CUDA_CHECK(cudaFree(histOut));
    FORGE_CUDA_CHECK(cudaFree(histColors));
    return 0;
}

inline
int divup(int a, int b)
{
    return (a+b-1)/b;
}

__global__
void baseNoiseKernel(float* baseNoise, curandState* state)
{
    int x = blockIdx.x * blockDim.x  + threadIdx.x;
    int y = blockIdx.y * blockDim.y  + threadIdx.y;

    if (x<IMGW && y<IMGH) {
        baseNoise[y*IMGW+x] = curand_uniform(&state[0]);
    }
}

void genBaseNoise(float* baseNoise)
{
    static const dim3 threads(8, 8);
    dim3 blocks(divup(IMGW, threads.x),
                divup(IMGH, threads.y));

    baseNoiseKernel<<< blocks, threads >>>(baseNoise, state);
}

__device__ __inline__
float interp(float x0, float x1, float t)
{
    return x0 + (x1 - x0) * t;
}

__global__
void perlinKernel(unsigned char* perlinNoise, float* baseNoise)
{
    int i = blockIdx.x * blockDim.x  + threadIdx.x;
    int j = blockIdx.y * blockDim.y  + threadIdx.y;

    if (i<IMGW && j<IMGH) {
        float pnoise = 0.0f;
        float amp    = 1.0f;
        float tamp   = 0.0f;

        for (int octave=6; octave>=0; --octave)
        {
            int period = 1 << octave;
            float freq = 1.0f / period;

            int si0 = (i/period) * period;
            int si1 = (si0 + period) % IMGW;
            float hblend = (i - si0) * freq;

            int sj0 = (j/period) * period;
            int sj1 = (sj0 + period) % IMGH;
            float vblend = (j - sj0) * freq;

            float top = interp(baseNoise[si0+IMGW*sj0], baseNoise[si1+IMGW*sj0], hblend);
            float bot = interp(baseNoise[si0+IMGW*sj1], baseNoise[si1+IMGW*sj1], hblend);

            pnoise += (amp * interp(top, bot, vblend));

            tamp += amp;
            amp *= PERSISTENCE;
        }

        uint offset = i+j*IMGW;
        perlinNoise[4*offset+0] = pnoise/tamp;
        perlinNoise[4*offset+1] = pnoise/tamp;
        perlinNoise[4*offset+2] = pnoise/tamp;
        perlinNoise[4*offset+3] = 255;
    }
}

void genPerlinNoise(unsigned char* perlinNoise, float* baseNoise)
{
    static const dim3 threads(8, 8);
    dim3 blocks(divup(IMGW, threads.x),
                divup(IMGH, threads.y));

    perlinKernel<<< blocks, threads >>>(perlinNoise, baseNoise);
}

__global__
void histogramKernel(const unsigned char* perlinNoise, int* histOut, const unsigned nbins)
{
    int x = blockIdx.x * blockDim.x  + threadIdx.x;
    int y = blockIdx.y * blockDim.y  + threadIdx.y;

    if (x<IMGW && y<IMGH) {
        int offset  = y * IMGW + x;
        unsigned char noiseVal = perlinNoise[offset*4 + 0];
        offset = __float2int_rd(nbins * (__int2float_rd(noiseVal)/255.f));
        atomicAdd(histOut + offset , 1);
    }
}

__global__
void histColorsKernel(float* histColors, curandState *states)
{
    int bin = blockIdx.x * blockDim.x  + threadIdx.x;

    histColors[3*bin+0] = curand_uniform(&states[bin]);
    histColors[3*bin+1] = curand_uniform(&states[bin]);
    histColors[3*bin+2] = curand_uniform(&states[bin]);
}

void genHistogram(unsigned char * perlinNoise, int* histOut, float* histColors)
{
    static const dim3 threads(8, 8);
    dim3 blocks(divup(IMGW, threads.x),
                divup(IMGH, threads.y));

    cudaMemset(histOut, 0, NBINS * sizeof(int));

    histogramKernel<<< blocks, threads >>>(perlinNoise, histOut, NBINS);

    histColorsKernel<<< 1, NBINS >>>(histColors, state);
}
