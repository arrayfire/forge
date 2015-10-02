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
#include <cuComplex.h>
#include <CUDACopy.hpp>
#include <cstdio>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;
const unsigned IMG_SIZE = DIMX * DIMY * 4;
const unsigned WIN_ROWS = 1;
const unsigned WIN_COLS = 2;

const unsigned NBINS = 5;


const static int hperm[] = {26, 58, 229, 82, 132, 72, 144, 251, 196, 192, 127, 16,
    68, 118, 104, 213, 91, 105, 203, 61, 59, 93, 136, 249, 27, 137, 141, 223, 119,
    193, 155, 43, 71, 244, 170, 115, 201, 150, 165, 78, 208, 53, 90, 232, 209, 83,
    45, 174, 140, 178, 220, 184, 70, 6, 202, 17, 128, 212, 117, 200, 254, 57, 248,
    62, 164, 172, 19, 177, 241, 103, 48, 38, 210, 129, 23, 211, 8, 112, 107,  126,
    252,  198, 32, 123, 111,  176,  206, 15, 219, 221, 147, 245, 67, 92, 108, 143,
    54, 102, 169, 22, 74, 124, 181, 186, 138, 18, 7, 34, 81, 46, 120, 236, 89,228,
    197, 205, 13, 63, 134,  242, 157, 135, 237, 35, 234, 49, 85, 76, 148, 188, 98,
    87, 173, 84, 226, 11, 125, 122, 2, 94, 191, 179, 175, 187, 133, 231, 154,  44,
    28, 110, 247, 121, 146, 240, 97, 88, 130,195, 30, 25, 56, 171, 80, 69, 139, 9,
    238, 160, 227, 204, 31, 40, 66, 77, 21, 159,  162, 207,  167, 214, 10, 3, 149,
    194, 239, 166,  145, 235, 20, 50, 113, 189, 99, 37, 86, 42, 168, 114, 96, 246,
    183, 250, 233, 156, 52,  65, 131, 47,  255, 5, 33, 217, 73, 4, 60, 64, 109, 0,
    215, 100, 180, 12, 24, 190, 222, 106, 41, 216, 230, 161, 55, 152, 79, 75, 142,
    36, 101, 1, 253, 225, 51, 224, 182, 116, 218, 95, 39, 158,  14, 243, 151, 163,
    29, 153, 199, 185
};

__constant__ int perm[256];
void kernel(unsigned char* dev_out);
void kernel_hist(unsigned char * src, int* hist_out);

int main(void)
{
    int *hist_out;
    unsigned char *dev_out;
    cudaMemcpyToSymbol(perm, hperm, 256 * sizeof(int));

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "Histogram Demo");
    wnd.makeCurrent();
    /* create an font object and load necessary font
     * and later pass it on to window object so that
     * it can be used for rendering text */
    fg::Font fnt;
#ifdef OS_WIN
    fnt.loadSystemFont("Calibri", 32);
#else
    fnt.loadSystemFont("Vera", 32);
#endif
    wnd.setFont(&fnt);

    /*
     * Split the window into grid regions
     */
    wnd.grid(WIN_ROWS, WIN_COLS);

    fg::Image img(DIMX, DIMY, fg::FG_RGBA, fg::FG_UNSIGNED_BYTE);
    /*
     * Create histogram object while specifying desired number of bins
     */
    fg::Histogram hist(NBINS, fg::FG_UNSIGNED_BYTE);

    /*
     * Set histogram colors
     */
    hist.setBarColor(fg::FG_YELLOW);

    /* set x axis limits to maximum and minimum values of data
     * and y axis limits to range [0, nBins]*/
    hist.setAxesLimits(1, 0, 1000, 0);
    CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_out, IMG_SIZE ));
    CUDA_ERROR_CHECK(cudaMalloc((void**)&hist_out, NBINS));
    kernel(dev_out);
    kernel_hist(dev_out, hist_out);
    fg::copy(img, dev_out);
    fg::copy(hist, hist_out);

    do {
        kernel(dev_out);
        kernel_hist(dev_out, hist_out);
        fg::copy(img, dev_out);
        fg::copy(hist, hist_out);
        wnd.draw(0, 0, img,  NULL );
        wnd.draw(1, 0, hist, NULL );
        wnd.draw();
    } while(!wnd.close());

    CUDA_ERROR_CHECK(cudaFree(dev_out));
    return 0;
}

__device__
inline float interp(float t){
    return ((6 * t - 15) * t + 10) * t * t * t;
}

__device__
inline float lerp (float x0, float x1, float t) {
        return x0 + (x1 - x0) * t;
}

__device__
inline float dot (float2 v0, float2 v1) {
        return v0.x*v1.x + v0.y*v1.y;
}

__device__
inline float2 sub (float2 v0, float2 v1) {
        return make_float2(v0.x-v1.x, v0.y-v1.y);
}

__device__
float perlinNoise(float x, float y, int tileSize) {
    const float2 default_gradients[] = { make_float2(1,1), make_float2(-1,1),make_float2 (1,-1), make_float2(-1,-1) };
    int x_grid = x/tileSize;
    int y_grid = y/tileSize;
    unsigned rand_id0 = perm[(x_grid+2*y_grid) % 256 ] % 4;
    unsigned rand_id1 = perm[(x_grid+1+2*y_grid) % 256 ] % 4;
    unsigned rand_id2 = perm[(x_grid+2*(y_grid+1)) % 256 ] % 4;
    unsigned rand_id3 = perm[(x_grid+1+2*(y_grid+1)) % 256 ] % 4;

    x=fmod(x,__int2float_rd(tileSize))/tileSize;
    y=fmod(y,__int2float_rd(tileSize))/tileSize;
    float u = interp(x);
    float v = interp(y);

    float influence_vecs[4];
    influence_vecs[0] = dot(sub(make_float2(x,y), make_float2(0,0)), default_gradients[rand_id0]);
    influence_vecs[1] = dot(sub(make_float2(x,y), make_float2(1,0)), default_gradients[rand_id1]);
    influence_vecs[2] = dot(sub(make_float2(x,y), make_float2(0,1)), default_gradients[rand_id2]);
    influence_vecs[3] = dot(sub(make_float2(x,y), make_float2(1,1)),             default_gradients[rand_id3]);

    return lerp(lerp(influence_vecs[0], influence_vecs[1], u), lerp(influence_vecs[2], influence_vecs[3], u), v);

}
 __device__
float octavesPerlin(float x, float y, int octaves, float persistence, int tileSize) {
    float total = 0, max_value = 0;
    float amplitude = 1, frequency = 1;
    for(int i=0; i<octaves; ++i){
        total += perlinNoise( x*frequency, y*frequency, tileSize) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= 2;
    }
    return total/max_value;
}

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

__global__
void image_gen(unsigned char* out, float persistance, int tileSize) {
    int x = blockIdx.x * blockDim.x  + threadIdx.x;
    int y = blockIdx.y * blockDim.y  + threadIdx.y;

    if (x<DIMX && y<DIMY) {
        int offset  = y * DIMX + x;
        int octaves = 4;
        int noiseValue    = 255 * octavesPerlin(x, y, octaves, persistance, tileSize);
        out[offset*4 + 0] = noiseValue;
        out[offset*4 + 1] = noiseValue;
        out[offset*4 + 2] = noiseValue;
        out[offset*4 + 3] = 255;
    }
}

void kernel(unsigned char* dev_out) {
    static const dim3 threads(8, 8);
    dim3 blocks(divup(DIMX, threads.x),
                divup(DIMY, threads.y));

    static int tileSize = 32; tileSize++;
    static float persistance = 0.1; persistance+=0.01;
    image_gen<<< blocks, threads >>>(dev_out, persistance, tileSize);
}

__global__
void hist_freq(const unsigned char* src, int* hist_array, const unsigned nbins) {
    int x = blockIdx.x * blockDim.x  + threadIdx.x;
    int y = blockIdx.y * blockDim.y  + threadIdx.y;

    if (x<DIMX && y<DIMY) {
        int offset  = y * DIMX + x;
        unsigned char noiseVal = src[offset*4 + 0];
        atomicAdd(hist_array + __float2int_rd(nbins * __int2float_rd(noiseVal)/255.f), 1);
    }
}

void kernel_hist(unsigned char * src, int* hist_out){
    static const dim3 threads(NBINS);
    dim3 blocks(1);
    cudaMemset(hist_out, 0, NBINS * sizeof(int));
    hist_freq<<< blocks, threads >>>(src, hist_out, NBINS);
}
