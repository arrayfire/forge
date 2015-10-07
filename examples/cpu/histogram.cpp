/*******************************************************
 * Copyright (c) 2015-2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <forge.h>
#include <CPUCopy.hpp>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>

const unsigned DIMX = 1000;
const unsigned DIMY = 800;
const unsigned WIN_ROWS = 1;
const unsigned WIN_COLS = 2;

const unsigned NBINS = 5;

static float t=0.1;

using namespace std;

struct Bitmap {
    unsigned char *ptr;
    unsigned width;
    unsigned height;
};
Bitmap createBitmap(unsigned w, unsigned h);
void destroyBitmap(Bitmap& bmp);
void kernel(Bitmap& bmp);
void hist_freq(Bitmap& bmp, int *hist_array, const unsigned nbins);

float perlinNoise(float x, float y, float z, int tileSize);
float octavesPerlin(float x, float y, float z, int octaves, float persistence, int tileSize);

int main(void) {

    Bitmap bmp = createBitmap(DIMX, DIMY);

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

    fg::Image img(DIMX, DIMY, fg::FG_RGBA, fg::u8);
    /*
     * Create histogram object while specifying desired number of bins
     */
    fg::Histogram hist(NBINS, fg::u8);

    /*
     * Set histogram colors
     */
    hist.setBarColor(fg::FG_YELLOW);

    /*
     * generate image, and prepare data to pass into
     * Histogram's underlying vertex buffer object
     */
    kernel(bmp);
    fg::copy(img, bmp.ptr);

    /* set x axis limits to maximum and minimum values of data
     * and y axis limits to range [0, nBins]*/
    hist.setAxesLimits(1, 0, 1000, 0);

    /* copy your data into the vertex buffer object exposed by
     * fg::Histogram class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    int histogram_array[NBINS] = {0};
    hist_freq(bmp, &histogram_array[0], NBINS);
    fg::copy(hist, histogram_array);

    do {
        kernel(bmp);
        fg::copy(img, bmp.ptr);

        int histogram_array[NBINS] = {0};
        hist_freq(bmp, &histogram_array[0], NBINS);
        // limit histogram update frequency
        if(fmod(t,0.4f) < 0.02f)
            fg::copy(hist, histogram_array);

        wnd.draw(0, 0, img,  "Dynamic Perlin Noise" );
        wnd.draw(1, 0, hist, "Histogram of Noisy Image");
        // draw window and poll for events last
        wnd.draw();
    } while(!wnd.close());

    return 0;
}


Bitmap createBitmap(unsigned w, unsigned h)
{
    Bitmap retVal;
    retVal.width = w;
    retVal.height= h;
    retVal.ptr   = new unsigned char[4*w*h];
    return retVal;
}

void destroyBitmap(Bitmap& bmp)
{
    delete[] bmp.ptr;
}

void kernel(Bitmap& bmp) {
    static unsigned tileSize=100;
    for (unsigned y=0; y<bmp.height; ++y) {
        for (unsigned x=0; x<bmp.width; ++x) {
            int offset  = x + y * bmp.width;
            unsigned char noiseVal = 255 * octavesPerlin((float)x, (float)y, 0, 4, t, tileSize);
            bmp.ptr[offset*4 + 0]   = noiseVal;
            bmp.ptr[offset*4 + 1]   = noiseVal;
            bmp.ptr[offset*4 + 2]   = noiseVal;
            bmp.ptr[offset*4 + 3]   = 255;
        }
    }
    t+=0.02;
    tileSize++;
}

void hist_freq(Bitmap& bmp, int *hist_array, const unsigned nbins){
    for (unsigned y=0; y<bmp.height; ++y) {
        for (unsigned x=0; x<bmp.width; ++x) {
            int offset  = x + y * bmp.width;
            unsigned char noiseVal = bmp.ptr[offset*4];
            hist_array[(int)((float)noiseVal/255.f * nbins)]++;
        }
    }

}

struct vec3{
    float x;
    float y;
    float z;
    vec3(){x=0;y=0;z=0;}
    vec3(float _x, float _y, float _z){x=_x;y=_y;z=_z;}
    float operator*(vec3 rhs){ return x*rhs.x + y*rhs.y + z*rhs.z;}
    vec3 operator-(vec3 rhs){ return vec3(x-rhs.x, y-rhs.y, z-rhs.z);}
};

float interp(float t){
    return ((6 * t - 15) * t + 10) * t * t * t;
}
const inline float lerp (float x0, float x1, float t) {
        return x0 + (x1 - x0) * t;
}

static const int perm[] = { 26, 58, 229, 82, 132, 72, 144, 251, 196, 192, 127, 16,
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

static const vec3 default_gradients[] = { {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0} };

float perlinNoise(float x, float y, float z, int tileSize){
    int x_grid = (int)x/tileSize;
    int y_grid = (int)y/tileSize;
    unsigned rand_id0 = perm[(x_grid+2*y_grid) % 256 ] % 4;
    unsigned rand_id1 = perm[(x_grid+1+2*y_grid) % 256 ] % 4;
    unsigned rand_id2 = perm[(x_grid+2*(y_grid+1)) % 256 ] % 4;
    unsigned rand_id3 = perm[(x_grid+1+2*(y_grid+1)) % 256 ] % 4;

    x=fmod(x,tileSize)/tileSize;
    y=fmod(y,tileSize)/tileSize;
    z=fmod(z,tileSize)/tileSize;
    float u = interp(x);
    float v = interp(y);
    float w = interp(z);

    float influence_vecs[4];
    influence_vecs[0] = (vec3(x,y,z) - vec3(0,0,0)) * default_gradients[rand_id0];
    influence_vecs[1] = (vec3(x,y,z) - vec3(1,0,0)) * default_gradients[rand_id1];
    influence_vecs[2] = (vec3(x,y,z) - vec3(0,1,0)) * default_gradients[rand_id2];
    influence_vecs[3] = (vec3(x,y,z) - vec3(1,1,0)) * default_gradients[rand_id3];

    return lerp(lerp(influence_vecs[0], influence_vecs[1], u), lerp(influence_vecs[2], influence_vecs[3], u), v);
}

float octavesPerlin(float x, float y, float z, int octaves, float persistence, int tileSize){
    float total = 0, max_value = 0;
    float amplitude = 1, frequency = 1;
    for(int i=0; i<octaves; ++i){
        total += perlinNoise( (x*frequency), (y*frequency), z*frequency, tileSize) * amplitude;
        max_value += amplitude;

        amplitude *= persistence;
        frequency *= 2;
    }
    return total/max_value;
}
