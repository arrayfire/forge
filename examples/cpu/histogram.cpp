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

const unsigned NBINS = 9;

using namespace std;

struct Bitmap {
    unsigned char *ptr;
    unsigned width;
    unsigned height;
};
Bitmap createBitmap(unsigned w, unsigned h);
void destroyBitmap(Bitmap& bmp);
void kernel(Bitmap& bmp);

float perlinNoise(float x, float y, float z);
float octavesPerlin(float x, float y, float z, int octaves, float persistence);
int noise(int x, int y, int width, int height);

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

    fg::Image img(DIMX, DIMY, fg::FG_RGBA, fg::FG_UNSIGNED_BYTE);
    /*
     * Create histogram object while specifying desired number of bins
     */
    fg::Histogram hist(NBINS, fg::FG_UNSIGNED_BYTE);

    /*
     * Set histogram colors and generate image
     */
    hist.setBarColor(fg::FG_YELLOW);

    hist.setAxesLimits(9, 0, 6.f, -1.1f);

    /*
     * generate image, and prepare data to pass into
     * Histogram's underlying vertex buffer object
     */
    kernel(bmp);
    fg::copy(img, bmp.ptr);

    /* copy your data into the vertex buffer object exposed by
     * fg::Histogram class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    unsigned htest[] = {1,5,20,5,60,255,4,9,2};
    fg::copy(hist, htest);

    do {
        kernel(bmp);
        fg::copy(img, bmp.ptr);
        wnd.draw(0, 0, img,  NULL );
        wnd.draw(1, 0, hist, NULL );
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
    static float t=0.1;
    for (unsigned y=0; y<bmp.height; ++y) {
        for (unsigned x=0; x<bmp.width; ++x) {
            int offset  = x + y * bmp.width;
            float juliaVal= octavesPerlin((float)x, (float)y, 0, 4, 0.1);
            bmp.ptr[offset*4 + 0]   = 255 * juliaVal;
            bmp.ptr[offset*4 + 1]   = 255 * juliaVal;
            bmp.ptr[offset*4 + 2]   = 255 * juliaVal;
            bmp.ptr[offset*4 + 3]   = 255;
        }
    }
}

int noise(int x, int y, int width, int height) {
    const float scale = 1.5;
    float jx = scale * (float)(width/2.0f - x)/(width/2.0f);
    float jy = scale * (float)(height/2.0f - y)/(height/2.0f);

    std::complex<float> c(-0.8f, 0.156f);
    std::complex<float> a(jx, jy);

    for (int i=0; i<200; i++) {
        a = a * a + c;
        if (abs(a) > 1000)
            return 0;
    }

    return 1;
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

static const vec3 default_gradients[] = { {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0} };

float perlinNoise(float x, float y, float z){
    x=fmod(x,1.f);
    y=fmod(y,1.f);
    z=fmod(z,1.f);
    float u = interp(x);
    float v = interp(y);
    float w = interp(z);

    unsigned rand_id = rand() % 4;
    float influence_vecs[4];
    influence_vecs[0] = (vec3(x,y,z) - default_gradients[0]) * default_gradients[rand_id];
    influence_vecs[1] = (vec3(x,y,z) - default_gradients[1]) * default_gradients[rand_id];
    influence_vecs[2] = (vec3(x,y,z) - default_gradients[2]) * default_gradients[rand_id];
    influence_vecs[3] = (vec3(x,y,z) - default_gradients[3]) * default_gradients[rand_id];

    return lerp(lerp(influence_vecs[0], influence_vecs[1], u), lerp(influence_vecs[2], influence_vecs[3], u), v);
}

float octavesPerlin(float x, float y, float z, int octaves, float persistence){
    float total=0, max_value=0;
    float amplitude=1, frequency=1;
    for(int i=0; i<octaves; ++i){
        total += perlinNoise( (x+frequency)/DIMX, (y+frequency)/DIMY, z+frequency) * amplitude;
        max_value += amplitude;

        amplitude *= persistence;
        frequency *= 2;
    }
    return total/max_value;
}
