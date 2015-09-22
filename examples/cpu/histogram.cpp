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
    for (unsigned y=0; y<bmp.height; ++y) {
        for (unsigned x=0; x<bmp.width; ++x) {
            int offset  = x + y * bmp.width;
            float juliaVal= perlinNoise((float)x/DIMX, (float)y/DIMY, 0);
            bmp.ptr[offset*4 + 0]   = 255 * juliaVal;
            bmp.ptr[offset*4 + 1]   = 128 * juliaVal;
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
static const int permutation[] = { 151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
        190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
        88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
        129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
        49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};
static const vec3 default_gradients[] = { {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0} };

float perlinNoise(float x, float y, float z){
    x=fmod(x,1.f);
    y=fmod(y,1.f);
    z=fmod(z,1.f);
    vec3 influence_vecs[4];
    influence_vecs[0] = (vec3(x,y,z) - default_gradients[0]) ;
    influence_vecs[1] = (vec3(x,y,z) - default_gradients[1]) ;
    influence_vecs[2] = (vec3(x,y,z) - default_gradients[2]) ;
    influence_vecs[3] = (vec3(x,y,z) - default_gradients[3]) ;
//default_gradients[0]
//default_gradients[1]
//default_gradients[2]    float final_influence = 
//default_gradients[3]

    //for(int i=0; i<4; ++i)
        //cout<<influence_vecs[i].x<<','<<influence_vecs[i].y<<endl;

    return x;
}
