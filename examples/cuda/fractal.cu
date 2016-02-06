#include <forge.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <CUDACopy.hpp>
#include <cstdio>

const unsigned DIMX = 512;
const unsigned DIMY = 512;
const size_t   SIZE = DIMX*DIMY*4;

void kernel(unsigned char* dev_out);

int main(void)
{
    unsigned char *dev_out;

    /*
     * First Forge call should be a window creation call
     * so that necessary OpenGL context is created for any
     * other fg::* object to be created successfully
     */
    fg::Window wnd(DIMX, DIMY, "Fractal Demo");
    wnd.makeCurrent();
    /* create an font object and load necessary font
     * and later pass it on to window object so that
     * it can be used for rendering text */
    fg::Font fnt;
#ifdef OS_WIN
    fnt.loadSystemFont("Calibri");
#else
    fnt.loadSystemFont("Vera");
#endif
    wnd.setFont(&fnt);

    /* Create an image object which creates the necessary
     * textures and pixel buffer objects to hold the image
     * */
    fg::Image img(DIMX, DIMY, fg::FG_RGBA, fg::u8);
    /* copy your data into the pixel buffer object exposed by
     * fg::Image class and then proceed to rendering.
     * To help the users with copying the data from compute
     * memory to display memory, Forge provides copy headers
     * along with the library to help with this task
     */
    CUDA_ERROR_CHECK(cudaMalloc((void**)&dev_out, SIZE));
    kernel(dev_out);
    fg::copy(img, dev_out);

    do {
        wnd.draw(img);
    } while(!wnd.close());

    CUDA_ERROR_CHECK(cudaFree(dev_out));
    return 0;
}

__device__
int julia(int x, int y)
{

    const float scale = 1.5;
    float jx = scale * (float)(DIMX/2.0f - x)/(DIMX/2.0f);
    float jy = scale * (float)(DIMY/2.0f - y)/(DIMY/2.0f);

    cuFloatComplex c = make_cuFloatComplex(-0.8f, 0.156f);
    cuFloatComplex a = make_cuFloatComplex(jx, jy);

    for (int i=0; i<200; i++) {
        a = cuCaddf(cuCmulf(a, a), c);
        if (cuCabsf(a) > 1000.0f)
            return 0;
    }

    return 1;
}

__global__
void julia(unsigned char* out)
{
    int x = blockIdx.x * blockDim.x  + threadIdx.x;
    int y = blockIdx.y * blockDim.y  + threadIdx.y;

    if (x<DIMX && y<DIMY) {
        int offset = x + y * DIMX;
        // now calculate the value at that position
        int juliaValue = julia(x, y);

        out[offset*4 + 2] = 255 * juliaValue;
        out[offset*4 + 0] = 0;
        out[offset*4 + 1] = 0;
        out[offset*4 + 3] = 255;
    }
}

inline int divup(int a, int b)
{
    return (a+b-1)/b;
}

void kernel(unsigned char* dev_out)
{
    static const dim3 threads(8, 8);
    dim3 blocks(divup(DIMX, threads.x),
                divup(DIMY, threads.y));

    julia<<< blocks, threads >>>(dev_out);
}
