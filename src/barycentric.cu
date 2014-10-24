#include <resamplers.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {printf("%s(%d): %s()\n\tExpression evaluated as failure.\n\t%s\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); throw 1; }} while(0)
#include <stdio.h>
#include <stdlib.h>
#define countof(e) (sizeof(e)/sizeof(*(e)))

#define WARPS_PER_BLOCK  4
#define BLOCKSIZE       (32*WARPS_PER_BLOCK) // threads per block

#define restrict __restrict

#if 0
template <class T> struct vol_t {
    T * const restrict data;
    unsigned shape[3];
    unsigned strides[4];
};

struct tetrahedron {
    float T[9];
    float ori[3];
};

template <class T>
__global__
void
__launch_bounds__(BLOCKSIZE,1)
barycentric_kernel(vol_t<T> dst,cudateTextureObject_t src, const struct tetrads[5]) {
    
}

#endif

/* INTERFACE */

static void resample(TPixel * const restrict dst,const unsigned * const restrict dst_shape,const unsigned * const restrict dst_strides,
                     TPixel * const restrict src,const unsigned * const restrict src_shape,const unsigned * const restrict src_strides,
                     const float * restrict cubeverts) {
}

static int runTests(void);

extern "C" const struct resampler_api BarycentricGPU = {
    resample,
    runTests
};

/*       */
/* TESTS */
/*       */

#define ASSERT(e)  do{if(!(e)) {printf("%s(%d): %s()(\n\tExpression evaluated as false.\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,#e); return 1; }}while(0)

/* simpleTransformWithTexture */

texture<float,cudaTextureType2D,cudaReadModeElementType> src;

__global__ void simpleTransformWithTexture_k(float *dst,int w,int h,float th) {
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y+threadIdx.y;
    float u = (x/(float)w)-0.5f,
          v = (y/(float)h)-0.5f,
         tu = u*cosf(th)-v*sinf(th)+0.5f,
         tv = u*sinf(th)+v*cosf(th)+0.5f;
    dst[y*w+x]=tex2D(src,tu,tv);
}

static int simpleTransformWithTexture(void) {
    const int w=256,h=256;
    try {
        cudaArray *a;

        src.addressMode[0]=cudaAddressModeWrap;
        src.addressMode[1]=cudaAddressModeWrap;
        src.filterMode=cudaFilterModeLinear;
        src.normalized=1;
        { 
            cudaChannelFormatDesc d=cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
            CUTRY(cudaMallocArray(&a,&d,w,h));
            cudaBindTextureToArray(src,a,d);

        }
        // --> should copy source data in at this point (cudaMemcpyToArray) <-- 

        float *out;
        CUTRY(cudaMalloc(&out,w*h*sizeof(float)));

        dim3 block(16,16),
             grid((w+block.x-1)/block.x,
                  (h+block.y-1)/block.y);
        simpleTransformWithTexture_k<<<grid,block>>>(out,w,h,15*3.14159f/180.0f);

        // clean up
        cudaFree(out);
        cudaFreeArray(a);

    } catch(...) {
        return 1;
    }
    return 0;
}

/* Test directory */

static int (*tests[])(void)={
    simpleTransformWithTexture
};

static int runTests() {
    int i;
    int nfailed=0;
    for(i=0;i<countof(tests);++i) {
        nfailed+=tests[i]();
    }
    return nfailed;
}
