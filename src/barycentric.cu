#include <resamplers.h>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <cstring>

/*
 *  LOGGING
 */

#include <stdio.h>
static void default_reporter(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr) {
    (void)usr;
    printf("%s(%d) - %s()\n\t%s\n\t%s\n",file,line,function,msg,expr);
}

typedef void (*reporter_t)(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr);

static void*      reporter_context_=0;
static reporter_t error_  =&default_reporter;
static reporter_t warning_=&default_reporter;
static reporter_t info_   =&default_reporter;

static void useReporters(
    void (*error)  (const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
    void (*warning)(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
    void (*info)   (const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
    void *usr) {
    error_  =error;
    warning_=warning;
    info_   =info;
    reporter_context_=usr;
}

#define ERR(e,msg)  error_(msg,#e,__FILE__,__LINE__,__FUNCTION__,reporter_context_)
#define WARN(e,msg) warning_(msg,#e,__FILE__,__LINE__,__FUNCTION__,reporter_context_)
#define INFO(e,msg) info_(msg,#e,__FILE__,__LINE__,__FUNCTION__,reporter_context_)

#define ASSERT(e) do{if(!(e)) {ERR(e,"Expression evaluated as false."); throw 1; }} while(0)
#define CUTRY(e)  do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {ERR(e,cudaGetErrorString(ecode)); throw 1; }} while(0)

/*
 *  PRELUDE
 */

#define countof(e) (sizeof(e)/sizeof(*(e)))

#define WARPS_PER_BLOCK  4
#define BLOCKSIZE       (32*WARPS_PER_BLOCK) // threads per block

#define restrict __restrict

template<typename t> cudaChannelFormatKind channelformatkind(void);
template<> cudaChannelFormatKind channelformatkind<uint8_t >(void) {return cudaChannelFormatKindUnsigned;}
template<> cudaChannelFormatKind channelformatkind<uint16_t>(void) {return cudaChannelFormatKindUnsigned;}
template<> cudaChannelFormatKind channelformatkind<uint32_t>(void) {return cudaChannelFormatKindUnsigned;}
template<> cudaChannelFormatKind channelformatkind<uint64_t>(void) {return cudaChannelFormatKindUnsigned;}
template<> cudaChannelFormatKind channelformatkind< int8_t >(void) {return cudaChannelFormatKindSigned;}
template<> cudaChannelFormatKind channelformatkind< int16_t>(void) {return cudaChannelFormatKindSigned;}
template<> cudaChannelFormatKind channelformatkind< int32_t>(void) {return cudaChannelFormatKindSigned;}
template<> cudaChannelFormatKind channelformatkind< int64_t>(void) {return cudaChannelFormatKindSigned;}
template<> cudaChannelFormatKind channelformatkind<float   >(void) {return cudaChannelFormatKindFloat;}
template<> cudaChannelFormatKind channelformatkind<double  >(void) {return cudaChannelFormatKindFloat;}

static unsigned prod(const unsigned * const v,unsigned n) {
    unsigned p=1;
    const unsigned *c=v+n;
    while(c-->v) p*=*c;
    return p;
}

/*
 *  INTERFACE
 */

texture<TPixel,cudaTextureType3D,cudaReadModeElementType> in;

struct ctx_t {
    cudaArray *src;
    TPixel    *out;   // cuda device pointer
    unsigned shape[3];
};

static int init(struct resampler* self,
                const unsigned * const shape,     /* output volume pixelation */
                const unsigned ndim) {
    try {
        memset(self,0,sizeof(*self));
        ASSERT(ndim==3);
        ctx_t *c=new ctx_t;
        self->ctx=c;
        memcpy(c->shape,shape,sizeof(c->shape));
        CUTRY(cudaMalloc(&c->out,sizeof(TPixel)*prod(shape,3)));
    } catch(int) {
        return 0;
    }
    return 1;
}

static void release(struct resampler* self) {
    try {
        if(self->ctx) {
            ctx_t *c=(ctx_t*)self->ctx;
            CUTRY(cudaFree(c->out));
            cudaFreeArray(c->src);
            delete c;
            self->ctx=0;
        }
    }
    catch(...) {;}
}

static int upload(struct resampler* self,TPixel * const src,const unsigned * const shape,const unsigned ndim) {
    try { /* CUTRY macro throws ints */
        ctx_t *c=(ctx_t*)self->ctx;
        ASSERT(ndim==3);
      
        in.addressMode[0]=cudaAddressModeClamp;
        in.addressMode[1]=cudaAddressModeClamp;
        in.addressMode[2]=cudaAddressModeClamp;
        in.filterMode=cudaFilterModePoint;
        in.normalized=1;
        { 
            cudaChannelFormatDesc d=cudaCreateChannelDesc(sizeof(TPixel)*8,0,0,0,channelformatkind<TPixel>());
            const struct cudaExtent extent=make_cudaExtent(shape[0],shape[1],shape[2]);
            
            CUTRY(cudaMalloc3DArray(&c->src,&d,extent));

            // copy data to 3D array
            cudaMemcpy3DParms copy={0};
            copy.srcPtr   = make_cudaPitchedPtr(src,shape[0]*sizeof(TPixel),shape[0],shape[1]);
            copy.dstArray = c->src;
            copy.extent   = extent;
            copy.kind     = cudaMemcpyHostToDevice;
            CUTRY(cudaMemcpy3D(&copy));

            CUTRY(cudaBindTextureToArray(&in,c->src,&d));
        }
    } catch(int) { /* CUTRY macro throws ints */
        return 0;
    }
    return 1;
}

static int download(const struct resampler* self, TPixel * const dst) {
    try {
        ctx_t *ctx=(ctx_t*)self->ctx;
        CUTRY(cudaMemcpy(dst,ctx->out,sizeof(TPixel)*prod(ctx->shape,3),cudaMemcpyDeviceToHost));
    } catch(int) {
        return 0;
    }
    return 1;
}

static int resample(struct resampler * const self,
                     const float * const cubeverts) {
    /* upload to texture */
    return 1;
}

static int runTests(void);

extern "C" const struct resampler_api BarycentricGPU = {
    init,
    upload,
    download,
    resample,
    release,
    useReporters,
    runTests
};

/*       */
/* TESTS */
/*       */


/* simpleTransformWithTexture */

namespace simpleTransformWithTexture {

    texture<float,cudaTextureType2D,cudaReadModeElementType> src;

    __global__ void simpleTransformWithTexture_k(float *dst,int w,int h,float th) {
        unsigned int x=__umul24(blockIdx.x,blockDim.x)+threadIdx.x;
        unsigned int y=__umul24(blockIdx.y,blockDim.y)+threadIdx.y;
        float u = (x/(float)w)-0.5f,
              v = (y/(float)h)-0.5f,
             tu = u*cosf(th)-v*sinf(th)+0.5f,
             tv=u*sinf(th)+v*cosf(th)+0.5f;
        {
            unsigned int i=__umul24(y,w)+x;
            dst[i]=tex2D(src,tu,tv);
        }
        
    }

    static int simpleTransformWithTexture(void) {
        const int w=256,h=256;
        try { /* CUTRY macro throws ints */
            cudaArray *a;

            src.addressMode[0]=cudaAddressModeWrap;
            src.addressMode[1]=cudaAddressModeWrap;
            src.filterMode=cudaFilterModeLinear;
            src.normalized=1;
            { 
                cudaChannelFormatDesc d=cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
                CUTRY(cudaMallocArray(&a,&d,w,h));
                CUTRY(cudaBindTextureToArray(src,a,d));

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

} // end namespace simpleTransformWithTexture

/* Test directory */

static int (*tests[])(void)={
    simpleTransformWithTexture::simpleTransformWithTexture
};

static int runTests() {
    int i;
    int nfailed=0;
    for(i=0;i<countof(tests);++i) {
        nfailed+=tests[i]();
    }
    return nfailed;
}
