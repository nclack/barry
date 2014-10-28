#include <resamplers.h>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <matrix.h>

#include <cstring>

#define EPS (1e-5f)
#define BIT(k,i) (((k)>>(i))&1)

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
    TPixel    *out;    // cuda device pointer
    unsigned shape[3]; // output shape
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

struct tetrahedron {
    float T[9];
    float ori[3];
};

static void tetrahedron(struct tetrahedron *self, const float * const v, const unsigned * idx) {
    const float T[9] = {
        v[3*idx[0]  ]-v[3*idx[3]  ],v[3*idx[1]  ]-v[3*idx[3]  ],v[3*idx[2]  ]-v[3*idx[3]  ],
        v[3*idx[0]+1]-v[3*idx[3]+1],v[3*idx[1]+1]-v[3*idx[3]+1],v[3*idx[2]+1]-v[3*idx[3]+1],
        v[3*idx[0]+2]-v[3*idx[3]+2],v[3*idx[1]+2]-v[3*idx[3]+2],v[3*idx[2]+2]-v[3*idx[3]+2],
    };
    Matrixf.inv33(self->T,T); 
    memcpy(self->ori,v+3*idx[3],sizeof(self->ori));
};

/* KERNEL */

inline __device__ unsigned prod(dim3 a)            {return a.x*a.y*a.z;}
inline __device__ unsigned stride(uint3 a, dim3 b) {return a.x+b.x*(a.y+b.y*a.z);}
inline __device__ unsigned sum(uint3 a)            {return a.x+a.y+a.z;}

inline __device__ 
void map(const struct tetrahedron * const restrict self,
         float * restrict dst,
         const float * const restrict src) {
    float tmp[3];
    memcpy(tmp,src,sizeof(float)*3);
    {
        const float * const o = self->ori;
        tmp[0]-=o[0];
        tmp[1]-=o[1];
        tmp[2]-=o[2];
    }
    Matrixf.mul(dst,self->T,3,3,tmp,1);    
    dst[3]=1.0f-dst[0]-dst[1]-dst[2];
}

inline __device__
unsigned find_best_tetrad(const float * const restrict ls) {
    float v=ls[0];
    unsigned i,argmin=0;
    for(i=1;i<4;++i) {
        if(ls[i]<v) {
            v=ls[i];
            argmin=i;
        }
    }
    if(v>=0.0f)
        return 0;
    return argmin+1;
}

inline __device__
unsigned any_less_than_zero(const float * const restrict v,const unsigned n) {
    const float *c=v+n;
    //while(c-->v) if(*c<EPS) return 1; // use this to show off the edges of the middle tetrad
    while(c-->v) if(*c<-EPS) return 1;
    return 0;
}

inline __device__ 
void idx2coord(float * restrict r,unsigned idx,const unsigned * const restrict shape) {
    r[0]=idx%shape[0];  idx/=shape[0];
    r[1]=idx%shape[1];  idx/=shape[1];
    r[2]=idx%shape[2];
}

__global__ void resample_k() {
    unsigned idst = sum(threadIdx)+stride(blockIdx,gridDim)*prod(blockDim);

            float r[3],lambdas[4];
            unsigned itetrad;
            idx2coord(r,idst,dst_shape);
            map(tetrads,lambdas,r);             // Map center tetrahedron
            
            itetrad=find_best_tetrad(lambdas);
            if(itetrad>0) {
                map(tetrads+itetrad,lambdas,r);   // Map best tetrahedron
            }
            
            if(any_less_than_zero(lambdas,4)) // other boundary
                continue;

            // Map source index
            {
                unsigned idim,ilambda,isrc=0;
                for(idim=0;idim<3;++idim) {
                    float s=0.0f;
                    const float d=(float)(src_shape[idim]);
                    for(ilambda=0;ilambda<4;++ilambda) {
                        const float      w=lambdas[ilambda];
                        const unsigned idx=indexes[itetrad][ilambda];
                        s+=w*BIT(idx,idim);
                    }
                    s*=d;
                    s=(s<0.0f)?0.0f:(s>(d-1))?(d-1):s;
                    isrc+=src_strides[idim]*((unsigned)s); // important to floor here.  can't change order of sums
                }
                dst[idst]=src[isrc];
            }
}


static int resample(struct resampler * const self,
                     const float * const cubeverts) {

    struct tetrahedron tetrads[5];
    for(unsigned i=0;i<5;i++)
        tetrahedron(tetrads+i,cubeverts,indexes[i]);

    try {
        unsigned r,
                 blocks=(unsigned)ceil(dst.nelem/float(BLOCKSIZE)),
                 tpb   =BLOCKSIZE;  
        const unsigned b=blocks;
        struct cudaDeviceProp prop;
        dim3 grid,
             threads=make_uint3(tpb,1,1);

        CUTRY(cudaGetDeviceProperties(&prop,0));
        INFO("MAX GRID: %7d %7d %7d"ENDL,prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
        // Pack our 1d indexes into cuda's 3d indexes
        ASSERT(grid.x=nextdim(blocks,prop.maxGridSize[0],&r));
        blocks/=grid.x;
        blocks+=r;
        ASSERT(grid.y=nextdim(blocks,prop.maxGridSize[1],&r));
        blocks/=grid.y;
        blocks+=r;
        ASSERT(grid.z=blocks);
        INFO("    GRID: %7d %7d %7d"ENDL,grid.x,grid.y,grid.z);

        INFO("blocks:%u threads/block:%u\n",b,tpb);
        resample_k<TSRC,TDST><<<grid,threads>>>(*self,tetrads);

        CUTRY(cudaGetLastError());
    } catch(int) {
        return 1;
    }
    return 0;
}

/*           */
/* INTERFACE */
/*           */

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

static int test_testrahedron(void) {
    const unsigned idx[]={1,4,5,7};
    float v[8*3]={0};
    int i;
    struct tetrahedron ans;
    struct tetrahedron expected={
            {18.0f,19.0f,20.0f}
    };
    for(i=0;i<8;++i) {
        v[3*i+0]=(float)BIT(i,0);
        v[3*i+1]=(float)BIT(i,1);
        v[3*i+2]=(float)BIT(i,2);
    }
    tetrahedron(&ans,v,idx);
    ASSERT(eq(ans.ori,v+3*7,3));
    return 0;
}

static int (*tests[])(void)={
    simpleTransformWithTexture::simpleTransformWithTexture,
    test_testrahedron,
};

static int runTests() {
    int i;
    int nfailed=0;
    for(i=0;i<countof(tests);++i) {
        nfailed+=tests[i]();
    }
    return nfailed;
}
