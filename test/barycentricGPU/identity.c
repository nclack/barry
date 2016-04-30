#include <resamplers.h>
#include <stdio.h>
#include <stdlib.h>
#include <tiff.write.h>
#include <tictoc.h>

#define ASSERT(e) do{if(!(e)) {printf("%s(%d): %s()(\n\tExpression evaluated as false.\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,#e); exit(2); }}while(0)
#define TIME(e) do{TicTocTimer t=tic(); {e;} printf("TIME %10fs\t%s\n",toc(&t),#e);} while(0)
#define countof(e) (sizeof(e)/sizeof(*(e)))

static unsigned eq(const TPixel * const a, const TPixel * const b,unsigned n) {
    unsigned i;
    for(i=0;i<n;++i) {
        if(a[i]!=b[i])
            return 0;
    }
    return 1;
}

#define NX (64)
#define NY (64)
#define NZ (64)

TPixel src[64*64*64];
TPixel dst[64*64*64];

const unsigned src_shape[]={64,64,64};
const unsigned src_stride[]={1,64,64*64};
const unsigned dst_shape[]={64,64,64};
const unsigned dst_stride[]={1,64,64*64};

const float cube[]={
     0, 0, 0,
    64, 0, 0,
     0,64, 0,
    64,64, 0,
     0, 0,64,
    64, 0,64,
     0,64,64,
    64,64,64,
};

int main(int argc,char* argv[]) {

    {
        unsigned x,y,z,i=0;
        for(z=0;z<64;++z) for(y=0;y<64;++y) for(x=0;x<64;++x,++i) {
            src[i]=(x/3)^(y/3)^(z/3);
        }
    }

    {
        struct resampler r;
        TIME(ASSERT( BarycentricGPU.init  (&r,src_shape,dst_shape,3)));
        TIME(ASSERT( BarycentricGPU.source(&r,src)));
        TIME(ASSERT( BarycentricGPU.destination(&r,dst)));
        TIME(ASSERT( BarycentricGPU.resample(&r,cube)));
        TIME(ASSERT( BarycentricGPU.result(&r,dst)));
                     BarycentricGPU.release(&r);
    }

#if 1
    {
        const size_t shape_sz[]={NX,NY,NZ};
        write_tiff_u8("src.tif",3,shape_sz,src);
    }
    {
        const size_t shape_sz[]={NX,NY,NZ};
        write_tiff_u8("dst.tif",3,shape_sz,dst);
    }
#endif

    ASSERT(eq(src,dst,countof(src)));
    return 0;
}
