#include <resamplers.h>
#include <stdio.h>
#include <stdlib.h>  
#include <nd.h>
#include <tictoc.h>

#define ASSERT(e) do{if(!(e)) {printf("%s(%d): %s()(\n\tExpression evaluated as false.\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,#e); exit(1); }}while(0)
#define countof(e) (sizeof(e)/sizeof(*(e)))

static unsigned eq(const TPixel * const a, const TPixel * const b,unsigned n) {
    unsigned i;
    for(i=0;i<n;++i) {
        if(a[i]!=b[i])
            return 0;
    }
    return 1;
}

#define NX (256)
#define NY (256)
#define NZ (256)


const unsigned src_shape []={NX,NY,NZ};
const unsigned src_stride[]={1,NX,NX*NY};
const unsigned dst_shape []={NX,NY,NZ};
const unsigned dst_stride[]={1,NX,NX*NY};

#define s (0.25f)
#define cos (0.866f)
#define sin (0.5f)

#define px(x,y) (cos*(x-0.5f)-sin*(y-0.5f)+0.5)
#define py(x,y) (sin*(x-0.5f)+cos*(y-0.5f)+0.5)

const float cube[]={
    NX*s*px(0,0),NY*s*py(0,0),0,
    NX*s*px(1,0),NY*s*py(1,0),0,
    NX*s*px(0,1),NY*s*py(0,1),0,
    NX*s*px(1,1),NY*s*py(1,1),0,
     0, 0,NZ,
    NX, 0,NZ,
     0,NY,NZ,
    NX,NY,NZ,
};
#undef s
#undef sin
#undef cos
#undef px
#undef py

int main(int argc,char* argv[]) {

    TPixel *src,*dst;

    ASSERT(src=(TPixel*)malloc(sizeof(TPixel)*NX*NY*NZ));
    ASSERT(dst=(TPixel*)calloc(NX*NY*NZ,sizeof(TPixel)));


    {

        unsigned x,y,z,i=0;
        for(z=0;z<NZ;++z) for(y=0;y<NY;++y) for(x=0;x<NX;++x,++i) {
            src[i]=(x)^(y)^(z);
        }

    }

    {
        TicTocTimer t=tic();
        Barycentric.resample(dst,dst_shape,dst_stride,
                             src,src_shape,src_stride,
                             cube);
        printf("TIME %fs\n",toc(&t));
    }
    

#if 1
    ndioAddPluginPath("plugins");
    {   
        const size_t shape_sz[]={NX,NY,NZ};
        nd_t v=ndref(ndreshape(ndcast(ndinit(),nd_u8),3,shape_sz),src,nd_static);
        ndioClose(ndioWrite(ndioOpen("src.mp4",NULL,"w"),v));
    }
    {
        const size_t shape_sz[]={NX,NY,NZ};
        nd_t v=ndref(ndreshape(ndcast(ndinit(),nd_u8),3,shape_sz),dst,nd_static);
        ndioClose(ndioWrite(ndioOpen("dst.mp4",NULL,"w"),v));
    }
#endif

    return 0;
}