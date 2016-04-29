#include <resamplers.h>
#include <stdio.h>
#include <stdlib.h>
#include <tictoc.h>
#include <string.h>

#define ASSERT(e)  do{if(!(e)) {printf("%s(%d): %s()(\n\tExpression evaluated as false.\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,#e); exit(1); }}while(0)
#define TIME(e)    do{TicTocTimer t=tic(); {e;} printf("TIME %10fs\t%s\n",toc(&t),#e);} while(0)
#define countof(e) (sizeof(e)/sizeof(*(e)))

static unsigned eq(const TPixel * const a, const TPixel * const b,unsigned n) {
    unsigned i;
    for(i=0;i<n;++i) {
        if(a[i]!=b[i])
            return 0;
    }
    return 1;
}

#define NX (512)
#define NY (512)
#define NZ (512)


const unsigned src_shape []={NX,NY,NZ};
const unsigned src_stride[]={1,NX,NX*NY};
const unsigned dst_shape []={NX,NY,NZ};
const unsigned dst_stride[]={1,NX,NX*NY};


#define s (0.5f)

#if 1
    // 30 degrees
    #define sn (0.5f)
    #define cs (0.866f)
#elif 0
    // 5 degrees
    #define sn (0.0872f)
    #define cs (0.9962)
#elif 1
    // 0 degrees
    #define sn (0.0f)
    #define cs (1.0f)
#endif

#define tx(x,y) (( cs*((x)-0.5f)-sn*((y)-0.5f) )*s+0.5f)
#define ty(x,y) (( sn*((x)-0.5f)+cs*((y)-0.5f) )*s+0.5f)

const float cube[]={
    NX*tx(0,0),NY*ty(0,0),0.0*NZ,
    NX*tx(1,0),NY*ty(1,0),0.0*NZ,
    NX*tx(0,1),NY*ty(0,1),0.0*NZ,
    NX*tx(1,1),NY*ty(1,1),0.0*NZ,
     0, 0,NZ,
    NX, 0,NZ,
     0,NY,NZ,
    NX,NY,NZ,
};

#undef s

#undef cs
#undef sn

#undef tx
#undef ty

void write_tiff_u8(const char* filename,size_t ndim,const size_t* shape,uint8_t *buf);

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
        struct resampler r;
        TIME(ASSERT( BarycentricGPU.init  (&r,src_shape,dst_shape,3)));
        TIME(ASSERT( BarycentricGPU.source(&r,src)));
        TIME(ASSERT( BarycentricGPU.destination(&r,dst)));
        TIME(ASSERT( BarycentricGPU.resample(&r,cube)));
        TIME(ASSERT( BarycentricGPU.result(&r,dst)));
                     BarycentricGPU.release(&r);
    }

    {
        const size_t shape_sz[]={NX,NY,NZ};
        write_tiff_u8("src.tif",3,shape_sz,src);
    }
    {
        const size_t shape_sz[]={NX,NY,NZ};
        write_tiff_u8("dst.tif",3,shape_sz,dst);
    }

    return 0;
}

// Tiff Output

// I'm going to assemble the file in a resizable buffer and then output that to a file.

#include <stdlib.h>

struct buf_t {
    char *data;
    size_t end,cap;
};

static int resize(struct buf_t * b, size_t request) {
    b->cap=request;
    b->data=realloc(b->data,b->cap);
    return b->data!=NULL;
}

static int bufwrite(struct buf_t * __restrict b,size_t offset,size_t nbytes, void* __restrict data) {
    size_t last=offset+nbytes;
    if(last>b->cap)
        if(!resize(b,last))
            return 0;
    b->end=(last>b->end)?last:b->end;
    memcpy(b->data+offset,data,nbytes);
    return 1;
}

static size_t prod(int n,size_t *v) {
    size_t a=*v++,*e=v+n-1;
    while(v<e) a*=*v++;
    return a;
}

/// Keep interesting tiff tags here.
enum TiffTag {
    TIFFTAG_IMAGEWIDTH                =256,   ///< The number of columns in the image, i.e., the number of pixels per scanline.
    TIFFTAG_IMAGELENGTH               =257,   ///< The number of rows (sometimes described as scanlines) in the image
    TIFFTAG_BITSPERSAMPLE             =258,
    TIFFTAG_COMPRESSION               =259,
    TIFFTAG_PHOTOMETRICINTERPRETATION =262,
    TIFFTAG_STRIPOFFSETS              =273,
    TIFFTAG_ROWSPERSTRIP              =278,
    TIFFTAG_STRIPBYTECOUNTS           =279,
    TIFFTAG_XRESOLUTION               =282,
    TIFFTAG_YRESOLUTION               =283,
    TIFFTAG_RESOLUTIONUNIT            =296,
};
#define NTAGS (11)

enum TiffType {
    TIFFTYPE_BYTE=1,
    TIFFTYPE_ASCII,
    TIFFTYPE_SHORT,
    TIFFTYPE_LONG,
    TIFFTYPE_RATIONAL,
    TIFFTYPE_SBYTE,
    TIFFTYPE_UNDEFINED,
    TIFFTYPE_SSHORT,
    TIFFTYPE_SLONG,
    TIFFTYPE_SRATIONAL,
    TIFFTYPE_FLOAT,
    TIFFTYPE_DOUBLE,
    TIFFTYPE_LONG8=16,
    TIFFTYPE_SLONG8,
    TIFFTYPE_IFD8,
};

#pragma pack(push,1)
struct bigtiff_header {
    uint16_t fmt;
    uint16_t ver;
    uint16_t sizeof_offset;
    uint16_t zero;
    uint64_t first_ifd;
};
struct bigtiff_tag {
    uint16_t tag,type;
    uint64_t count,value;
};
struct bigtiff_ifd {
    uint64_t ntags;
    struct bigtiff_tag tags[NTAGS];
    uint64_t next;
};
#pragma pop

static struct bigtiff_tag imagewidth(size_t w) {
    struct bigtiff_tag tag={TIFFTAG_IMAGEWIDTH,TIFFTYPE_LONG,1,w};
    return tag;
}

static struct bigtiff_tag imagelength(size_t h) {
    struct bigtiff_tag tag={TIFFTAG_IMAGELENGTH,TIFFTYPE_LONG,1,h};
    return tag;
}

static struct bigtiff_tag bitspersample(unsigned b) {
    struct bigtiff_tag tag={TIFFTAG_BITSPERSAMPLE,TIFFTYPE_SHORT,1,b};
    return tag;
}

static struct bigtiff_tag uncompressed() {
    struct bigtiff_tag tag={TIFFTAG_COMPRESSION,TIFFTYPE_SHORT,1,0};
    return tag;
}

static struct bigtiff_tag photometricinterpretation(unsigned v) {
    struct bigtiff_tag tag={TIFFTAG_PHOTOMETRICINTERPRETATION,TIFFTYPE_SHORT,1,1};
    return tag;
}

static struct bigtiff_tag stripoffsets(size_t o) {
    struct bigtiff_tag tag={TIFFTAG_STRIPOFFSETS,TIFFTYPE_LONG8,1,o};
    return tag;
}

static struct bigtiff_tag rowsperstrip(size_t v) {
    struct bigtiff_tag tag={TIFFTAG_ROWSPERSTRIP,TIFFTYPE_LONG,1,v};
    return tag;
}

static struct bigtiff_tag stripbytecounts(size_t v) {
    struct bigtiff_tag tag={TIFFTAG_STRIPBYTECOUNTS,TIFFTYPE_LONG8,1,v};
    return tag;
}

static struct bigtiff_tag xresolution(size_t v) {
    struct bigtiff_tag tag={TIFFTAG_XRESOLUTION,TIFFTYPE_SHORT,1,v};
    return tag;
}

static struct bigtiff_tag yresolution(size_t v) {
    struct bigtiff_tag tag={TIFFTAG_YRESOLUTION,TIFFTYPE_SHORT,1,v};
    return tag;
}

static struct bigtiff_tag resolutionunit_inch() {
    struct bigtiff_tag tag={TIFFTAG_RESOLUTIONUNIT,TIFFTYPE_SHORT,1,2};
    return tag;
}

void breakme() {
    printf("ruh roh\n");
}
#define CHK(e) do{if(!(e)) {breakme(); goto Error;}}while(0)

static void write_tiff_u8(const char* filename,size_t ndim,const size_t* shape,uint8_t *data) {
    const size_t w=shape[0],
                 h=shape[1],
                 nifd=(ndim==2)?1:prod(ndim-2,shape+2),
                 nbytes=prod(ndim,shape);
    struct buf_t buf={0};
    CHK(resize(&buf,nbytes+1024));

    //write_header
    {
        const struct bigtiff_header hdr={0x4D4D,0x002B,8,0,sizeof(hdr)};
        CHK(bufwrite(&buf,0,sizeof(hdr),&hdr));
    }
    // write_ifds
    {
        size_t i;
        const size_t start=sizeof(struct bigtiff_header);
        const size_t bytesof_ifds=nifd*sizeof(struct bigtiff_ifd);
        const size_t end=start+bytesof_ifds;
        for(i=0;i<nifd;++i) {
            struct bigtiff_ifd ifd={NTAGS,{0},start+(i+1)*sizeof(ifd)};
            int j=0;
            ifd.tags[j++]=imagewidth(w);
            ifd.tags[j++]=imagelength(h);
            ifd.tags[j++]=bitspersample(8);
            ifd.tags[j++]=uncompressed();
            ifd.tags[j++]=photometricinterpretation(0);
            ifd.tags[j++]=stripoffsets(end+w*h*i);
            ifd.tags[j++]=rowsperstrip(h);
            ifd.tags[j++]=stripbytecounts(w*h);
            ifd.tags[j++]=xresolution(72);
            ifd.tags[j++]=yresolution(72);
            ifd.tags[j++]=resolutionunit_inch();
            CHK(j==NTAGS);
            bufwrite(&buf,start+i*sizeof(ifd),sizeof(ifd),&ifd);
        }
        // write the data
        bufwrite(&buf,end,nbytes,data);
    }
    {
        FILE *fp=fopen(filename,"wb");
        CHK(fp);
        fwrite(buf.data,1,buf.end,fp);
        fclose(fp);
    }
    return;
Error:
    printf("Couldn't write.  Something went wrong.\n");
}
