// Tiff Output

// I'm going to assemble the file in a resizable buffer and then output that to a file.

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

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
    struct rational{ uint32_t num,den; } rat={v,1};
    struct bigtiff_tag tag={TIFFTAG_XRESOLUTION,TIFFTYPE_RATIONAL,1,*(uint64_t*)&rat};
    return tag;
}

static struct bigtiff_tag yresolution(size_t v) {
    struct rational{ uint32_t num,den; } rat={v,1};
    struct bigtiff_tag tag={TIFFTAG_YRESOLUTION,TIFFTYPE_RATIONAL,1,*(uint64_t*)&rat};
    return tag;
}

static struct bigtiff_tag resolutionunit_inch() {
    struct bigtiff_tag tag={TIFFTAG_RESOLUTIONUNIT,TIFFTYPE_SHORT,1,2};
    return tag;
}

#define CHK(e) do{if(!(e)) goto Error;}while(0)

void write_tiff_u8(const char* filename,size_t ndim,const size_t* shape,uint8_t *data) {
    const size_t w=shape[0],
                 h=shape[1],
                 nifd=(ndim==2)?1:prod(ndim-2,shape+2),
                 nbytes=prod(ndim,shape),
                 bytesof_ifds=nifd*sizeof(struct bigtiff_ifd);
    struct buf_t buf={0};
    CHK(resize(&buf,nbytes+bytesof_ifds+sizeof(struct bigtiff_header)));

    //write_header
    {
        const struct bigtiff_header hdr={0x4949,0x002B,8,0,sizeof(hdr)};
        CHK(bufwrite(&buf,0,sizeof(hdr),&hdr));
    }
    // write_ifds
    {
        size_t i;
        const size_t start=sizeof(struct bigtiff_header);
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
            if(i+1==nifd)
                ifd.next=0;
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
    free(buf.data);
    return;
Error:
    printf("Couldn't write.  Something went wrong.\n");
}
