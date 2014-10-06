#include <resamplers.h>
#include <matrix.h>
#include <stdlib.h>
#include <string.h>

struct tetrahedron {
    float T[9];
    float ori[3];
};

#define restrict __restrict
#define EPS (1e-5f)
#define BIT(k,i) (((k)>>(i))&1)

/* UTILITIES */

static void tetrahedron(struct tetrahedron *self, const float * const v, const unsigned * idx) {
    const float T[9] = {
        v[3*idx[0]  ]-v[3*idx[3]  ],v[3*idx[1]  ]-v[3*idx[3]  ],v[3*idx[2]  ]-v[3*idx[3]  ],
        v[3*idx[0]+1]-v[3*idx[3]+1],v[3*idx[1]+1]-v[3*idx[3]+1],v[3*idx[2]+1]-v[3*idx[3]+1],
        v[3*idx[0]+2]-v[3*idx[3]+2],v[3*idx[1]+2]-v[3*idx[3]+2],v[3*idx[2]+2]-v[3*idx[3]+2],
    };
    Matrixf.inv33(self->T,T); 
    memcpy(self->ori,v+idx[3],sizeof(*self->ori));
};


static float sum(const float * const restrict v, const unsigned n) {
    float s=0.0f;
    const float *c=v+n;
    while(c-->v) s+=*c;
    return s;
}

/* 
    Maps source points into barycentric coordinates.

    src must be sized like float[3]
    dst must be sized like float[3] (same size as src)

    dst will hold four lambdas: Must be sized float[4] 
        lambda4 is computed based on the others: lambda4 = 1-lambda1-lambda2-lambda3
*/
static void map(struct tetrahedron *self,float * dst,const float * const src) {
    float tmp[3];
    memcpy(tmp,src,sizeof(float)*3);
    {
        const float * const o = self->ori;
        tmp[0]-=o[0];
        tmp[1]-=o[1];
        tmp[2]-=o[2];
    }
    Matrixf.mul(dst,self->T,3,3,tmp,1);
    dst[4]=1-sum(dst,3);
}

static unsigned prod(const unsigned * const v,unsigned n) {
    unsigned p=1;
    const unsigned *c=v+n;
    while(c-->v) p*=*c;
    return p;
}

static unsigned any_less_than_zero(const float * const restrict v,const unsigned n) {
    const float *c=v+n;
    while(c-->v) if(*c<-EPS) return 1;
    return 0;
}

static unsigned find_best_tetrad(const float * const restrict ls) {
    float mx=ls[0];
    unsigned i,argmax=0;
    for(i=1;i<4;++i) {
        if(ls[i]>mx) {
            mx=ls[i];
            argmax=i;
        }
    }
    if(mx<1.0f)
        return 0;
    return argmax+1;
}

/** 3d only */
static void idx2coord(float * restrict r,unsigned idx,const unsigned * const restrict shape) {
    r[0]=idx%shape[0];  idx/=shape[0];
    r[1]=idx%shape[1];  idx/=shape[1];
    r[2]=idx%shape[2];
}


/* THE CRUX */

/**
    @param cubeverts [in]   An array of floats ordered like float[8][3].
                            Describes the vertices of a three dimensional cube.
                            The vertices must be Morton ordered.  That is, 
                            when bit 0 of the index (0 to 7) is low, that 
                            corresponds to a vertex on the face of the cube that
                            is more minimal in x; 1 on the maximal side.
                            Bit 1 is the y dimension, and bit 2 the z dimension.
*/

static void resample(TPixel * const restrict dst,const unsigned * const restrict dst_shape,const unsigned * const restrict dst_strides,
                     TPixel * const restrict src,const unsigned * const restrict src_shape,const unsigned * const restrict src_strides,
                     const float * restrict cubeverts) {
    /* Approach

    1. Build tetrahedra from cube vertices
    2. Over pixel indexes for dst, for central tetrad
        1. map to lambdas
        2. check for oob/best tetrad.
        3. For best tetrad
           1. map to uvw
           2. sample source
    */

    /*4 indexes each for 5 tetrads; the first is the center tetrad*/
    const unsigned indexes[5][4] = {
        {1,2,4,7},
        {2,4,6,7}, // opposite 1
        {1,4,5,7}, // opposite 2
        {1,2,3,7}, // opposite 4
        {0,1,2,3}  // opposite 7
    };
    struct tetrahedron tetrads[5];
    unsigned i;
    for(i=0;i<5;i++)
        tetrahedron(tetrads+i,cubeverts,indexes[i]); // TODO: VERIFY the indexing on "indexes" works correctly here

    {
        unsigned idst;
        for(idst=0;idst<prod(dst_shape,3);++idst) {
            float r[3],lambdas[4];
            unsigned itetrad;
            idx2coord(r,idst,dst_shape);
            map(tetrads,lambdas,r);             // Map center tetrahedron
            if(any_less_than_zero(lambdas,4)) 
                continue;
            itetrad=find_best_tetrad(lambdas);
            if(itetrad>0) {
                map(tetrads+itetrad,lambdas,r);   // Map best tetrahedron
                if(any_less_than_zero(lambdas,4)) // boundary check again
                    continue;
            }

            // At this point we're ready to map source index
            {
                unsigned idim,ilambda,isrc=0;
                for(idim=0;idim<3;++idim) {
                    float s=0.0f;
                    for(ilambda=0;ilambda<4;++ilambda) {
                        const float      w=lambdas[ilambda];
                        const unsigned idx=indexes[ilambda][itetrad];                    
                        s+=w*src_shape[idim]*BIT(idx,idim);
                    }
                    isrc+=src_strides[idim]*((unsigned)s); // important to floor here.  can't change order of sums
                }
                dst[idst]=src[isrc];
            }

        }
    }

}



const struct resampler_api Barycentric = {
    resample
};
