#include "mat.h"
#include <string.h>

static float* i33(float * restrict inverse,const float * const restrict t) {
#define det(a,b,c,d) (t[a]*t[d]-t[b]*t[c])
    const float n=t[0]*det(4,5,7,8)-t[1]*det(3,5,6,8)+t[2]*det(3,4,6,7);
    float invT[3][3]={
        {det(4,5,7,8)/n, det(2,1,8,7)/n, det(1,2,4,5)/n},
        {det(5,3,8,6)/n, det(0,2,6,8)/n, det(2,0,5,3)/n},
        {det(3,4,6,7)/n, det(1,0,7,6)/n, det(0,1,3,4)/n},
    }; 
#undef det
    memcpy(inverse,invT,sizeof(invT));
    return inverse;
}

static float* mul(float * restrict dst, 
                  const float * const restrict lhs, const unsigned nrows_lhs, const unsigned ncols_lhs,
                  const float * const restrict rhs, const unsigned ncols_rhs){
    unsigned i,j,k;
    for(k=0;k<nrows_lhs;++k) {
        for(j=0;j<ncols_rhs;++j) {
            float s=0.0f;
            for(i=0;i<ncols_lhs;++i) {
                s+=lhs[ncols_lhs*k+i]*rhs[ncols_rhs*i+j];
            }
            dst[ncols_rhs*k+j]=s;
        }
    }
    return dst;
}

const struct mat_f32_api Matrixf={
    i33,
    mul
};
