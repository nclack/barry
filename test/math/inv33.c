#include <matrix.h>


const float A[]={
    0.8147f,   0.9134f,   0.278f,
    0.9058f,   0.6324f,   0.546f,
    0.1270f,   0.0975f,   0.957f,
};

const float invA[]={
   -1.9958f,   3.0630f,  -1.1690f,
    2.8839f,  -2.6919f,   0.6987f,
   -0.0291f,  -0.1320f,   1.1282f,
};
 


static unsigned eq(const float * const a, const float * const b) {
    int i;
    for(i=0;i<9;++i) {
        const float v = a[i]-b[i];
        if(v*v>1e-3)
            return 0;
    }
    return 1;
}

#include <stdio.h>
#include <stdlib.h>
#define ASSERT(e) do{if(!(e)) {printf("%s(%d): %s()(\n\tExpression evaluated as false.\n\t%s\n",__FILE__,__LINE__,__FUNCTION__,#e); exit(1); }}while(0)

int main() {
    float ws[9];
    ASSERT(eq(invA,Matrixf.inv33(ws,A)));
    return 0;
}
