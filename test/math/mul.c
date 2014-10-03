#include <mat.h>

const float A[]={
    0.8147f,   0.9134f,   0.278f,
    0.9058f,   0.6324f,   0.546f,
    0.1270f,   0.0975f,   0.957f,
};

const float B[]={
    0.9649,   0.9572,   0.1419,   0.7922,   0.0357,
    0.1576,   0.4854,   0.4218,   0.9595,   0.8491,
    0.9706,   0.8003,   0.9157,   0.6557,   0.9340,
};

const float res[]={
    1.2004,   1.4460,   0.7559,   1.7044,   1.0648,
    1.5045,   1.6116,   0.8961,   1.6830,   1.0801,
    1.0673,   0.9352,   0.9360,   0.8220,   0.9816,
};
 


static unsigned eq(const float * const a, const int n, const float * const b) {
    int i;
    for(i=0;i<n;++i) {
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
    float ws[3*5];
    ASSERT(eq(res,sizeof(res)/sizeof(*res),
              Matrixf.mul(ws,A,3,3,B,5)));
    return 0;
}
