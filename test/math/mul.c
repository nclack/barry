#include <matrix.h>

const float A[]={
    0.8147f,   0.9134f,   0.278f,
    0.9058f,   0.6324f,   0.546f,
    0.1270f,   0.0975f,   0.957f,
};

const float B[]={
    0.9649f,   0.9572f,   0.1419f,   0.7922f,   0.0357f,
    0.1576f,   0.4854f,   0.4218f,   0.9595f,   0.8491f,
    0.9706f,   0.8003f,   0.9157f,   0.6557f,   0.9340f,
};

const float res[]={
    1.2004f,   1.4460f,   0.7559f,   1.7044f,   1.0648f,
    1.5045f,   1.6116f,   0.8961f,   1.6830f,   1.0801f,
    1.0673f,   0.9352f,   0.9360f,   0.8220f,   0.9816f,
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
