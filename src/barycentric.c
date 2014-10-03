#include "render.h"
#include "mat.h"

static void tetrahedron(struct barycentric *self, const float * const v) {
    float T[9] = {
        v[0]-v[9 ] ,v[3]-v[9 ],v[6]-v[9 ],
        v[1]-v[10] ,v[4]-v[10],v[7]-v[10],
        v[2]-v[11] ,v[5]-v[11],v[8]-v[11],
    };
    Matrixf.inv33(self->T,T); 
};


const struct barycentric_api Barycentric = {
    tetrahedron
};
