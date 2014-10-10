#include <cuda.h>
#include <cuda_runtime.h>

#define WARPS_PER_BLOCK  4
#define BLOCKSIZE       (32*WARPS_PER_BLOCK) // threads per block

#define restrict __restrict

template <class T> struct vol_t {
    T * const restrict data;
    unsigned shape[3];
    unsigned strides[4];
};

struct tetrahedron {
    float T[9];
    float ori[3];
};

template <class T>
__global__
void
__launch_bounds__(BLOCKSIZE,1)
barycentric_kernel(vol_t<T> dst,cudateTextureObject_t src, const struct tetrads[5]) {
    
}