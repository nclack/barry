#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define restrict __restrict

typedef uint8_t TPixel;

struct resampler_api {
    void (*resample)(TPixel * const restrict dst,const unsigned * const restrict dst_shape,const unsigned * const restrict dst_strides,
                     TPixel * const restrict src,const unsigned * const restrict src_shape,const unsigned * const restrict src_strides,
                     const float * restrict cubeverts);
    int  (*runTests)(void);
};

#if 0
struct job_api {
    struct ctx* (*alloc)();
    void (*destroy )(struct ctx*);
    void (*source  )(struct ctx*,TPixel * const restrict src,const unsigned * const restrict src_shape,const unsigned * const restrict src_strides);
    void (*resample)(struct ctx*,TPixel * const restrict dst,const unsigned * const restrict dst_shape,const unsigned * const restrict dst_strides,
                     const float * restrict cubeverts);
};
#endif

extern const struct resampler_api BarycentricGPU;
extern const struct resampler_api BarycentricCPU;


#ifdef __cplusplus
} // extern "C"
#endif


/* TODO

1. refactor resample call to support
  
   - load src once
   - render many outputs

*/