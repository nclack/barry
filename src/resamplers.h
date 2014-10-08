#pragma once
#include <stdint.h>

#define restrict __restrict

typedef uint8_t TPixel;

struct resampler_api {
    void (*resample)(TPixel * const restrict dst,const unsigned * const restrict dst_shape,const unsigned * const restrict dst_strides,
                     TPixel * const restrict src,const unsigned * const restrict src_shape,const unsigned * const restrict src_strides,
                     const float * restrict cubeverts);
    int  (*runTests)(void);
};

extern const struct resampler_api Barycentric;
