#pragma once

#define restrict __restrict

struct mat_f32_api {
    float* (*inv33)(float * restrict inverse,const float * const restrict t);
    float* (*mul)  (float * restrict dst, 
                    const float * const restrict lhs, const unsigned nrows_lhs, const unsigned ncols_lhs,
                    const float * const restrict rhs, const unsigned ncols_rhs);
};

extern const struct mat_f32_api Matrixf;
