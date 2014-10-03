#pragma once

//#include <nd.h>
typedef struct nd_t_ * nd_t;

struct barycentric {
    float T[9];
};

struct render {
    union {
        struct barycentric barycentric;
    };
};

struct field_api {
    void (*init)(struct render *params);
    void (*compute)(nd_t dst, nd_t src, struct render *params);
};

struct barycentric_api {
    void (*init)(struct barycentric *params,const float * const vertices);
};

extern const struct barycentric_api Barycentric;


extern const struct field_api BarycentricCPU;
//extern const struct render_api BarycentricGPU;
//extern const struct render_api DownsampleGPU;
//extern const struct render_api DownsampleCPU;
