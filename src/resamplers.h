#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef uint8_t TPixel; /* Note (3) */

struct resampler {
    void *ctx;
};

struct resampler_api {
    /* See Notes (1,4) */
    int (*init)(struct resampler* self,
                const unsigned * const shape,     /* output volume pixelation */
                const unsigned ndim
               );
    int (*source)(struct resampler * self,
                  TPixel * const src,
                  const unsigned * const shape,
                  const unsigned ndim);
    int (*result)(const struct resampler * const self,
                  TPixel * const dst);
    int(*resample)(struct resampler * const self,
                    const float * const cubeverts  /* corners of 3d rectangular prism */
                    );
    void (*release)(struct resampler *self);

    void (*useReporters)( void (*error)  (const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
                          void (*warning)(const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
                          void (*info)   (const char* msg, const char* expr, const char* file, int line, const char* function,void* usr),
                          void *usr
                        );
    int  (*runTests)(void); /* See Note (2) */
};

extern const struct resampler_api BarycentricGPU; /* See Note (4) */
extern const struct resampler_api BarycentricCPU;


#ifdef __cplusplus
} // extern "C"
#endif


/* TODO

1. refactor resample call to support
  
   - load src once
   - render many outputs

2. Add ability to query capabilities (max dim,max shape,allowable pixel types)

*/

/* NOTES

   1. Renderer will not handle all possible inputs. See Todo (2).
      Assumes dst and src (of course) are allocated by caller.

   2. runTests() is here so that static utility functions can be run through
      their paces.  The utility functions are static because they're private
      parts of the interface; I don't want to worry about namespace pollution.

   3. Figure out how to generalize the interface over pixel types later.
      It's probably not necessary, and if it is, it will be simple to adapt
      the existing code.

   4. GPU renderer assumes transfer to/from RAM is desired.  See Todo (1).

      Recommend implementing another interface if src,dst are supposed to 
      be device pointers.  Esp since there may be additional requirements on 
      the type of gpu storage there...

      There's probably a way of generalizing so code gets reused on the 
      backend.  The interface here can stay the same and the caller just
      chooses the right implementaiton for what they want.

*/