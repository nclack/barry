#pragma once
#include <stdint.h>

void write_tiff_u8(const char* filename,size_t ndim,const size_t* shape,uint8_t *data);
