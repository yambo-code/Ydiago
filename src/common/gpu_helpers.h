#pragma once
#include <stddef.h>
#include <stdbool.h>

enum gpuMemcpyDir
{
    Copy2GPU,
    Copy2CPU
};

bool isGPUpresent(void);
void set_elpa_gpu_str(char* str);
void* gpu_malloc(size_t size);
int gpu_memcpy(void* dest_ptr, void* src_ptr, size_t size, enum gpuMemcpyDir dir);
int gpu_free(void* gpu_ptr);
