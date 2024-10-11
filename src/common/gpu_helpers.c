/* */
/*This file contains, gpu helpers functions. */
/*All these functions are No Op function */
/*when the code is not compiled for gpu */
/* */

#include "gpu_helpers.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "../diago.h"

#ifdef WITH_GPU

#if defined(WITH_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(WITH_HIP)
#include <hip/hip_runtime.h>
#elif defined(WITH_INTEL_GPU)
#include <omp.h>
#else
#error unsupported gpu flag
#endif

#endif

bool isGPUpresent(void)
{
#ifdef WITH_GPU
    return true;
#else
    return false;
#endif
}

void set_elpa_gpu_str(char* str)
{
#ifdef WITH_GPU

#if defined(WITH_CUDA)
    strcpy(str, "nvidia-gpu");
#elif defined(WITH_HIP)
    strcpy(str, "amd-gpu");
#elif defined(WITH_INTEL_GPU)
    strcpy(str, "intel-gpu");
#else
#error unsupported gpu flag
#endif

#else
    strcpy(str, "not-supported");
#endif
}

void* gpu_malloc(size_t size)
{
    void* gpu_ptr = NULL;
#ifdef WITH_GPU

#if defined(WITH_CUDA)
    if (cudaMalloc(&gpu_ptr, size) != cudaSuccess)
    {
        gpu_ptr = NULL;
    }
#elif defined(WITH_HIP)
    if (hipMalloc(&gpu_ptr, size) != hipSuccess)
    {
        gpu_ptr = NULL;
    }
#elif defined(WITH_INTEL_GPU)
    int device_id = omp_get_default_device();
    gpu_ptr = omp_target_alloc(size, device_id);
#else
#error unsupported gpu flag
#endif

#endif
    return gpu_ptr;
}

int gpu_memcpy(void* dest_ptr, void* src_ptr, size_t size,
               enum gpuMemcpyDir dir)
{
    int error = 0;
#ifdef WITH_GPU

#if defined(WITH_CUDA)
    if (dir == Copy2GPU)
    {
        if (cudaMemcpy(dest_ptr, src_ptr, size, cudaMemcpyHostToDevice) !=
            cudaSuccess)
        {
            error = 1;
        }
    }
    else if (dir == Copy2CPU)
    {
        if (cudaMemcpy(dest_ptr, src_ptr, size, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
        {
            error = 1;
        }
    }
#elif defined(WITH_HIP)
    if (dir == Copy2GPU)
    {
        if (hipMemcpy(dest_ptr, src_ptr, size, hipMemcpyHostToDevice) !=
            hipSuccess)
        {
            error = 1;
        }
    }
    else if (dir == Copy2CPU)
    {
        if (hipMemcpy(dest_ptr, src_ptr, size, hipMemcpyDeviceToHost) !=
            hipSuccess)
        {
            error = 1;
        }
    }
#elif defined(WITH_INTEL_GPU)
    int device_id = omp_get_default_device();
    int host_id = omp_get_initial_device();
    if (dir == Copy2GPU)
    {
        error = omp_target_memcpy(dest_ptr, src_ptr, size, 0, 0, device_id,
                                  host_id);
    }
    else if (dir == Copy2CPU)
    {
        error = omp_target_memcpy(dest_ptr, src_ptr, size, 0, 0, host_id,
                                  device_id);
    }
#else
#error unsupported gpu flag
#endif

#endif

    return error;
}

int gpu_free(void* gpu_ptr)
{
    if (!gpu_ptr)
    {
        return 0;
    }
    int error = 0;
#ifdef WITH_GPU
#if defined(WITH_CUDA)
    if (cudaFree(gpu_ptr) != cudaSuccess)
    {
        error = 1;
    }
#elif defined(WITH_HIP)
    if (hipFree(gpu_ptr) != hipSuccess)
    {
        error = 1;
    }
#elif defined(WITH_INTEL_GPU)
    int device_id = omp_get_default_device();
    omp_target_free(gpu_ptr, device_id);
#else
#error unsupported gpu flag
#endif

#endif
    return error;
}
