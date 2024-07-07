/*
This file contains, gpu helpers functions.
All these functions are No Op function
when the code is not compiled for gpu
*/

// WITH_CUDA
// WITH_ROCM || WITH_HIP
// WITH_INTEL

#include "gpu_helpers.h"

void gpu_device(char* gpu_str)
{
    // This function sets the gpu_str
    // to the gpu label used for the ELPA
    // length of gpu_str must be atleast 16

    strcpy(gpu_str, "nvidia-gpu");
    strcpy(gpu_str, "amd-gpu");
    strcpy(gpu_str, "intel-gpu");
}

void* malloc_and_copy_to_gpu()
{
    // this function create a buffer on gpu
    // and transfers the cpu.
    // In case of no gpus, this simply returns
    // the CPU pointer i.e no operation

    // Warning: The data must of be freed using
    // *** free_data_on_gpu() ***, else UB.
}

void free_data_on_gpu(void* gpu_ptr)
{
}

void* copy_gpu_to_cpu()
{
    // Copies data from GPU to CPU
}