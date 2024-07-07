#pragma once

void gpu_device(char* gpu_str);
void* malloc_and_copy_to_gpu();
void free_data_on_gpu(void* gpu_ptr);
void* copy_gpu_to_cpu();