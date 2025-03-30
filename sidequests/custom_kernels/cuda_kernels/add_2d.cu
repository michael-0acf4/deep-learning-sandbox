#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

extern "C" {
    __global__ void add_kernel_2d(int *a, int *b, int *c, int rows, int cols) {
        // TODO:
        // handle...
        // min blockIdx and max? 
        // min threadIdx and max?
        // both sets the range of maximum size of a matrix

        // Block 1 
        // | ----------> thread 1   |
        // | ----------> thread 2   |
        // | ....................   |
        // | ----------> thread Max?|
        // ...
        // Block Max

        int bloc_pos_x =  blockIdx.x * blockDim.x;
        int bloc_pos_y =  blockIdx.y * blockDim.y;
        int i = bloc_pos_x + threadIdx.x;
        int j = bloc_pos_y + threadIdx.y;
        // if (j < rows && i < cols) {
            int idx = j * rows + i;
            c[idx] = a[idx] + b[idx];
        // }
    }

    void launch_add(int *a, int *b, int *c, int rows, int cols) {
        int *d_a, *d_b, *d_c;
        size_t size = rows * cols * sizeof(int);

        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);

        clock_t begin1 = clock();
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
        clock_t end1 = clock();
        printf(" [cuda] memcpy python --> cuda: %fs\n", (double) (end1 - begin1) / CLOCKS_PER_SEC);

        dim3 blocks(rows);
        dim3 threads(cols);
        clock_t begin2 = clock();
        add_kernel_2d<<<blocks, threads>>>(d_a, d_b, d_c, rows, cols);
        clock_t end2 = clock();
        printf(" [cuda] kernel raw compute cost: %fs\n", (double) (end2 - begin2) / CLOCKS_PER_SEC);

        clock_t begin3 = clock();
        cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
        clock_t end3 = clock();
        printf(" [cuda] memcpy cuda --> python: %fs\n", (double) (end3 - begin3) / CLOCKS_PER_SEC);

        printf(" [cuda] Total %fs\n", (double) (end3 - begin1) / CLOCKS_PER_SEC);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
}