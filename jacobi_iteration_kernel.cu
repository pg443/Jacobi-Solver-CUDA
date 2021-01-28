#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */

__global__ void jacobi_iteration_kernel_naive(matrix_t Ad, matrix_t Bd, matrix_t Xd, double* globalSSD)
{
    __shared__ double sum_per_thread[THREAD_BLOCK_SIZE];    /* Shared memory for thread block */
    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    if (i > Xd.num_rows-1){
        sum_per_thread[threadIdx.x] = 0;
        return;
    }

    int num_cols = Ad.num_columns;
    float new_x;
    double sum = -Ad.elements[i * num_cols + i] * Xd.elements[i];
    for (int j = 0; j < num_cols; j++){
        sum += Ad.elements[i * num_cols + j] * Xd.elements[j];
    }
    new_x = (Bd.elements[i] - sum)/Ad.elements[i * num_cols + i];
    sum_per_thread[threadIdx.x] = (new_x - Xd.elements[i]) * (new_x - Xd.elements[i]);
    __syncthreads();

    int j = blockDim.x/2;
    while (j != 0) {
        if (threadIdx.x < j)
            sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + j];
        __syncthreads();
        j /= 2;
    }

    Xd.elements[i] = new_x;
    /* Check for convergence and update the unknowns. */
    if (threadIdx.x == 0){
        atomicAdd(globalSSD, sum_per_thread[0]);
    }

    return;
}

__global__ void jacobi_iteration_kernel_optimized(matrix_t Ad, matrix_t Bd, matrix_t Xd, double* globalSSD)
{
    __shared__ double sum_per_thread[THREAD_BLOCK_SIZE];    /* Shared memory for thread block */
    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    if (i > Xd.num_rows-1){
        sum_per_thread[threadIdx.x] = 0;
        return;
    }

    int num_cols = Ad.num_columns;
    float new_x;
    double sum = -Ad.elements[i * num_cols + i] * Xd.elements[i];
    for (int j = 0; j < num_cols; j++){
        sum += Ad.elements[j * num_cols + i] * Xd.elements[j];
    }
    new_x = (Bd.elements[i] - sum)/Ad.elements[i * num_cols + i];
    sum_per_thread[threadIdx.x] = (new_x - Xd.elements[i]) * (new_x - Xd.elements[i]);
    __syncthreads();

    int j = blockDim.x/2;
    while (j != 0) {
        if (threadIdx.x < j)
            sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + j];
        __syncthreads();
        j /= 2;
    }

    Xd.elements[i] = new_x;
    /* Check for convergence and update the unknowns. */
    if (threadIdx.x == 0){
        atomicAdd(globalSSD, sum_per_thread[0]);
    }
    return;
}

