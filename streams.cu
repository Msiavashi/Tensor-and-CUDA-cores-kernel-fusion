#include <cuda_runtime.h>
#include <stdio.h>

#define N (128 * 128)
#define ITERATIONS 100  // Increase the number of iterations

__global__ void dummy_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Dummy computation to simulate work
        float val = data[idx];
        for (int i = 0; i < 10000; i++) {  // Increase the number of iterations
            val = sinf(val);
        }
        data[idx] = val;
    }
}

int main() {
    float *d_data1, *d_data2;
    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));

    // Initialize data
    float *h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    cudaMemcpy(d_data1, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    free(h_data);

    // Create two CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Time measurement variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float parallel_time, sequential_time;

    // Test parallel execution with two streams
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        dummy_kernel<<<(N+255)/256, 256, 0, stream1>>>(d_data1, N);
        dummy_kernel<<<(N+255)/256, 256, 0, stream2>>>(d_data2, N);
    }
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&parallel_time, start, stop);

    // Test sequential execution
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        dummy_kernel<<<(N+255)/256, 256>>>(d_data1, N);
        dummy_kernel<<<(N+255)/256, 256>>>(d_data2, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sequential_time, start, stop);

    printf("Parallel execution time (ms): %f\n", parallel_time);
    printf("Sequential execution time (ms): %f\n", sequential_time);
    printf("Speedup: %.2fx\n", sequential_time / parallel_time);

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data1);
    cudaFree(d_data2);

    return 0;
}