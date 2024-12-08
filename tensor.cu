#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <chrono>
#include <cfloat>

#define M 4096
#define N 4096
#define K 4096
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda;

void runCudaFunction();
void runTensorFunction();
void runDynamicFunction();
float timePartialCudaGemm(int start_row, int num_rows, float* d_a, float* d_b, float* d_c);
float timePartialTensorGemm(int start_row, int num_rows, half* d_a, half* d_b, float* d_c);

__global__ void cuda_gemm(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) {
            acc += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = acc;
    }
}

__global__ void wmma_tensor_gemm(half *a, half *b, float *c, int m, int n, int k) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warp_n = blockIdx.y * blockDim.y + threadIdx.y;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < k; i += WMMA_K) {
        int matrix_a_idx = warp_m * WMMA_M * k + i;
        int matrix_b_idx = i * n + warp_n * WMMA_N;

        wmma::load_matrix_sync(a_frag, a + matrix_a_idx, k);
        wmma::load_matrix_sync(b_frag, b + matrix_b_idx, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int c_idx = warp_m * WMMA_M * n + warp_n * WMMA_N;
    wmma::store_matrix_sync(c + c_idx, c_frag, n, wmma::mem_row_major);
}

__global__ void fused_split_gemm(float *a_float, float *b_float, float *c_float,
                                half *a_half, half *b_half, float *c_half,
                                int m, int n, int k, int split_point) {
    int warp_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < split_point) {
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        if (col < n) {
            float acc = 0.0f;
            for (int i = 0; i < k; i++) {
                acc += a_float[row * k + i] * b_float[i * n + col];
            }
            c_float[row * n + col] = acc;
        }
    } else if (row >= split_point && row < m) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

        int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize - split_point/warpSize;
        int warp_n = blockIdx.y * blockDim.y + threadIdx.y;

        if (warp_m * WMMA_M < (m - split_point) && warp_n < n/WMMA_N) {
            wmma::fill_fragment(c_frag, 0.0f);
            
            for (int i = 0; i < k; i += WMMA_K) {
                int matrix_a_idx = (warp_m * WMMA_M + split_point) * k + i;
                int matrix_b_idx = i * n + warp_n * WMMA_N;

                wmma::load_matrix_sync(a_frag, a_half + matrix_a_idx, k);
                wmma::load_matrix_sync(b_frag, b_half + matrix_b_idx, n);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            int c_idx = (warp_m * WMMA_M + split_point) * n + warp_n * WMMA_N;
            wmma::store_matrix_sync(c_half + c_idx, c_frag, n, wmma::mem_row_major);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " --cuda | --tensor | --dynamic" << std::endl;
        return 1;
    }

    if (strcmp(argv[1], "--cuda") == 0) {
        runCudaFunction();
    } else if (strcmp(argv[1], "--tensor") == 0) {
        runTensorFunction();
    } else if (strcmp(argv[1], "--dynamic") == 0) {
        runDynamicFunction();
    } else {
        std::cerr << "Invalid argument: " << argv[1] << std::endl;
        std::cerr << "Usage: " << argv[0] << " --cuda | --tensor | --dynamic" << std::endl;
        return 1;
    }
    return 0;
}

template<typename T>
T getValue() {
    return T(1.0f);
}

template<>
half getValue<half>() {
    return __float2half(1.0f);
}

template<typename T>
void runGemmFunction(bool isCuda) {
    double total_operations = 2.0 * M * N * K;
    double total_gb = (M*K + K*N + M*N) * sizeof(float) / 1e9;
    
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Total operations: %.2f GFLOPs\n", total_operations / 1e9);
    printf("Total memory: %.2f GB\n", total_gb);

    size_t size_a = M * K * sizeof(T);
    size_t size_b = K * N * sizeof(T);
    size_t size_c = M * N * sizeof(float);

    T *d_a, *d_b;
    float *d_c;

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    T *h_a = (T *)malloc(size_a);
    T *h_b = (T *)malloc(size_b);
    float *h_c = (float *)malloc(size_c);

    for (int i = 0; i < M * K; i++) {
        h_a[i] = getValue<T>();
    }
    for (int i = 0; i < K * N; i++) {
        h_b[i] = getValue<T>();
    }

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (isCuda) {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                          (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
        cuda_gemm<<<blocksPerGrid, threadsPerBlock>>>((float*)d_a, (float*)d_b, d_c, M, N, K);
    } else {
        dim3 threadsPerBlock(32, 8);
        dim3 blocksPerGrid((M + (WMMA_M * 2) - 1) / (WMMA_M * 2), 
                          (N + WMMA_N - 1) / WMMA_N);
        wmma_tensor_gemm<<<blocksPerGrid, threadsPerBlock>>>((half*)d_a, (half*)d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);

    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time for %s: %f ms\n", isCuda ? "cuda_gemm" : "tensor_gemm", milliseconds);
    double gflops = (total_operations / milliseconds) / 1e6;
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Memory throughput: %.2f GB/s\n", total_gb/(milliseconds/1000.0));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

void runCudaFunction() {
    runGemmFunction<float>(true);
}

void runTensorFunction() {
    runGemmFunction<half>(false);
}

float timePartialCudaGemm(int start_row, int num_rows, float* d_a, float* d_b, float* d_c) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((num_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    cuda_gemm<<<blocksPerGrid, threadsPerBlock>>>(
        d_a + start_row * K, d_b, d_c + start_row * N, num_rows, N, K);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

float timePartialTensorGemm(int start_row, int num_rows, half* d_a, half* d_b, float* d_c) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 threadsPerBlock(32, 8);
    dim3 blocksPerGrid((num_rows + (WMMA_M * 2) - 1) / (WMMA_M * 2),
                       (N + WMMA_N - 1) / WMMA_N);

    cudaEventRecord(start);
    wmma_tensor_gemm<<<blocksPerGrid, threadsPerBlock>>>(
        d_a + start_row * K, d_b, d_c + start_row * N, num_rows, N, K);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

void runDynamicFunction() {
    float *d_a_float, *d_b_float, *d_c_float;
    half *d_a_half, *d_b_half;
    float *d_c_half;
    
    cudaMalloc(&d_a_float, M * K * sizeof(float));
    cudaMalloc(&d_b_float, K * N * sizeof(float));
    cudaMalloc(&d_c_float, M * N * sizeof(float));
    cudaMalloc(&d_a_half, M * K * sizeof(half));
    cudaMalloc(&d_b_half, K * N * sizeof(half));
    cudaMalloc(&d_c_half, M * N * sizeof(float));

    float *h_a_float = (float *)malloc(M * K * sizeof(float));
    half *h_a_half = (half *)malloc(M * K * sizeof(half));
    for (int i = 0; i < M * K; i++) {
        h_a_float[i] = 1.0f;
        h_a_half[i] = __float2half(1.0f);
    }
    cudaMemcpy(d_a_float, h_a_float, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_half, h_a_half, M * K * sizeof(half), cudaMemcpyHostToDevice);

    int left = 1;
    int right = M - 1;
    float best_diff = FLT_MAX;
    int best_split = M/2;
    float best_cuda_time = 0;
    float best_tensor_time = 0;
    
    while (left < right && (right - left) > 32) {
        int mid = (left + right) / 2;
        
        float cuda_time = timePartialCudaGemm(0, mid, d_a_float, d_b_float, d_c_float);
        float tensor_time = timePartialTensorGemm(mid, M-mid, d_a_half, d_b_half, d_c_half);
        
        float time_diff = fabs(cuda_time - tensor_time);
        if (time_diff < best_diff) {
            best_diff = time_diff;
            best_split = mid;
            best_cuda_time = cuda_time;
            best_tensor_time = tensor_time;
        }
        
        if (cuda_time > tensor_time) {
            right = mid;
        } else {
            left = mid + 1;
        }
        
        printf("Split at %d rows: CUDA %.2fms, Tensor %.2fms, Diff %.2fms\n", 
               mid, cuda_time, tensor_time, time_diff);
    }
    
    printf("\nOptimal split found:\n");
    printf("CUDA GEMM: %d rows (%.2f ms)\n", best_split, best_cuda_time);
    printf("Tensor GEMM: %d rows (%.2f ms)\n", M - best_split, best_tensor_time);
    printf("Time difference: %.2f ms\n", best_diff);
    printf("Split ratio: %.2f%% CUDA, %.2f%% Tensor\n", 
           (float)best_split/M * 100, (float)(M-best_split)/M * 100);

    printf("\nRunning split-aware fused kernel with optimal split at row %d...\n", best_split);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    fused_split_gemm<<<numBlocks, threadsPerBlock>>>(
        d_a_float, d_b_float, d_c_float,
        d_a_half, d_b_half, d_c_half,
        M, N, K, best_split);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float fused_time;
    cudaEventElapsedTime(&fused_time, start, stop);
    
    printf("\nPerformance Comparison:\n");
    printf("Separate execution total time: %.2f ms (CUDA: %.2f ms + Tensor: %.2f ms)\n", 
           best_cuda_time + best_tensor_time, best_cuda_time, best_tensor_time);
    printf("Fused kernel execution time: %.2f ms\n", fused_time);
    printf("Speedup from fusion: %.2fx\n", 
           (best_cuda_time + best_tensor_time) / fused_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_a_float);
    cudaFree(d_b_float);
    cudaFree(d_c_float);
    cudaFree(d_a_half);
    cudaFree(d_b_half);
    cudaFree(d_c_half);
    free(h_a_float);
    free(h_a_half);
}