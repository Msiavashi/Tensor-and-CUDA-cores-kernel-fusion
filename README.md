# CUDA & Tensor Cores Fused Gemm Operation

This repository contains CUDA implementations of Gemm operation to compare CUDA and Tensor cores performance.

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed

### Installation

1. Clone the repository:
    ```sh
    $ git clone https://github.com/msiavashi/cuda-tensor-operations.git
    $ cd cuda-tensor-operations
    ```

2. Compile and execute the `tensor.cu` file:
    ```sh
    $ nvcc -o tensor ./tensor.cu
    $ nvcc -o streams ./streams.cu

### Project structure

```sh
.
├── README.md
├── streams.cu  # To experiment overlapping execution of CUDA streams.
└── tensor.cu  # To experiment Tensor & CUDA cores Gemm operation
```

### Running the Code
To run the compiled executable:

```sh
$ ./tensor_gemm --cuda
$ ./tensor_gemm --tensor
$ ./tensor_gemm --dynamic
```

- `--cuda`: Executes Gemm on CUDA cores
- `--tensor`: Executes Gemm on Tensor cores
- `--dynamic`: Determines the optimal split ratio for matrices such that the execution times of Tensor Cores and CUDA are approximately equal. Note that this method identifies the closest match in execution times, rather than an exact equivalence. Next, it will compare the fused kernel utilizing both CUDA Cores and Tensor Cores with the independent execution of these two kernels.

__Note__: To set Matrix size you may adjust the followings macros at top of the file:
```c
#define M 4096
#define N 4096
#define K 4096
```

### Sample Outputs

`$ ./tensor --dynamic`:

```sh
Split at 8192 rows: CUDA 3373.64ms, Tensor 466.34ms, Diff 2907.30ms
Split at 4096 rows: CUDA 1689.79ms, Tensor 710.32ms, Diff 979.47ms
Split at 2048 rows: CUDA 840.36ms, Tensor 879.80ms, Diff 39.44ms
Split at 3072 rows: CUDA 1260.73ms, Tensor 794.31ms, Diff 466.42ms
Split at 2560 rows: CUDA 1045.89ms, Tensor 851.10ms, Diff 194.79ms
Split at 2304 rows: CUDA 942.11ms, Tensor 880.01ms, Diff 62.10ms
Split at 2176 rows: CUDA 890.77ms, Tensor 894.44ms, Diff 3.67ms
Split at 2240 rows: CUDA 917.71ms, Tensor 878.79ms, Diff 38.92ms
Split at 2208 rows: CUDA 903.73ms, Tensor 901.98ms, Diff 1.74ms

Optimal split found:
CUDA GEMM: 2208 rows (903.73 ms)
Tensor GEMM: 14176 rows (901.98 ms)
Time difference: 1.74 ms
Split ratio: 53.91% CUDA, 346.09% Tensor

Running split-aware fused kernel with optimal split at row 2208...

Performance Comparison:
Separate execution total time: 1805.71 ms (CUDA: 903.73 ms + Tensor: 901.98 ms)
Fused kernel execution time: 1851.02 ms
Speedup from fusion: 0.98x
```

### How to profile

To profile using Nsight Systems use the following command as example:

`$ CUDA_VISIBLE_DEVICES=1 nsys  profile --force-overwrite true  -t nvtx,cuda -o ./profile/tensor_dynamic ./tensor --dynamic`