# winograd\_avx

3x3 Winograd convolution on CPU, optimized for AVX with FMA support, faster than Torch (singlethreaded.)

## Running

`./run.sh` to compile, run, and benchmark against PyTorch. `#define PROFILE` in `winograd.c` to see a breakdown of execution time.

Requires clang for compilation, and Python, numpy, and torch for benchmarking. Requires an x64 CPU with the FMA featureset.

## Acknowledgements

The matrix multiplication functions in helpers.h were adapted from [tinygrad.](https://github.com/tinygrad/tinygrad)

## Notes

This is primarily a speed demonstration. If you want to use it for some practical purpose, and you have more than 1 core, you could speed it up greatly by implementing a multithreaded version of `gemm_preallocated()`. (See tinygrad's `gemm.c` for an example of how to do this.)
