# winograd\_avx

3x3 Winograd convolution on CPU, optimized for AVX with FMA support, faster than Torch (singlethreaded.)

## Running

`./run.sh` to compile, run, and benchmark against PyTorch. `#define PROFILE` in `winograd.c` to see a breakdown of execution time.

Requires clang for compilation, and Python, numpy, and torch for benchmarking. Requires an x64 CPU with the FMA featureset.

## Sample Output (Ryzen 7700X)

    Input tensor: 4.718592 MB, output tensor: 603.979776 MB.
    Preallocating 55.705600 MB for temp storage.
    Batch size 4 with 512 channels, 512 kernels, image size 24x24
    Winograd theoretical complexity: 4.979687 GFLOPs.
    Calling winograd 16 times...

    We're achieving 115.497243 GFLOP/S, 43.115206 ms per iteration.

    =========================================================================================
    PyTorch verification: Loading tensors from file... 
    Transposing tensors... 
    Running conv2d...
    PyTorch conv2d: 85.94 GFLOP/S, 57.94 ms
    MATCH. All Winograd outputs are close to Torch's conv2d.


## Acknowledgements

The matrix multiplication functions in helpers.h were adapted from [tinygrad.](https://github.com/tinygrad/tinygrad)

## Notes

This is primarily a speed demonstration. If you want to use it for some practical purpose, and you have more than 1 core, you could speed it up greatly by implementing a multithreaded version of `gemm_preallocated()`. (See tinygrad's `gemm.c` for an example of how to do this.)
