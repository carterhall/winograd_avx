# winograd\_avx

3x3 Winograd convolution on CPU, optimized for AVX with FMA support, faster than Torch.

`./run.sh` to compile, run, and benchmark against PyTorch.

Requires clang for compilation, and Python, numpy, and torch for benchmarking. Requires an x64 CPU with the FMA featureset.


## Acknowledgements

The matrix multiplication functions in helpers.h were adapted from [tinygrad.](https://github.com/tinygrad/tinygrad)
