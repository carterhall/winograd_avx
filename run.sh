#!/bin/bash
clang -O2 -mfma winograd.c && ./a.out && python3 verify_with_torch.py
#clang -fsanitize=address -mfma winograd.c && ./a.out && python3 verify_with_torch.py
