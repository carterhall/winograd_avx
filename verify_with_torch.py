import torch; import torch.nn as nn; import torch.nn.functional as F
import time, math
import numpy as np

torch.set_num_threads(1)

from reference_winograd import winograd_3x3

print()
print('=========================================================================================')
print('PyTorch verification: Loading tensors from file... ')

from tmp_data import *

print('Transposing tensors... ')

# We use a different memory order than Torch, so we pretranspose here for a fair comparison
#transposed_inputs = torch.transpose(inputs, 0, 1)
transposed_inputs = torch.transpose(inputs, 1, 3)             # NHWC ==> NCWH
transposed_inputs = torch.transpose(transposed_inputs, 2, 3)  # NCWH ==> NCHW

transposed_kernels = torch.transpose(kernels, 1, 3)              # K 3_1 3_2 C ==> K C 3_2 3_1
transposed_kernels = torch.transpose(transposed_kernels, 2, 3)   # K C 3_2 3_1 ==> K C 3_1 3_2

our_winograd_outputs = torch.transpose(outputs, 0, 1)[:,:,:-2,:-2]


print('Running conv2d...')
with torch.no_grad():
  tstart = time.monotonic_ns()
  for _ in range(16):
    correct_outputs = F.conv2d(transposed_inputs, transposed_kernels)
  tend = time.monotonic_ns()
seconds = (tend - tstart)*1e-9 / 16

print(f"PyTorch conv2d: {total_flop/seconds * 1e-9:.2f} GFLOP/S, {seconds*1e3:.2f} ms")

if False:   # Useful for debugging C speed tricks (not really anymore though....)
  print('Running our reference Python winograd...')
  our_python_winograd_outputs = winograd_3x3(transposed_kernels.numpy(), inputs.numpy(), OUTPUT_TILE_DIM)
  print(f"{np.allclose(np.transpose(our_python_winograd_outputs, (1,0,2,3)), correct_outputs.numpy(), atol=1e-4) = }")

allclose = torch.allclose(our_winograd_outputs, correct_outputs, atol=2e-4)
max_error = torch.max(torch.abs(our_winograd_outputs - correct_outputs)).item()

if allclose:
  print(f'MATCH. All Winograd outputs are close to Torch\'s conv2d.')
else:
  print(f'{correct_outputs=}')
  print(f'{our_winograd_outputs=}')
  print(f'MISMATCH. {max_error = }. (note max_error can be misleading if the correct numbers are big.)')
