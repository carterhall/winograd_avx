import numpy as np
import math
import torch; import torch.nn as nn; import torch.nn.functional as F


def naive_conv(kernels, inputs):
  assert kernels.shape[-1] == 3 and kernels.shape[-2] == 3

  K = kernels.shape[0]
  N, C, H, W = inputs.shape
  result = np.zeros((N, K, H, W))
  
  for n in range(N):              # Batch dimension (number of images)
    for ikernel in range(K):      # Which kernel
      for iy in range(1, H-1):         # Image height index
        for ix in range(1, W-1):       # Image width index
          acc = 0.0
          for c in range(C):            # Number of channels per image
            kernel = kernels[ikernel][c]
            for ky in range(-1,2):     # Kernel height index
              for kx in range(-1,2):   # Kernel width index
                kernel_val = kernel[ky + 1][kx + 1]
                image_val  = inputs[n][c][iy + ky][ix + kx]
                acc += kernel_val*image_val

          result[n][ikernel][iy][ix] = acc

  return result 


# 2x2 is output tile size
# 3x3 is kernel size
def winograd_3x3(kernels, inputs, m=2):
  #print(f'reference winograd: {kernels.shape=}, {inputs.shape=}')
  #print(f'reference winograd: {kernels=}, {inputs=}')

  # kernels is shape (K,C,3,3)
  C, N, H, W = inputs.shape
  K = kernels.shape[0]
  #N, C, H, W = inputs.shape

  # Trying to stick to the variable names of the paper....
  assert m == 2 or m == 4
  r = 3
  
  if m == 2:
    B_T = np.array([
      [1, 0, -1, 0],
      [0, 1, 1, 0],
      [0, -1, 1, 0],
      [0, 1, 0, -1]
    ], dtype=np.float32)
    B = np.transpose(B_T)

    G = np.array([
      [1, 0, 0],
      [0.5, 0.5, 0.5],
      [0.5, -0.5, 0.5],
      [0, 0, 1]
    ], dtype=np.float32)
    G_T = np.transpose(G)

    A_T = np.array([
      [1, 1, 1, 0],
      [0, 1, -1, -1]
    ], dtype=np.float32)
    A = np.transpose(A_T)

  elif m == 4:
    B_T = np.reshape(np.array([
      4.0, 0, -5, 0, 1, 0,
      0, -4, -4, 1, 1, 0,
      0, 4, -4, -1, 1, 0,
      0, -2, -1, 2, 1, 0,
      0, 2, -1, -2, 1, 0,
      0, 4, 0, -5, 0, 1,
    ], dtype=np.float32), (6,6))
    B = np.transpose(B_T)

    G = np.reshape(np.array([
      1/4, 0, 0,
      -1/6, -1/6, -1/6,
      -1/6, 1/6, -1/6,
      1/24, 1/12, 1/6,
      1/24, -1/12, 1/6,
      0, 0, 1
    ], dtype=np.float32), (6,3))
    G_T = np.transpose(G)

    A_T = np.reshape(np.array([
      1, 1, 1, 1, 1, 0,
      0, 1, -1, 2, -2, 0,
      0, 1, 1, 4, 4, 0,
      0, 1, -1, 8, -8, 1,
    ], dtype=np.float32), (4,6))
    A = np.transpose(A_T)

  # P is number of input image tiles per channel
  P = N*math.ceil(H/m)*math.ceil(W/m)    # Don't do integer division if we want to ceil!
  alpha = m + r - 1   # input image tile dim, always 4 in this case

  # Note: in the paper, the x and y with squiggly, we'll call them ~x and ~y, are *tile*
  # coordinates, not single-element indices.

  #D = np.transpose(inputs, (1,0,2,3))  # Transpose to put C on the outside
  D = inputs  # Pre-transposing the input now to match other impl

  # Resist the urge to reshape this to be cleanly indexed by b = 0..P-1. Remember that the tiles overlap!
  # So indexing is different than the actual amount of data there.
  D = np.reshape(D, (C,N*H,W))   # Flatten the batch and height dims to make indexing easier


  # Deviating from the symbols in the paper:
  # tx and ty take the places of xi and nu (squiggly E, leaning v)
  
  U = np.zeros((alpha,alpha,K,C))  # K,C last for eventual matmul by C,P..?
  for k in range(0, K):
    for c in range(0, C):
      g_kc = kernels[k][c]   # r by r
      u = G @ g_kc @ G_T   # For F(2x2, 3x3), this u matrix will be 4x4
      for tx in range(alpha):  
        for ty in range(alpha):
          U[tx][ty][k][c] = u[tx][ty]  # "Scatter u to matrices U"
  print('U:', U)
  
  # The P dimension lets us index in with b (0..P-1) and get an input tile.
  # Of course, it seems that the input tile is actually stored in the earlier dimensions, and we do the 
  # matmul with dimensions C and P.
  V = np.zeros((alpha,alpha,C,P))

  # This runs range(0, P) in the paper, but the l
  for b in range(0, P):  
    for c in range(0, C):
      # d_cb = input tile b in channel c (4x4 for this F(2x2, 3x3)
      # where b is just a linear index through the multi dim tensor (N,H,W)

      # This indexing seems overly complicated? It's just selecting one tile out of the big 
      # matrix that is (N*H, W).
      ic = (b // ((N*H)//m)) * m
      ir = (b %  ((N*H)//m)) * m

      # The following block is just doing:
      #   d_cb = D[c][ir:ir+alpha, ic:ic+alpha]
      # but safely handles the case where we have to zero-pad the input tile.

      d_cb = np.zeros((alpha,alpha))
      rbound = min(alpha, N*H-ir)
      cbound = min(alpha, W-ic)
      #print(f'{alpha=}, {N*H=}, {W=}, {ir=}, {ic=}, {rbound=}, {cbound=}')
      for tx in range(rbound):
        for ty in range(cbound):
          #Y[k][ir+tx][ic+ty] = Ypart[tx][ty]
          d_cb[tx][ty] = D[c][ir+tx][ic+ty]
          #d_cb[tx][ty] = D[c][tx][ty]

      #print(f'{b=}, {c=}, tile:')
      #print(d_cb)
      '''
      # Leaving this less-elegant version in, commented out, in case we 
      # somehow were wrong about the above.
      d_cb = np.zeros((alpha,alpha))
      unpadded_d_cb = D[c][ir:ir+alpha, ic:ic+alpha]
      if ir > (H*N) - alpha or ic > W - alpha: 
        print(f'at index {b=}, {ir=}, {ic=}, we think we have to pad.')
      if unpadded_d_cb.shape != d_cb.shape:
        print(f'at index {b=}, we had to pad. {P=}, {W=}, {H*N=}, {ic=}, {ir=}, {alpha=}, {m=}.')
        print(f'  {unpadded_d_cb.shape=}')
      d_cb[:unpadded_d_cb.shape[0], :unpadded_d_cb.shape[1]] = unpadded_d_cb
      '''

      v = B_T @ d_cb @ B   # For F(2x2, 3x3), this v matrix is also 4x4
      for tx in range(alpha):
        for ty in range(alpha):
          V[tx][ty][c][b] = v[tx][ty]  # "Scatter v to matrices V"
  print('V:', V)

  M = np.zeros((alpha,alpha,K,P))
  for tx in range(0, alpha):  
    for ty in range(0, alpha):
      M[tx][ty] = U[tx][ty] @ V[tx][ty]  # The cheeky matrix multiply doing the bulk of the work

  Y = np.zeros((K,N,H,W))   # Here are the actual dims, with batch and kernel already flipped
  Y = np.reshape(Y, (K, N*H,W))  # Flatten batch and height dims
  
  for k in range(0, K):
    for b in range(0, P):
      Mpart = np.zeros((alpha,alpha)) 
      for tx in range(0, alpha):
        for ty in range(0, alpha):
          Mpart[tx][ty] = M[tx][ty][k][b]   # "Gather m from matrices M"

      # This indexing seems overly complicated? It's just selecting one tile out of the big 
      # matrix that is (N*H, W).
      ir = (b %  ((N*H)//m)) * m
      ic = (b // ((N*H)//m)) * m
      Ypart = (A_T @ Mpart @ A)
      #print(f'{Ypart.shape=}, {A_T.shape=}, {Mpart.shape=}')
      #Y[k][ir:ir+m, ic:ic+m] = Ypart

      rbound = min(m, N*H-ir)
      cbound = min(m, W-ic)
      for tx in range(rbound):
        for ty in range(cbound):
          Y[k][ir+tx][ic+ty] = Ypart[tx][ty]

  Y = np.reshape(Y, (K,N,H,W))  # Undo the previous flattening
  #Y = np.transpose(Y, (1,0,2,3))  # Finally flip the kernel and batch dims (do we want this?)
  return Y[:,:,:-2,:-2]  # Drop the last two rows and columns to match Torch padding='valid' or none


if __name__ == '__main__':
  # Note: it seems pretty likely that Torch conv2d is using the 2x2,3x3 version, because 
  # we seem to get nearly exactly the same results even for large param sizes. The errors
  # are larger if we use the 4x4,3x3 version, so that might trip np.allclose for large params.

  N = 1  # Batch size
  C = 1  # Channels (maybe RGB, maybe more abstract features)
  H = 6 # Height of image
  W = 6 # Width of image

  K = 1  # Number of filters per channel.
         # I'm sticking with the relatively simple case here where any given kernel
         # operates on only one channel, but it seems like that's not always the case..?
         # I'm a conv2d noob.

  #inputs = np.random.randn(N, C, H, W)
  inputs = np.reshape(np.arange(N*C*H*W, dtype=np.float32), (C,N,H,W))

  # K=filts per chan, C = chans
  kernels = np.reshape(np.arange(K*C*3*3, dtype=np.float32), (K,C,3,3))

  #kernels = np.random.randn(K, C, 3, 3).astype('f')
  #kerns = [np.reshape(np.array([0.,0,0,0,1,0,0,0,1]), (3,3))]*K*C
  #kernels = np.reshape(np.array(kerns), (K,C,3,3))


  #naive_outputs = naive_conv(kernels, inputs)
  print(f'{inputs = }')
  print(f'{kernels = }')
  #print(f'{naive_outputs = }')
  #print(f'{np.allclose(naive_outputs, inputs) = }')  # Sketchy test of identity kernel

  #print(f'{inputs = }')
  #print(f'{outputs = }')

  winograd_outputs = winograd_3x3(kernels, inputs, m=2)
  print(f'{winograd_outputs = }')


  tinputs = torch.from_numpy(np.transpose(inputs, (1,0,2,3)))
  tkernels = torch.from_numpy(kernels)
  #toutputs = F.conv2d(tinputs, tkernels, padding='same')
  #toutputs = F.conv2d(tinputs, tkernels).numpy()
  toutputs = np.transpose(F.conv2d(tinputs, tkernels).numpy(), (1,0,2,3))
  print(f'torch {toutputs = }')


  print()
  print(f'All winograd and torch outputs close? {np.allclose(toutputs, winograd_outputs)}')
  print(f'Mean squared error in case we barely tripped np.allclose: {np.mean((toutputs - winograd_outputs)**2)}')

