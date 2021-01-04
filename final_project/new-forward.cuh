#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 8

//__constant__ float k[15000];

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W)
{

    /*
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
#define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
#define k4d(i3,i2,i1,i0) k[(i3) * (C * 49) + (i2)*49 + (i1)*7 + i0]

    const int H_out = H - 6;
    const int W_out = W - 6;

    int n, m, h0, w0, h_base, w_base, h, w;
    extern __shared__ float shmem[];
    int X_tile_width = TILE_WIDTH + 6;
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];
    n = blockIdx.x;
    m = blockIdx.y;
    h0 = threadIdx.x;
    w0 = threadIdx.y;
    h_base = blockIdx.z / ((H_out-1)/TILE_WIDTH+1) * (TILE_WIDTH);
    w_base = blockIdx.z % ((W_out-1)/TILE_WIDTH+1) * (TILE_WIDTH);
    h = h_base + h0;
    w = w_base + w0;
    const int base0 = h0 * X_tile_width; 


    float acc = 0;
    int c, i, j, p, q;

    for(c = 0; c < C; c++)
    {
      int base1 = base0;
      int base2 = base0;
      if((h0 < 7) && (w0 < 7))
      {
        W_shared[h0 * 7 + w0] = k4d(m, c, h0, w0);
      }

      for(i = h; i < h_base + X_tile_width; i += TILE_WIDTH)
      {
        for(j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
        {
            X_shared[base1 + (j - w_base)] = x4d(n, c, i, j);
        }
        base1 += 112;
      }
      __syncthreads();

      for(p = 0; p < 49; p += 7)
      {
        for(q = 0; q < 7; q++)
        {
            acc += X_shared[base2 + (w0 + q)] * W_shared[p+q];
        }
        base2 += 14;
      }
      __syncthreads();
    }

    if (h < H_out && w < W_out) {
     y4d(n, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   We only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) 
    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    /*printf("Value of B: %d\n", B);
    printf("Value of M: %d\n", M);
    printf("Value of C: %d\n", C);
    printf("Value of H: %d\n", H);
    printf("Value of W: %d\n", W);
    printf("Value of K: %d\n", K);*/

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_grid = ((W_out ) / TILE_WIDTH+1);
    int H_grid = ((H_out ) / TILE_WIDTH+1);
    int dimz = H_grid * W_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, dimz);

    size_t mem = sizeof(float)*((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+K*K);
    //cudaMemcpyToSymbol(k, w.dptr_, M * K * K * C * sizeof(float));

    //MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    forward_kernel<<<gridDim, blockDim, mem>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed");
}
}
}

#endif
