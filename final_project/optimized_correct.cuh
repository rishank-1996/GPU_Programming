#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 8

__constant__ float k[15000];

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x,/* const float *k,*/ const int B, const int M, const int C, const int H, const int W)
{

    /*
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
    const int H_out = H - 6;
    const int W_out = W - 6;

#define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
#define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
#define k4d(i3,i2,i1,i0) k[(i3) * (C * 49) + (i2)*49 + (i1)*7 + i0]

    const int X_tile_width = TILE_WIDTH + 6;
    const int x_tile_tile_width = X_tile_width * TILE_WIDTH;

    // shared memory array
    extern __shared__ float shmem[];

    // shared memory for input X and weights K
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];

    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int h0 = threadIdx.x;
    const int w0 = threadIdx.y;
    const int h_base = blockIdx.z / ((H_out-1)/TILE_WIDTH+1) * (TILE_WIDTH);
    const int w_base = blockIdx.z % ((W_out-1)/TILE_WIDTH+1) * (TILE_WIDTH);
    const int h = h_base + h0;
    const int w = w_base + w0;

    float acc = 0;

    int c, i, j, p, q;

    const int h0temp = h0 * X_tile_width;
    const int k2 = 49;

    for(c = 0; c < C; c++)
    {
      if((h0 < 7) && (w0 < 7))
      {
        W_shared[h0 * 7 + w0] = k4d(m, c, h0, w0);
      }

      //__syncthreads();

      int itemp = h0temp;
      for(i = h; i < h_base + X_tile_width; i += TILE_WIDTH, itemp += x_tile_tile_width)
      {
        for(j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
        {
            X_shared[itemp + (j - w_base)] = x4d(n, c, i, j);
        }
      }

      __syncthreads();

      int ptemp = h0temp;
      for(p = 0; p < k2; p += 7, ptemp += X_tile_width)
      {
        for(q = 0; q < 7; q++)
        {
            acc += X_shared[ptemp + (w0 + q)] * W_shared[p+q];
        }
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
 int n, m, h0, w0, h_base, w_base, h ,w;
 int X_tile_width = TILE_WIDTH + K - 1;
 const int H_out = H - K + 1;
 const int W_out = W - K + 1;
 int W_grid = W_out/TILE_WIDTH; 
 int H_grid = H_out/TILE_WIDTH;
 extern __shared__ float shmem[];
 float* X_shared = &shmem[0];
 float* W_shared = &shmem[X_tile_width * X_tile_width];
 n = blockIdx.x;
 m = blockIdx.y;
 h0 = threadIdx.x;
 w0 = threadIdx.y;
 h_base = (blockIdx.z / W_grid) * TILE_WIDTH; 
 w_base = (blockIdx.z % W_grid) * TILE_WIDTH; 
 h = h_base + h0;
 w = w_base + w0;

 float acc = 0;
 int c, i, j, p ,q;

 for(c = 0; c < C; c++){
    if((h0 < K) && (w0 < K)) {
        W_shared[h0, w0] = k4d[m, c, h0, w0];
    }
    __syncthreads();

    for(i = h; i < h_base + X_tile_width; i += TILE_WIDTH){
        for(j = w; j < w_base + X_tile_width; j += TILE_WIDTH)
            X_shared[i - h_base, j - w_base] = x4d[n, c, h, w];
    }
    __syncthreads();

    for(p = 0; p < K; p++){
        for(q = 0; q< K; q++)
            acc = acc + (X_shared[h + p, w + q] * W_shared[p,q]); 
    }
    __syncthreads();
 }
 y4d[n, m, h, w] = acc;
*/


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
    //cudaStream_t s = y.stream_->stream_;

    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int W_grid = ((W_out ) / TILE_WIDTH+1);
    int H_grid = ((H_out ) / TILE_WIDTH+1);

    int Z = H_grid * W_grid;

    // Set the kernel dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    // Declare shared memory variable (FOR SECOND KERNEL)
    size_t shmem_size = sizeof(float)*((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+K*K);
    cudaMemcpyToSymbol(k, w.dptr_, M * K * K * C * sizeof(float));

    //MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    //forward_kernel<<<gridDim, blockDim, shmem_size, s>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);
    //forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W);
    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_, x.dptr_, B, M, C, H, W);
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
