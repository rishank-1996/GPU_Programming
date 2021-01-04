// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>

//template <unsigned int BLOCK_SIZE> 32 //@@ You can change this

#define BLOCK_SIZE 512

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void total(float *input, float *output, int len) {

  //not using volatile causes the threads to not work together (compiler oprimization)
  //could also resolve this using syncthreads between the statements of the unrolled loop
  volatile __shared__ int sdata[BLOCK_SIZE];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(BLOCK_SIZE*2) + tid;
  unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;
  sdata[tid] = 0;
  //@@ Load a segment of the input vector into shared memory
  //load two elements at once and perform addition
  while (i < len) {
   sdata[tid] += input[i] + input[i+BLOCK_SIZE];
   i += gridSize;
  }
  __syncthreads();
  //@@ Traverse the reduction tree
  if (BLOCK_SIZE >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();}
  if (BLOCK_SIZE >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();}
  if (BLOCK_SIZE >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();}
  if (BLOCK_SIZE >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();}
  if (tid < 32) {
    if (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32];
    //__syncthreads(); //dont require these anymore using the volatile keyword for shared memory!
    if (BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16]; 
    //__syncthreads();
    if (BLOCK_SIZE >= 16) sdata[tid] += sdata[tid + 8];
    //__syncthreads();
    if (BLOCK_SIZE >= 8) sdata[tid] += sdata[tid + 4];
    //__syncthreads();
    if (BLOCK_SIZE >= 4) sdata[tid] += sdata[tid + 2];
    //__syncthreads();
    if (BLOCK_SIZE >= 2) sdata[tid] += sdata[tid + 1];
    //__syncthreads();
  }

  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  wbLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void **) &deviceOutput, numOutputElements * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil( (ceil(numInputElements / (float) BLOCK_SIZE)) / 2) ); 
  //^only half the number of blocks are required since two are added together while loading 
  dim3 dimBlock(BLOCK_SIZE);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  total<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += (int)hostOutput[ii];
  }

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
