#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MAX_MASK_WIDTH 3
#define TILE_WIDTH  4
#define TOTAL_WIDTH   (TILE_WIDTH + MAX_MASK_WIDTH - 1)  //including halo and ghost elements

//@@ Define constant memory for device kernel here
__constant__ float Mc[MAX_MASK_WIDTH*MAX_MASK_WIDTH*MAX_MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                        const int y_size, const int x_size) {

  // Insert kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int row_op    =   blockIdx.y * TILE_WIDTH + ty;
    int col_op    =   blockIdx.x * TILE_WIDTH + tx;
    int height_op =   blockIdx.z * TILE_WIDTH + tz;

    //translation
    int row_ip    =  row_op - MAX_MASK_WIDTH/2;
    int col_ip    =  col_op - MAX_MASK_WIDTH/2;
    int height_ip =  height_op - MAX_MASK_WIDTH/2;

    __shared__ float N_ds[TOTAL_WIDTH][TOTAL_WIDTH][TOTAL_WIDTH];

   if((row_ip >= 0 && row_ip < y_size) && (col_ip >= 0 && col_ip < x_size) && (height_ip >= 0 && height_ip < z_size) ){
     N_ds[tz][ty][tx] = input[col_ip + row_ip * x_size + height_ip * y_size * x_size];
   }
   else{
     N_ds[tz][ty][tx] = 0;  //0.0f
   }
    //N_ds[tz][ty][tx] = 1.0f;
    __syncthreads();
    
    float p_val = 0.0f;   //0.0f

    //Actual kernel multiplication
    if(ty < TILE_WIDTH && tx < TILE_WIDTH && tz < TILE_WIDTH){

      for(int z=0; z < MAX_MASK_WIDTH; ++z){
        for(int y=0; y < MAX_MASK_WIDTH; ++y){
          for(int x=0; x < MAX_MASK_WIDTH; ++x){
            p_val += N_ds[z+tz][y+ty][x+tx] * Mc[x + (y * MAX_MASK_WIDTH)
                          + (z * MAX_MASK_WIDTH * MAX_MASK_WIDTH) ];
          }
        }
      }

    if(row_op < y_size && col_op < x_size && height_op < z_size){
      output[col_op + (row_op * x_size) + (height_op * x_size * y_size)]= p_val;
    }
  }
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  float realLength = inputLength - 3;
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, realLength * sizeof(float));
  cudaMalloc((void **) &deviceOutput, realLength * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpyToSymbol(Mc, hostKernel, 27 * sizeof(float));
  cudaMemcpy(deviceInput, &hostInput[3], realLength * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize the grid and block dimensions here
  // int gridx = (x_size % TILE_WIDTH == 0) ? 
  //             x_size/TILE_WIDTH : 
  //             (x_size + TILE_WIDTH - 1)/TILE_WIDTH;
  // int gridy = (y_size % TILE_WIDTH == 0) ? 
  //             y_size/TILE_WIDTH : 
  //             (y_size + TILE_WIDTH - 1)/TILE_WIDTH;
  // int gridz = (z_size % TILE_WIDTH == 0) ? 
  //             z_size/TILE_WIDTH : 
  //             (z_size + TILE_WIDTH - 1)/TILE_WIDTH;
  // int big;
  // if (gridx >= gridy && gridx >= gridz)
  //   big = gridx;
  // else if (gridy >= gridx && gridy >= gridz)
  //   big = gridy;
  // else
  //   big = gridz;     
  
  dim3 dimGrid(ceil(x_size / (float)TILE_WIDTH), ceil(y_size / (float)TILE_WIDTH), ceil(z_size / (float)TILE_WIDTH));  
  dim3 dimBlock(TOTAL_WIDTH, TOTAL_WIDTH, TOTAL_WIDTH);
  
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");
  cudaMemcpy( &hostOutput[3], deviceOutput, realLength * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
