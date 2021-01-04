
#include <wb.h>
#define TILE_WIDTH  32

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  //Identify the row and column of the p_element to work on
  //int Row = blockIdx.y * blockDim.y + threadIdx.y;
  //int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = by * TILE_WIDTH +ty;
  int Col = bx * TILE_WIDTH +tx;
  float Pvalue = 0;
  int Width =  (numAColumns % TILE_WIDTH == 0) ? numAColumns : numAColumns + TILE_WIDTH - 1 ;


  //Loop over the A and B tiles required to compute the P element
  for (int m = 0; m < (Width/TILE_WIDTH); ++m){
    //Collaborative loading of M and N tiles into shared memory
    //subTileA[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
    //subTileB[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
    if(Row < numARows && (m*TILE_WIDTH + tx) < numAColumns)
      subTileA[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH + tx];
    else
      subTileA[ty][tx] = 0;
    if(Col < numBColumns && (m*TILE_WIDTH + ty) < numBRows)
      subTileB[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
    else
      subTileB[ty][tx] = 0;
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) 
      Pvalue += subTileA[ty][k] * subTileB[k][tx];
    __syncthreads();
  }
  if(Row < numCRows && Col < numCColumns)
    C[Row * numCColumns + Col] = Pvalue;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)


  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  int sizeC = numCRows * numCColumns * sizeof(float);

  //@@ Allocate the hostC matrix
  hostC = (float *) malloc(sizeC);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **) &deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **) &deviceC, numCRows * numCColumns * sizeof(float));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int gridx = (numARows % TILE_WIDTH == 0) ? 
              numARows/TILE_WIDTH : 
              (numARows + TILE_WIDTH - 1)/TILE_WIDTH;
  int gridy = (numAColumns % TILE_WIDTH == 0) ? 
              numAColumns/TILE_WIDTH : 
              (numAColumns + TILE_WIDTH - 1)/TILE_WIDTH;
  int big;
  if (gridx >= gridy)
    big = gridx;
  else
    big = gridy;            
  
  dim3 dimGrid(big + 1, big+1);  
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
    numARows, numAColumns, numBRows,
    numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
