// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//Float to CHAR conversion
__global__ void floattochar(float *i_Image, unsigned char *o_Image, int len) {
 int tid = blockIdx.x * blockDim.x + threadIdx.x;
 int stride = gridDim.x * blockDim.x;
 for(int i = tid; i < len; i += stride){
    o_Image[i] = (unsigned char) (255 * i_Image[i]);
  }
}

//GRAYSCALE conversion
__global__ void converttograyscale(unsigned char *i_Image, unsigned char *o_Image, 
                                    int width, int height,  int channel) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int col = tx + bx * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  for(int i = col; i < width * height; i += stride) {
    int rgboffset = col * channel;
    unsigned char r = i_Image[rgboffset];
    unsigned char g = i_Image[rgboffset+1];
    unsigned char b = i_Image[rgboffset+2];
    o_Image[col] = (unsigned char)(0.21f*r + 0.71f*g + 0.07f*b);
  }
}

//HISTOGRAM creation
__global__ void histogramAdd( unsigned char *Input, uint *Output, 
                              int width, int height) {
  const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  const int numThreads = blockDim.x * gridDim.x;

  __shared__ uint s_hist[HISTOGRAM_LENGTH];
  for(int pos = threadIdx.x; pos < HISTOGRAM_LENGTH; pos += blockDim.x){
    s_hist[pos] = 0;
  }
  __syncthreads();

  for(int pos = globalTid; pos < width*height; pos += numThreads){
    uint data4 = Input[pos];
    atomicAdd (s_hist + ((data4 >> 0) & 0xFFU), 1);
    // atomicAdd (s_hist + (data4 >> 8) & 0xFFU, 1);
    // atomicAdd (s_hist + (data4 >> 16) & 0xFFU, 1);
    // atomicAdd (s_hist + (data4 >> 24) & 0xFFU, 1);
  }
  __syncthreads();
  for(int pos = threadIdx.x; pos < HISTOGRAM_LENGTH; pos+=blockDim.x)
    atomicAdd(Output + pos,  s_hist[pos]);
}

//probability function
__device__ float prob(uint hist_value, int width, int height) {
  return ((float)hist_value / (width * height)); 
}

//Calculate CDF of histogram
__global__ void parallelscan(uint *Input, float *Output, 
                              int width, int height){
  const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;
  volatile __shared__ float sdata[HISTOGRAM_LENGTH];
  if(globalTid <256){
   sdata[tid] = prob(Input[tid], width, height);
  }
  for(int d = 1; d <  HISTOGRAM_LENGTH; d*=2){
    __syncthreads();      
    float temp = (tid >= d) ? sdata[tid - d] : 0;
    sdata[tid] += temp;
  }

  __syncthreads();
  if(globalTid < 256) {
    Output[globalTid] = sdata[globalTid];
  }
}

//clamp function
__device__ unsigned char clamp(float x, float start, float end) {
  return min(max(x, start), end);
}

//color correction function
__device__ unsigned char correct_color(unsigned char val, float *cdf){
  return clamp(255 * ((cdf[val] - cdf[0])/ (1.0 - cdf[0])), 0, 255.0);
}

// Histogram Equalization
__global__ void hist_eq (float * cdf_array, unsigned char *input, int len) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int numthreads = blockDim.x*gridDim.x;
  for(int k = tid; k < len; k += numthreads) {
    input[tid] = correct_color(input[tid], cdf_array);
  }
}

// CHAR to float conversion
__global__ void chartofloat (unsigned char *input, float *output, int len) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;

  for (int ii = tid; ii < len; ii+=gridDim.x*blockDim.x) {
    if (ii < len) {
        output[ii] = (float) (input[ii]/255.0);
    }
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *deviceInputImage;
  unsigned char *ucharImage_out;
  unsigned char *grayScale_out;
  uint *hist_image_out;
  float *cdf_image_out;
  float *deviceOutputImage;

  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  int numInputElements;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  numInputElements = imageWidth * imageHeight * imageChannels;

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceInputImage, imageHeight * imageWidth * imageChannels * sizeof(float));
  cudaMalloc((void **) &ucharImage_out,imageHeight * imageWidth * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &grayScale_out, imageHeight * imageWidth * sizeof(unsigned char));
  cudaMalloc((void **) &hist_image_out, HISTOGRAM_LENGTH * sizeof(uint));
  cudaMalloc((void **) &cdf_image_out, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **) &deviceOutputImage, imageHeight * imageWidth * imageChannels * sizeof(float));

  //@@ insert code here
  cudaMemcpy(deviceInputImage, hostInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numInputElements / (float) HISTOGRAM_LENGTH) ); 
  dim3 dimBlock(HISTOGRAM_LENGTH);


  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  floattochar <<<dimGrid, dimBlock>>>        (deviceInputImage, ucharImage_out, numInputElements);  
  cudaDeviceSynchronize();

  converttograyscale <<<dimGrid, dimBlock>>> (ucharImage_out, grayScale_out, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  
  histogramAdd <<<dimGrid, dimBlock>>>       (grayScale_out, hist_image_out, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  parallelscan <<<dimGrid, dimBlock>>>       (hist_image_out, cdf_image_out, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  hist_eq <<<dimGrid, dimBlock>>>            (cdf_image_out, ucharImage_out, numInputElements);
  cudaDeviceSynchronize();
  
  chartofloat <<<dimGrid, dimBlock>>>        (ucharImage_out, deviceOutputImage, numInputElements);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutputImage, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImage);
  cudaFree(ucharImage_out);
  cudaFree(grayScale_out);
  cudaFree(hist_image_out);
  cudaFree(cdf_image_out);
  cudaFree(deviceOutputImage);

  
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
