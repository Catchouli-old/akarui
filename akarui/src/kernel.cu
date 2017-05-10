
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"

#include <stdio.h>

cudaError_t renderScreen(cudaSurfaceObject_t);

__global__ void renderPixel(cudaSurfaceObject_t surface)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    //if (y != 0)
    //printf("running thread %d %d\n", x, y);
    surf2Dwrite(float4{x/(800.0f*600.0f), x/800.0f, y/600.0f, 0.0f}, surface, x * 4 * sizeof(float), y);
}

int runkernel(cudaSurfaceObject_t cudaSurfaceObject)
{
  // Add vectors in parallel.
  cudaError_t cudaStatus = renderScreen(cudaSurfaceObject);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "renderScreen failed!");
      return 1;
  }

  return 0;
}

cudaError_t renderScreen(cudaSurfaceObject_t surface)
{
    cudaError_t cudaStatus = cudaSuccess;

    // calculate occupancy
    int blockSize, minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, renderPixel, 0, 0);

    // Launch a kernel on the GPU with one thread for each element.
    dim3 dimBlock(1, 1);
    dim3 dimGrid(800, 600);
    renderPixel<<<dimGrid, dimBlock>>>(surface);

    //// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    //// cudaDeviceSynchronize waits for the kernel to finish, and returns
    //// any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    }
    
    return cudaStatus;
}
