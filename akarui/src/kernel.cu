
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"

#include <stdio.h>
#include <math.h>

cudaError_t renderScreen(cudaSurfaceObject_t, dim3 screen_res);

__global__ void renderPixel(cudaSurfaceObject_t surface, dim3 screenRes, dim3 blockSize)
{
    int x = blockIdx.x * blockSize.x + threadIdx.x;
    int y = blockIdx.y * blockSize.y + threadIdx.y;
    //if (y != 0)
    //printf("running thread %d %d\n", x, y);
    float4 pixel = {x/float(screenRes.x*screenRes.y), x/float(screenRes.x), y/float(screenRes.y), 0.0f};
    if (x < screenRes.x && y < screenRes.y)
      surf2Dwrite(pixel, surface, x * 4 * sizeof(float), y);
}

int runkernel(cudaSurfaceObject_t cudaSurfaceObject, dim3 screen_res)
{
  // Add vectors in parallel.
  cudaError_t cudaStatus = renderScreen(cudaSurfaceObject, screen_res);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "renderScreen failed!");
      return 1;
  }

  return 0;
}

cudaError_t renderScreen(cudaSurfaceObject_t surface, dim3 screenRes)
{
    cudaError_t cudaStatus = cudaSuccess;

    // calculate occupancy
    int recBlockSize, minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &recBlockSize, renderPixel, 0, 0);

    // Convert block size to 2d
    dim3 blockSize(1, recBlockSize);
    while (blockSize.x < blockSize.y) {
      blockSize.x *= 2;
      blockSize.y /= 2;
    }

    dim3 gridSize(int(ceil(screenRes.x/float(blockSize.x))), int(ceil(screenRes.y/float(blockSize.y))));

    renderPixel<<<gridSize, blockSize>>>(surface, screenRes, blockSize);

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
