
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"

#include <stdio.h>
#include <math.h>

#include <glm/glm.hpp>

cudaError_t renderScreen(cudaSurfaceObject_t, dim3 screen_res);

__global__ void renderPixel(cudaSurfaceObject_t surface, dim3 screenRes, dim3 blockSize, float time)
{
    int x = blockIdx.x * blockSize.x + threadIdx.x;
    int y = blockIdx.y * blockSize.y + threadIdx.y;

    glm::vec2 resolution(blockSize.x, blockSize.y);
    glm::vec2 normalisedCoord = 2.0f * glm::vec2(x, y) / resolution - glm::vec2(1.0f);

    float tz = 0.5f + 0.5f * time * 0.1f;
    float zoo = glm::pow(0.5, (double)13.0f * tz);
    glm::vec2 c = glm::vec2(-0.05f, 0.6805f) + normalisedCoord*zoo;

    // iterate
    float di = 1.0;
    glm::vec2 z = glm::vec2(0.0);
    float m2 = 0.0;
    glm::vec2 dz = glm::vec2(0.0);
    for (int i = 0; i<300; i++)
    {
      if (m2>2048.0) { di = 0.0; break; }

      // Z' -> 2·Z·Z' + 1
      dz = 2.0f*glm::vec2(z.x*dz.x - z.y*dz.y, z.x*dz.y + z.y*dz.x) + glm::vec2(1.0, 0.0);

      // Z -> Z² + c			
      z = glm::vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;

      m2 = dot(z, z);
    }

    // distance	
    // d(c) = |Z|·log|Z|/|Z'|
    float d = 0.5*sqrt(dot(z, z) / dot(dz, dz))*log(dot(z, z));
    if (di>0.5) d = 0.0;

    // do some soft coloring based on distance
    d = glm::clamp(pow(4.0*d / zoo, 0.2), 0.0, 1.0);

    auto backgroundColour = glm::vec4(x/float(screenRes.x*screenRes.y), x/float(screenRes.x), y/float(screenRes.y), 0.0f);

    auto outColour = glm::vec4(1.0f - d);

    if (x < screenRes.x && y < screenRes.y)
      surf2Dwrite(float4{outColour.x, outColour.y, outColour.z, outColour.w}, surface, x * 4 * sizeof(float), y);
}

cudaError_t renderScreen(cudaSurfaceObject_t surface, dim3 screenRes, float time)
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

    renderPixel<<<gridSize, blockSize>>>(surface, screenRes, blockSize, time);

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
