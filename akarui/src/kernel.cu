
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"

#include <stdio.h>
#include <math.h>

#include <glm/glm.hpp>

__device__ glm::vec4 intersect(glm::vec3 a, glm::vec3 b, glm::vec3 c,
  glm::vec3 o, glm::vec3 d, float minT,
  float mat_index, glm::vec4 lasthit)
{
  glm::vec3 e1 = b - a;
  glm::vec3 e2 = c - a;
  glm::vec3 p = cross(d, e2);
  float det = dot(p, e1);
  bool isHit = det > 0.00001f;
  float invdet = 1.0f / det;
  glm::vec3 tvec = o - a;
  float u = dot(p, tvec) * invdet;
  glm::vec3 q = cross(tvec, e1);
  float v = dot(q, d) * invdet;
  float t = dot(q, e2) * invdet;
  isHit = (u >= 0.0f) && (v >= 0.0f)
    && (u + v <= 1.0f)
    && (t < lasthit.z)
    && (t > minT);
  return isHit ? glm::vec4(u, v, t, mat_index) : lasthit;
}

__global__ void renderPixel(cudaSurfaceObject_t surface, dim3 screenRes, dim3 blockSize, float time, glm::vec3* vertexBuf, int vertexCount)
{
    int x = blockIdx.x * blockSize.x + threadIdx.x;
    int y = blockIdx.y * blockSize.y + threadIdx.y;

    glm::vec2 normalisedCoord = 2.0f * glm::vec2(x, y) / glm::vec2(screenRes.x, screenRes.y) - glm::vec2(1.0f);

    glm::vec3 origin(0.0f, 0.0f, -10.0f);
    glm::vec3 direction = glm::normalize(glm::vec3(normalisedCoord.x, normalisedCoord.y, 1.0f));

    // test intersection with each tri
    glm::vec4 lastHit(0.0f, 0.0f, INFINITY, 0.0f);
    for (int i = 0; i < vertexCount; i += 3) {
      glm::vec3 a = vertexBuf[i*3+0];
      glm::vec3 b = vertexBuf[i*3+1];
      glm::vec3 c = vertexBuf[i*3+2];
      lastHit = intersect(a, b, c, origin, direction, 0.0f, 0.0f, lastHit);
      if (lastHit.z != INFINITY)
        break;
    }

    glm::vec4 outColour;

    if (lastHit.z != INFINITY) {
      outColour = glm::vec4(1.0f);
    }
    else {
      outColour = glm::vec4(normalisedCoord, 0.0f, 1.0f);
    }

    /*
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
    */

    if (x < screenRes.x && y < screenRes.y)
      surf2Dwrite(float4{outColour.x, outColour.y, outColour.z, outColour.w}, surface, x * 4 * sizeof(float), y);
}

cudaError_t renderScreen(cudaSurfaceObject_t surface, dim3 screenRes, float time, glm::vec3* vertexBuf, int vertexCount)
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

    renderPixel<<<gridSize, blockSize>>>(surface, screenRes, blockSize, time, vertexBuf, vertexCount);

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
