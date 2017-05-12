
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"

#include <stdio.h>
#include <math.h>

#include <glm/glm.hpp>

#include "kernel.h"

__device__ bool intersectRayTriangle(glm::vec3 o, glm::vec3 d,
  glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
  glm::vec2& uv, float& t)
{
  glm::vec3 v0v1 = v1 - v0;
  glm::vec3 v0v2 = v2 - v0;
  glm::vec3 pvec = glm::cross(d, v0v2);
  float det = glm::dot(v0v1, pvec);
#ifdef CULLING 
  // if the determinant is negative the triangle is backfacing
  // if the determinant is close to 0, the ray misses the triangle
  if (det < FLT_EPSILON) return false;
#else 
  // ray and triangle are parallel if det is close to 0
  if (fabs(det) < FLT_EPSILON) return false;
#endif 
  float invDet = 1 / det;

  glm::vec3 tvec = o - v0;
  uv.x = glm::dot(tvec, pvec) * invDet;
  if (uv.x < 0.0f || uv.x > 1.0f) return false;

  glm::vec3 qvec = glm::cross(tvec, v0v1);
  uv.y = glm::dot(d, qvec) * invDet;
  if (uv.y < 0.0f || uv.x + uv.y > 1.0f) return false;

  t = glm::dot(v0v2, qvec) * invDet;

  return true;
}

__global__ void renderPixel(cudaSurfaceObject_t surface, dim3 screenRes, dim3 blockSize, float time, mesh_t* mesh, glm::vec3 camPos, glm::mat4 viewRot)
{
    int x = blockIdx.x * blockSize.x + threadIdx.x;
    int y = blockIdx.y * blockSize.y + threadIdx.y;

    float aspect = (float)screenRes.y / screenRes.x;

    glm::vec2 normalisedCoord = 2.0f * glm::vec2(x, screenRes.y - y) / glm::vec2(screenRes.x, screenRes.y) - glm::vec2(1.0f);
    normalisedCoord.y *= aspect;

    glm::vec3 origin = camPos;
    glm::vec3 direction = glm::mat3(viewRot) * glm::normalize(glm::vec3(normalisedCoord.x, normalisedCoord.y, 1.0f));

    // test intersection with each tri
    float minT = INFINITY;
    glm::vec2 hitUV;
    int hitFace;
    for (int i = 0; i < mesh->vertexCount/3; ++i) {
      glm::vec3 a = mesh->vertices[i*3+0];
      glm::vec3 b = mesh->vertices[i*3+1];
      glm::vec3 c = mesh->vertices[i*3+2];

      glm::vec2 uv;
      float t = minT;
      intersectRayTriangle(origin, direction, a, b, c, uv, t);
      if (t < minT && t > 0.0f) {
        minT = t;
        hitFace = i;
        hitUV = uv;
      }
    }

    glm::vec4 outColour;

    if (minT != INFINITY) {
      // we got a hit. calculate the normal
      glm::vec3 a = mesh->vertices[hitFace*3+0];
      glm::vec3 b = mesh->vertices[hitFace*3+1];
      glm::vec3 c = mesh->vertices[hitFace*3+2];
      glm::vec3 normal = glm::normalize(glm::cross(b - a, b - c));
      
      outColour = glm::vec4(normal, 1.0f);
    }
    else {
      outColour = glm::vec4(normalisedCoord, 0.0f, 1.0f);
    }

    if (x < screenRes.x && y < screenRes.y)
      surf2Dwrite(float4{outColour.x, outColour.y, outColour.z, outColour.w}, surface, x * 4 * sizeof(float), y);
}

cudaError_t renderScreen(cudaSurfaceObject_t surface, dim3 screenRes, float time, mesh_t* mesh, glm::vec3 camPos, glm::mat4& viewMat)
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

    renderPixel<<<gridSize, blockSize>>>(surface, screenRes, blockSize, time, mesh, camPos, viewMat);

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
