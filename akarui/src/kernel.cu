
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

__device__ bool rayIntersectSphere(glm::vec3 origin, glm::vec3 direction, glm::vec4 sphere, glm::vec2& intersectionPoints) {
  float c = glm::length(origin - glm::vec3(sphere)) - sphere.w*sphere.w;
  float dotVal = glm::dot(direction, (origin - glm::vec3(sphere)));
  float sqrtVal = dotVal*dotVal - glm::dot(origin - glm::vec3(sphere), origin - glm::vec3(sphere)) + sphere.w*sphere.w;
  if (sqrtVal <= 0.0) {
    return false;
  }
  if (sqrtVal == 0.0) {
    intersectionPoints.x = -dotVal;
    intersectionPoints.y = -dotVal;
    return true;
  }
  else {
    float d1 = -(dotVal)+sqrt(sqrtVal);
    float d2 = -(dotVal)-sqrt(sqrtVal);
    if (d1 < 0.0 && d2 < 0.0) {
      return false;
    }
    else if (d1 < 0.0 || d2 < 0.0) {
      intersectionPoints.x = glm::min(d1, d2);
      intersectionPoints.y = glm::max(d1, d2);
      return true;
    }
    else {
      intersectionPoints.x = glm::min(d1, d2);
      intersectionPoints.y = glm::max(d1, d2);
      return true;
    }
  }
}

__device__ void traceScene(glm::vec3 origin, glm::vec3 dir, const Scene* scene, float& t, glm::vec2& hitUV, int& hitFace, const Mesh*& hitMesh, const Material*& hitMat)
{
  for (int meshIdx = 0; meshIdx < scene->meshCount; ++meshIdx) {
    Mesh* mesh = scene->meshes[meshIdx];
    for (int tri = 0; tri < mesh->idxCount/3; ++tri) {
      int a = mesh->idx[tri*3];
      int b = mesh->idx[tri*3+1];
      int c = mesh->idx[tri*3+2];

      glm::vec3 v0 = mesh->pos[a];
      glm::vec3 v1 = mesh->pos[b];
      glm::vec3 v2 = mesh->pos[c];

      glm::vec2 uv;
      float hitT = INFINITY;
      intersectRayTriangle(origin, dir, v0, v1, v2, uv, hitT);
      if (hitT != INFINITY && hitT < t && hitT >= 0.0f) {
        t = hitT;
        hitFace = tri;
        hitUV = uv;
        hitMesh = mesh;
        hitMat = &scene->defaultMat;
      }
    }
  }
}

__global__ void renderPixel(cudaSurfaceObject_t surface, dim3 screenRes, dim3 blockSize, float time, Scene* scene, glm::vec3 camPos, glm::mat4 viewRot)
{
    int x = blockIdx.x * blockSize.x + threadIdx.x;
    int y = blockIdx.y * blockSize.y + threadIdx.y;

    float aspect = (float)screenRes.y / screenRes.x;

    glm::vec2 normalisedCoord = 2.0f * glm::vec2(x, screenRes.y - y) / glm::vec2(screenRes.x, screenRes.y) - glm::vec2(1.0f);
    normalisedCoord.y *= aspect;

    glm::vec3 origin = camPos;
    glm::vec3 direction = glm::mat3(viewRot) * glm::normalize(glm::vec3(normalisedCoord.x, normalisedCoord.y, -1.0f));

    // test intersection with each tri
    float minT = INFINITY;
    glm::vec2 hitUV = glm::vec2(0.0f);
    int hitFace = -1;
    const Mesh* hitMesh = nullptr;
    const Material* hitMat = &scene->defaultMat;

    // raytrace the lights for debug drawing
    for (int i = 0; i < scene->lightCount; ++i) {
      Light* light = &scene->lights[i];
      if (light->type == Light::Type_Point) {
        glm::vec2 hit;
        if (rayIntersectSphere(origin, direction, glm::vec4(light->pos + glm::vec3(sin(time), cos(time), 0.0f) * 0.5f, 0.1f), hit))
          minT = glm::min(hit.x, minT);
      }
    }

    // raytrace the scene
    traceScene(origin, direction, scene, minT, hitUV, hitFace, hitMesh, hitMat);

    glm::vec4 outColour;

    if (minT != INFINITY) {
      glm::vec3 normal = glm::vec3(1.0f);

      if (hitMesh != nullptr && hitFace >= 0) {
        // we got a hit. calculate the normal
        int a = hitMesh->idx[hitFace * 3 + 0];
        int b = hitMesh->idx[hitFace * 3 + 1];
        int c = hitMesh->idx[hitFace * 3 + 2];
        glm::vec3 v0(hitMesh->pos[a]), v1(hitMesh->pos[b]), v2(hitMesh->pos[c]);
        normal = glm::normalize(glm::cross(v1 - v0, v1 - v2));
      }

      glm::vec3 hitPoint = origin + minT * direction;

      glm::vec3 light = scene->Ia;

      for (int i = 0; i < scene->lightCount; ++i) {
        Light* l = &scene->lights[i];

        glm::vec3 Lm;

        if (l->type == l->Type_Point) {
          glm::vec3 lightDiff = hitPoint - (l->pos + glm::vec3(sin(time), cos(time), 0.0f) * 0.5f);
          float lightDist = glm::length(lightDiff);
          Lm = lightDiff / lightDist;
        }
        else if (l->type == l->Type_Directional) {
          Lm = l->dir;
        }

        // Shadow ray
        float hit = INFINITY;
        glm::vec2 tmp0; int tmp1; Mesh* tmp2;
        //traceScene(hitPoint + normal * 0.1f, Lm, scene, hit, tmp0, tmp1, tmp2);

        if (hit == INFINITY) {
          // lambert
          light += glm::dot(normal, Lm);

          // blinn-phong
        }
      }
      
      outColour = glm::clamp(glm::vec4(light, 1.0f), 0.0f, 1.0f);
    }
    else {
      outColour = glm::vec4(normalisedCoord, 0.0f, 1.0f);
      //outColour = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    if (x < screenRes.x && y < screenRes.y)
      surf2Dwrite(float4{outColour.x, outColour.y, outColour.z, outColour.w}, surface, x * 4 * sizeof(float), y);
}

cudaError_t renderScreen(cudaSurfaceObject_t surface, dim3 screenRes, float time, Scene* scene, glm::vec3 camPos, glm::mat4& viewMat)
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

    renderPixel<<<gridSize, blockSize>>>(surface, screenRes, blockSize, time, scene, camPos, viewMat);

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
