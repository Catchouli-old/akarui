
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"

#include <stdio.h>
#include <math.h>

#include <glm/glm.hpp>

#include "kernel.h"

// hit result
struct hit_t {
  float minT = INFINITY;
  glm::vec2 hitUV = glm::vec2(0.0f);
  int hitFace = -1;
  const Mesh* hitMesh = nullptr;
  const Material* hitMat = nullptr;
  // front (1) or back (-1) face
  float face = 1.0f;
};


// intersect a ray with a triangle
__device__ bool intersectRayTriangle(glm::vec3 o, glm::vec3 d,
  glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
  glm::vec2& uv, float& t, float& face)
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
  face = signbit(det) ? -1.0f : 1.0f;

  return true;
}

// intersect a ray with a sphere
__device__ bool rayIntersectSphere(glm::vec4 sphere, glm::vec3 origin, glm::vec3 direction, hit_t& out) {
  float c = glm::length(origin - glm::vec3(sphere)) - sphere.w*sphere.w;
  float dotVal = glm::dot(direction, (origin - glm::vec3(sphere)));
  float sqrtVal = dotVal*dotVal - glm::dot(origin - glm::vec3(sphere), origin - glm::vec3(sphere)) + sphere.w*sphere.w;

  bool hit;
  glm::vec2 intersections;

  if (sqrtVal <= 0.0) {
    hit = false;
  }
  if (sqrtVal == 0.0) {
    intersections.x = -dotVal;
    intersections.y = -dotVal;
    hit = true;
  }
  else {
    float d1 = -(dotVal)+sqrt(sqrtVal);
    float d2 = -(dotVal)-sqrt(sqrtVal);
    if (d1 < 0.0 && d2 < 0.0) {
      hit = false;
    }
    else if (d1 < 0.0 || d2 < 0.0) {
      intersections.x = glm::min(d1, d2);
      intersections.y = glm::max(d1, d2);
      hit = true;
    }
    else {
      intersections.x = glm::min(d1, d2);
      intersections.y = glm::max(d1, d2);
      hit = true;
    }
  }

  if (hit && intersections.x < out.minT) {
    out.hitFace = -1;
    out.hitMat = nullptr;
    out.hitMesh = nullptr;
    out.hitUV = glm::vec2(0.0f, 0.0f);
    out.minT = intersections.x;
  }

  return hit;
}

// raytrace a scene
__device__ void traceScene(glm::vec3 origin, glm::vec3 dir, const Scene* scene, hit_t& out)
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
      float face;
      intersectRayTriangle(origin, dir, v0, v1, v2, uv, hitT, face);
      if (hitT != INFINITY && hitT < out.minT && hitT >= 0.0f) {
        out.minT = hitT;
        out.hitFace = tri;
        out.hitUV = uv;
        out.hitMesh = mesh;
        int mat = mesh->matIdx[tri];
        out.hitMat = mat == -1 ? &scene->defaultMat : &mesh->mat[mat];
        out.face = face;
      }
    }
  }
}

// trace a ray recursively
//__device__ glm::vec3 traceRay(Scene* scene, glm::vec3 origin, glm::vec3 direction, int depth)
//{
//
//}

__global__ void renderPixel(cudaSurfaceObject_t surface, dim3 screenRes, dim3 blockSize, float time, Scene* scene, glm::vec3 camPos, glm::mat3 viewRot)
{
  // pixel coordinate
  int x = blockIdx.x * blockSize.x + threadIdx.x;
  int y = blockIdx.y * blockSize.y + threadIdx.y;

  // aspect ratio
  float aspect = (float)screenRes.x / screenRes.y;

  // normalised screen coordinates (-1..1) in the x axis and (-aspect..aspect) in the y axis
  glm::vec2 normalisedCoord = 2.0f * glm::vec2(x, screenRes.y - y) / glm::vec2(screenRes.x, screenRes.y) - glm::vec2(1.0f);
  normalisedCoord.y /= aspect;

  // construct ray
  glm::vec3 origin = camPos;
  glm::vec3 direction = viewRot * glm::normalize(glm::vec3(normalisedCoord.x, normalisedCoord.y, -1.0f));

  // ray hit
  hit_t hit;
  hit.hitMat = &scene->defaultMat;

  // raytrace the lights for debug drawing
  for (int i = 0; i < scene->lightCount; ++i) {
    Light* light = &scene->lights[i];
    if (light->type == Light::Type_Point) {
      glm::vec4 sphere = glm::vec4(light->pos, 0.1f);
      //rayIntersectSphere(sphere, origin, direction, hit);
    }
  }

  // raytrace the scene
  traceScene(origin, direction, scene, hit);

  glm::vec3 outColour(0.0f);

  //outColour = hit.minT != INFINITY ? glm::vec3(1.0f) : glm::vec3(0.0f);
  //if (false)
  if (hit.minT != INFINITY) {
    glm::vec3 normal = glm::vec3(1.0f);

    if (hit.hitMesh != nullptr && hit.hitFace >= 0) {
      // Barycentric coordinates
      float v = hit.hitUV.x;
      float t = hit.hitUV.y;
      float u = 1.0f - hit.hitUV.x - hit.hitUV.y;

      // Get face indices
      int a = hit.hitMesh->idx[hit.hitFace * 3 + 0];
      int b = hit.hitMesh->idx[hit.hitFace * 3 + 1];
      int c = hit.hitMesh->idx[hit.hitFace * 3 + 2];

      // interpolate normal
      glm::vec3 nA = hit.hitMesh->nrm[a];
      glm::vec3 nB = hit.hitMesh->nrm[b];
      glm::vec3 nC = hit.hitMesh->nrm[c];
      normal = u * nA + v * nB + t * nC;
      normal *= hit.face;
    }

    glm::vec3 hitPoint = origin + hit.minT * direction;
    glm::vec3 V = origin - hitPoint;

    glm::vec3 light = scene->Ia;

    for (int i = 0; i < scene->lightCount; ++i) {
      Light* l = &scene->lights[i];
      glm::vec3 Lm;

      if (l->type == l->Type_Point) {
        glm::vec3 lightDiff = l->pos - hitPoint;
        float lightDist = glm::length(lightDiff);
        Lm = lightDiff / lightDist;
      }
      else if (l->type == l->Type_Directional) {
        Lm = l->dir;
      }

      // Shadow ray
      hit_t shadowHit;
      shadowHit.hitMat = &scene->defaultMat;
      //traceScene(hitPoint + normal * 0.01f, Lm, scene, shadowHit);

      light = glm::vec3(0.0f);

      if (shadowHit.minT == INFINITY) {
        // lambert
        light += glm::dot(normal, Lm) * l->Id * hit.hitMat->Kd;

        // blinn-phong
        glm::vec3 R = glm::reflect(-Lm, normal);
        glm::vec3 halfVector = (Lm + V) / glm::length(Lm + V);
        light += glm::pow(glm::dot(normal, halfVector), hit.hitMat->Ns) * l->Is * hit.hitMat->Ks;
      }
    }
    
    outColour = glm::clamp(light, 0.0f, 1.0f);
  }

  if (x < screenRes.x && y < screenRes.y)
    surf2Dwrite(float4{outColour.x, outColour.y, outColour.z, 0.0f}, surface, x * 4 * sizeof(float), y);
}

cudaError_t renderScreen(cudaSurfaceObject_t surface, dim3 screenRes, float time, Scene* scene, glm::vec3 camPos, const glm::mat3& viewMat)
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

  // calculate grid size to cover the whole screen
  dim3 gridSize(int(ceil(screenRes.x/float(blockSize.x))), int(ceil(screenRes.y/float(blockSize.y))));

  // invoke kernel
  renderPixel<<<gridSize, blockSize>>>(surface, screenRes, blockSize, time, scene, camPos, viewMat);

  // check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
  }
  
  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
  }
  
  return cudaStatus;
}
