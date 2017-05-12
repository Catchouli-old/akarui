#pragma once

#include <glm/glm.hpp>

struct mesh_t {
  int vertexCount;
  glm::vec3 *vertices, *texCoords;
};

inline mesh_t* uploadMesh(mesh_t* mesh)
{
  mesh_t newMeshValue;
  memcpy(&newMeshValue, mesh, sizeof(mesh_t));

  cudaMalloc(&newMeshValue.vertices, mesh->vertexCount * sizeof(glm::vec3));
  cudaMemcpy(newMeshValue.vertices, mesh->vertices, mesh->vertexCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
  cudaMalloc(&newMeshValue.texCoords, mesh->vertexCount);
  cudaMemcpy(newMeshValue.texCoords, mesh->texCoords, mesh->vertexCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);

  mesh_t* devMesh;
  cudaMalloc(&devMesh, sizeof(mesh_t));
  cudaMemcpy(devMesh, &newMeshValue, sizeof(mesh_t), cudaMemcpyHostToDevice);

  return devMesh;
}


cudaError_t renderScreen(cudaSurfaceObject_t surface, dim3 screenRes, float time, mesh_t*, glm::vec3 camPos, glm::mat4& viewMat);