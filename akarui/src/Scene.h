#pragma once

#include <glm/glm.hpp>
#include "cuda_runtime.h"
#include "tiny_obj_loader.h"

template <typename T>
T* cudaCopyArray(T* arr, int count)
{
  T* devPtr = nullptr;
  cudaMalloc(&devPtr, sizeof(T) * count);
  cudaMemcpy(devPtr, arr, sizeof(T) * count, cudaMemcpyHostToDevice);
  return devPtr;
}

// http://paulbourke.net/dataformats/mtl/
// loosely based around obj material support
struct Material
{
  enum Type
    // color = Kd
    { Type_Constant = 0
    // color = KaIa + Kd { SUM j=1..ls, (N * Lj)Ij }
    , Type_Lambert = 1
    // color = KaIa
    //         + Kd{ SUM j = 1..ls, (N*Lj)Ij }
    //         + Ks{ SUM j = 1..ls, ((H*Hj) ^ Ns)Ij }
    , Type_BlinnPhong = 2
    };

  Material() {}

  __device__ Material(Type t, glm::vec3 Ka, glm::vec3 Kd, glm::vec3 Ks, float Ns)
    : type(t), Ka(Ka), Kd(Kd), Ks(Ks), Ns(Ns) {}

  Type type = Type_Constant;

  glm::vec3 Ka = glm::vec3(0.0f); //^ ambient reflectance
  glm::vec3 Kd = glm::vec3(0.0f); //^ diffuse reflectance
  glm::vec3 Ks = glm::vec3(0.0f); //^ specular reflectance
  float Ns = 1.0f; //^ specular exponent
};

struct Mesh
{
  Mesh()
    : pos(new glm::vec3[0])
    , nrm(new glm::vec3[0])
    , uv(new glm::vec3[0])
    , mat(new Material[0])
    , idx(new int[0])
  {
  }

  ~Mesh()
  {
    delete[] pos;
    delete[] nrm;
    delete[] uv;
    delete[] mat;
    delete[] idx;
  }

  Mesh* cudaCopy()
  {
    Mesh* mesh = (Mesh*)malloc(sizeof(Mesh));
    memcpy(mesh, this, sizeof(Mesh));

    mesh->pos = cudaCopyArray(pos, posCount);
    mesh->nrm = cudaCopyArray(nrm, nrmCount);
    mesh->uv = cudaCopyArray(uv, uvCount);
    mesh->mat = cudaCopyArray(mat, materialCount);
    mesh->idx = cudaCopyArray(idx, idxCount);

    Mesh* devPtr = nullptr;
    cudaMalloc(&devPtr, sizeof(Mesh));
    cudaMemcpy(devPtr, mesh, sizeof(Mesh), cudaMemcpyHostToDevice);

    free(mesh);

    return devPtr;
  }

  static void cudaFree(Mesh* devPtr)
  {
    Mesh* mesh = (Mesh*)malloc(sizeof(Mesh));

    cudaMemcpy(mesh, devPtr, sizeof(Mesh), cudaMemcpyDeviceToHost);

    ::cudaFree(mesh->pos);
    ::cudaFree(mesh->nrm);
    ::cudaFree(mesh->uv);
    ::cudaFree(mesh->mat);
    ::cudaFree(mesh->idx);
    ::cudaFree(devPtr);

    free(mesh);
  }

  void set(std::vector<glm::vec3> pos, std::vector<glm::vec3> normals,
    std::vector<glm::vec3> uvs, std::vector<int> indices, std::vector<Material> materials)
  {
    delete[] this->pos;
    delete[] this->nrm;
    delete[] this->uv;
    delete[] this->mat;
    delete[] this->idx;

    posCount = (int)pos.size();
    this->pos = new glm::vec3[pos.size()];
    memcpy(this->pos, pos.data(), sizeof(glm::vec3) * pos.size());

    nrmCount = (int)normals.size();
    this->nrm = new glm::vec3[normals.size()];
    memcpy(this->nrm, normals.data(), sizeof(glm::vec3) * normals.size());

    uvCount = (int)uvs.size();
    this->uv = new glm::vec3[uvs.size()];
    memcpy(this->uv, uvs.data(), sizeof(glm::vec3) * uvs.size());

    idxCount = (int)indices.size();
    this->idx = new int[indices.size()];
    memcpy(this->idx, indices.data(), sizeof(int) * indices.size());

    materialCount = (int)materials.size();
    this->mat = new Material[materials.size()];
    memcpy(this->mat, materials.data(), sizeof(Material) * materials.size());
  }

  int posCount = 0;
  glm::vec3* pos = nullptr;

  int nrmCount = 0;
  glm::vec3* nrm = nullptr;

  int uvCount = 0;
  glm::vec3* uv = nullptr;

  int idxCount = 0;
  int* idx = nullptr;

  int materialCount = 0;
  Material* mat = nullptr;
};

struct Light
{
  enum Type { Type_Point };

  glm::vec3 pos = glm::vec3(0.0f);

  glm::vec3 Id = glm::vec3(0.0f); //^ diffuse intensity * colour
  glm::vec3 Is = glm::vec3(0.0f); //^ specular intensity * colour
};

struct Scene
{
  Scene()
    : meshes(new Mesh*[0])
    , lights(new Light[0])
  {
  }

  ~Scene()
  {
    for (int i = 0; i < meshCount; ++i)
      delete meshes[i];
    delete[] meshes;
    delete[] lights;
  }

  void load(const char* filename, glm::mat3 orientation)
  {
    // laod from obj file
    tinyobj::attrib_t vertexAttribs;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string err;
    if (!tinyobj::LoadObj(&vertexAttribs, &shapes, &materials, &err, filename) || !err.empty()) {
      fprintf(stderr, "Error loading %s: %s\n", filename, err.c_str());
    }

    // Load indices
    std::vector<int> idx;
    for (auto shape = shapes.begin(); shape != shapes.end(); ++shape) {
      for (auto it = shape->mesh.indices.begin(); it != shape->mesh.indices.end(); ++it) {
        idx.push_back(it->vertex_index);
      }
    }

    // Load vertex attribs
    std::vector<glm::vec3> pos;
    std::vector<glm::vec3> nrm;
    std::vector<glm::vec3> uvs;
    for (int i = 0; i + 2 < vertexAttribs.vertices.size(); i += 3)
      pos.push_back(orientation * glm::vec3(vertexAttribs.vertices[i], vertexAttribs.vertices[i + 1], vertexAttribs.vertices[i + 2]));
    for (int i = 0; i + 2 < vertexAttribs.texcoords.size(); i += 3)
      uvs.push_back(glm::vec3(vertexAttribs.texcoords[i], vertexAttribs.texcoords[i + 1], vertexAttribs.texcoords[i + 2]));
    for (int i = 0; i + 2 < vertexAttribs.normals.size(); i += 3)
      nrm.push_back(glm::vec3(vertexAttribs.normals[i], vertexAttribs.normals[i + 1], vertexAttribs.normals[i + 2]));

    // Make sure there's the same amount of uvs and normals as positions
    if (nrm.size() != pos.size())
      nrm.resize(pos.size());
    if (uvs.size() != pos.size())
      uvs.resize(pos.size());

    // Load materials
    std::vector<Material> mat;

    // Make mesh
    delete[] meshes;
    meshes = new Mesh*[1];
    meshCount = 1;
    meshes[0] = new Mesh;
    meshes[0]->set(pos, nrm, uvs, idx, mat);
  }

  void addLight(const Light& light)
  {
    Light* newLights = new Light[lightCount + 1];
    memcpy(newLights, lights, sizeof(Light) * lightCount);
    delete[] lights;
    lights = newLights;
    memcpy(lights + lightCount, &light, sizeof(Light));
    lightCount++;
  }

  Scene* cudaCopy()
  {
    Scene* scene = (Scene*)malloc(sizeof(Scene));
    memcpy(scene, this, sizeof(Scene));

    Mesh** cudaMeshes = (Mesh**)malloc(sizeof(Mesh*) * meshCount);
    for (int i = 0; i < meshCount; ++i) {
      cudaMeshes[i] = meshes[i]->cudaCopy();
    }

    scene->lights = cudaCopyArray(lights, lightCount);

    Mesh** devMeshes = nullptr;
    cudaMalloc(&devMeshes, sizeof(Mesh*) * meshCount);
    cudaMemcpy(devMeshes, cudaMeshes, sizeof(Mesh*) * meshCount, cudaMemcpyHostToDevice);
    scene->meshes = devMeshes;

    Scene* devPtr = nullptr;
    cudaMalloc(&devPtr, sizeof(Scene));
    cudaMemcpy(devPtr, scene, sizeof(Scene), cudaMemcpyHostToDevice);

    free(scene);

    return devPtr;
  }

  static void cudaFree(Scene* devPtr)
  {
    Scene* scene = (Scene*)malloc(sizeof(Scene));
    cudaMemcpy(scene, devPtr, sizeof(Scene), cudaMemcpyDeviceToHost);

    Mesh** meshes = (Mesh**)malloc(sizeof(Mesh*) * scene->meshCount);
    cudaMemcpy(meshes, scene->meshes, sizeof(Mesh*) * scene->meshCount, cudaMemcpyDeviceToHost);

    for (int i = 0; i < scene->meshCount; ++i) {
      Mesh::cudaFree(scene->meshes[i]);
    }
    ::cudaFree(scene->lights);
    ::cudaFree(devPtr);

    free(scene);
    free(meshes);
  }

  glm::vec3 Ia = glm::vec3(0.0f); //^ ambient light

  int meshCount = 0;
  Mesh** meshes = nullptr;

  int lightCount = 0;
  Light* lights = nullptr;
};