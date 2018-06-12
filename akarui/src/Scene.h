#pragma once

#include <glm/glm.hpp>
//#include "cuda_runtime.h"
#include "tiny_obj_loader.h"
#include "kdtree.h"
#include <vector>
#include <map>

template <typename T>
T* cudaCopyArray(T* arr, int count)
{
  T* devPtr = nullptr;
  //cudaMalloc(&devPtr, sizeof(T) * count);
  //cudaMemcpy(devPtr, arr, sizeof(T) * count, cudaMemcpyHostToDevice);
  return devPtr;
}

// http://paulbourke.net/dataformats/mtl/
// loosely based around obj material support
struct Material
{
  Material() {}

  Material(glm::vec3 Ka, glm::vec3 Kd, glm::vec3 Ks, float Ns)
    : Ka(Ka), Kd(Kd), Ks(Ks), Ns(Ns) {}

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
    , matIdx(new int[0])
  {
  }

  ~Mesh()
  {
    delete[] pos;
    delete[] nrm;
    delete[] uv;
    delete[] mat;
    delete[] idx;
    delete[] matIdx;
    delete kdtree;
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
    mesh->matIdx = cudaCopyArray(matIdx, matIdxCount);
    mesh->kdtree = kdtree->cudaCopy();

    Mesh* devPtr = nullptr;
    //cudaMalloc(&devPtr, sizeof(Mesh));
    //cudaMemcpy(devPtr, mesh, sizeof(Mesh), cudaMemcpyHostToDevice);

    free(mesh);

    return devPtr;
  }

  static void cudaFree(Mesh* devPtr)
  {
    Mesh* mesh = (Mesh*)malloc(sizeof(Mesh));

    //cudaMemcpy(mesh, devPtr, sizeof(Mesh), cudaMemcpyDeviceToHost);

    //::cudaFree(mesh->pos);
    //::cudaFree(mesh->nrm);
    //::cudaFree(mesh->uv);
    //::cudaFree(mesh->mat);
    //::cudaFree(mesh->idx);
    //::cudaFree(mesh->matIdx);
    Kdtree::cudaFree(mesh->kdtree);
    //::cudaFree(devPtr);

    free(mesh);
  }

  void set(std::vector<glm::vec3> pos, std::vector<glm::vec3> normals,
    std::vector<glm::vec3> uvs, std::vector<int> indices, std::vector<int> materialIndices,
    std::vector<Material> materials)
  {
    delete[] this->pos;
    delete[] this->nrm;
    delete[] this->uv;
    delete[] this->mat;
    delete[] this->idx;
    delete[] this->matIdx;
    delete kdtree;
    kdtree = nullptr;

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

    matIdxCount = (int)materialIndices.size();
    this->matIdx = new int[materialIndices.size()];
    memcpy(this->matIdx, materialIndices.data(), sizeof(int) * materialIndices.size());

    materialCount = (int)materials.size();
    this->mat = new Material[materials.size()];
    memcpy(this->mat, materials.data(), sizeof(Material) * materials.size());
  }

  void buildKdtree()
  {
    delete kdtree;
    kdtree = new Kdtree;
    kdtree->buildTree(pos, idx, idxCount);
  }

  int posCount = 0;
  glm::vec3* pos = nullptr;

  int nrmCount = 0;
  glm::vec3* nrm = nullptr;

  int uvCount = 0;
  glm::vec3* uv = nullptr;

  int idxCount = 0;
  int* idx = nullptr;

  // material index is the per-triangle index into the materials
  // -1 = default material
  int matIdxCount = 0;
  int* matIdx = nullptr;

  int materialCount = 0;
  Material* mat = nullptr;

  Kdtree* kdtree = nullptr;
};

struct Light
{
  Light() {}

  enum Type { Type_Directional, Type_Point };

  Type type = Type_Directional;

  glm::vec3 Id = glm::vec3(0.0f); //^ diffuse intensity * colour
  glm::vec3 Is = glm::vec3(0.0f); //^ specular intensity * colour

  glm::vec3 pos = glm::vec3(0.0f);
  glm::vec3 dir = glm::normalize(glm::vec3(0.0f, -1.0f, -1.0f));
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

  // todo: this clears the meshes and adds the new one
  // might as well make it add the data instead
  void load(const char* dir, const char* filename)
  {
    // laod from obj file
    tinyobj::attrib_t vertexAttribs;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string err;
    std::string dirPath = std::string(dir) + "/";
    std::string fullPath = std::string(dir) + "/" + filename;
    if (!tinyobj::LoadObj(&vertexAttribs, &shapes, &materials, &err, fullPath.c_str(), dirPath.c_str()) || !err.empty()) {
      fprintf(stderr, "Error loading %s: %s\n", filename, err.c_str());
    }

    // Mesh data
    struct {
      std::vector<glm::vec3> pos;
      std::vector<glm::vec3> nrm;
      std::vector<glm::vec3> uvs;
      std::vector<int> idx;
      std::vector<int> matIdx;
    } data;

    // Load indices
    for (auto shape = shapes.begin(); shape != shapes.end(); ++shape) {
      for (auto it = shape->mesh.indices.begin(); it != shape->mesh.indices.end(); ++it) {
        data.idx.push_back(it->vertex_index);
      }
      for (auto it = shape->mesh.material_ids.begin(); it != shape->mesh.material_ids.end(); ++it) {
        data.matIdx.push_back(*it);
      }
    }

    // Load vertex attribs
    for (int i = 0; i + 2 < vertexAttribs.vertices.size(); i += 3)
      data.pos.push_back(glm::vec3(vertexAttribs.vertices[i], vertexAttribs.vertices[i + 1], vertexAttribs.vertices[i + 2]));
    for (int i = 0; i + 2 < vertexAttribs.texcoords.size(); i += 3)
      data.uvs.push_back(glm::vec3(vertexAttribs.texcoords[i], vertexAttribs.texcoords[i + 1], vertexAttribs.texcoords[i + 2]));
    for (int i = 0; i + 2 < vertexAttribs.normals.size(); i += 3)
      data.nrm.push_back(glm::vec3(vertexAttribs.normals[i], vertexAttribs.normals[i + 1], vertexAttribs.normals[i + 2]));

    // If there's a different number of normals to positions, calculate smooth normals
    if (true || data.nrm.size() != data.pos.size()) {
      data.nrm.resize(data.pos.size());

      // calculate normals for each face
      std::vector<glm::vec3> faceNormals;
      faceNormals.resize(data.idx.size() / 3);
      for (int i = 0; i < data.idx.size() / 3; ++i) {
        int a = data.idx[i * 3 + 0];
        int b = data.idx[i * 3 + 1];
        int c = data.idx[i * 3 + 2];
        glm::vec3 v0 = data.pos[a];
        glm::vec3 v1 = data.pos[b];
        glm::vec3 v2 = data.pos[c];
        faceNormals[i] = glm::normalize(glm::cross(v1 - v0, v2 - v1));
      }

      // for each vertex, average the normals of the connected faces
      if (false)
      for (int i = 0; i < data.nrm.size(); ++i) {
        std::vector<glm::vec3> normalsToAvg;
        for (int j = 0; j < data.idx.size()/3; ++j) {
          int a = data.idx[j * 3 + 0];
          int b = data.idx[j * 3 + 1];
          int c = data.idx[j * 3 + 2];
          if (a == i || b == i || c == i) {
            normalsToAvg.push_back(faceNormals[j]);
          }
        }
        data.nrm[i] = glm::vec3(0.0f);
        for (int j = 0; j < normalsToAvg.size(); ++j) {
          data.nrm[i] += normalsToAvg[j] * (1.0f / normalsToAvg.size());
        }
      }
    }

    if (data.uvs.size() != data.pos.size())
      data.uvs.resize(data.pos.size());

    // Load materials
    std::vector<Material> mat;
    for (auto it = materials.begin(); it != materials.end(); ++it) {
      Material m;
      m.Ka = glm::vec3(it->ambient[0], it->ambient[1], it->ambient[2]);
      m.Kd = glm::vec3(it->diffuse[0], it->diffuse[1], it->diffuse[2]);
      m.Ks = glm::vec3(it->specular[0], it->specular[1], it->specular[2]);
      m.Ns = it->shininess;
      mat.push_back(m);
    }

    // Make mesh
    delete[] meshes;
    meshes = new Mesh*[1];
    meshCount = 1;
    meshes[0] = new Mesh;
    meshes[0]->set(data.pos, data.nrm, data.uvs, data.idx, data.matIdx, mat);
    meshes[0]->buildKdtree();
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
    //cudaMalloc(&devMeshes, sizeof(Mesh*) * meshCount);
    //cudaMemcpy(devMeshes, cudaMeshes, sizeof(Mesh*) * meshCount, cudaMemcpyHostToDevice);
    scene->meshes = devMeshes;

    Scene* devPtr = nullptr;
    //cudaMalloc(&devPtr, sizeof(Scene));
    //cudaMemcpy(devPtr, scene, sizeof(Scene), cudaMemcpyHostToDevice);

    free(scene);

    return devPtr;
  }

  static void cudaFree(Scene* devPtr)
  {
    Scene* scene = (Scene*)malloc(sizeof(Scene));
    //cudaMemcpy(scene, devPtr, sizeof(Scene), cudaMemcpyDeviceToHost);

    Mesh** meshes = (Mesh**)malloc(sizeof(Mesh*) * scene->meshCount);
    //cudaMemcpy(meshes, scene->meshes, sizeof(Mesh*) * scene->meshCount, cudaMemcpyDeviceToHost);

    for (int i = 0; i < scene->meshCount; ++i) {
      Mesh::cudaFree(scene->meshes[i]);
    }
    //::cudaFree(scene->lights);
    //::cudaFree(devPtr);

    free(scene);
    free(meshes);
  }

  void cudaUpdate(Scene* devPtr)
  {
    // update just the parts that are likely to change
    Scene* copy = (Scene*)malloc(sizeof(Scene));
    //cudaMemcpy(copy, devPtr, sizeof(Scene), cudaMemcpyDeviceToHost);

    copy->Ia = Ia;
    copy->lightCount = lightCount;
    //::cudaFree(copy->lights);
    //copy->lights = cudaCopyArray(lights, lightCount);

    //cudaMemcpy(devPtr, copy, sizeof(Scene), cudaMemcpyHostToDevice);
    free(copy);
  }

  glm::vec3 Ia = glm::vec3(0.0f); //^ ambient light

  int meshCount = 0;
  Mesh** meshes = nullptr;

  int lightCount = 0;
  Light* lights = nullptr;

  Material defaultMat = Material(glm::vec3(1.0f), glm::vec3(0.0f), glm::vec3(0.0f), 1.0f);
};