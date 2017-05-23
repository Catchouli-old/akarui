#pragma once

#include <vector>
#include "AABB.h"

struct prim {
  int a, b, c;
};

class KdtreeNode
{
public:
  KdtreeNode() {}

  ~KdtreeNode()
  {
    delete m_left;
    delete m_right;
    delete[] m_prims;
  }

  KdtreeNode(const KdtreeNode& other)
  {
    if (other.m_left)
      m_left = new KdtreeNode(*other.m_left);
    if (other.m_right)
      m_right = new KdtreeNode(*other.m_right);
    m_primCount = other.m_primCount;
    if (other.m_prims) {
      m_prims = new prim[m_primCount];
      memcpy(m_prims, other.m_prims, m_primCount * sizeof(prim));
    }
  }
  
  void buildNode(const glm::vec3* pos, const std::vector<prim>& idx, const AABB& aabb, int depth);

  KdtreeNode* m_left = nullptr;
  KdtreeNode* m_right = nullptr;
  prim* m_prims = nullptr;
  int m_primCount = 0;
};

class Kdtree
{
public:
  Kdtree() {}

  ~Kdtree()
  {
    delete m_root;
  }

  Kdtree(const Kdtree& kdtree)
  {
    m_aabb = kdtree.m_aabb;
    if (kdtree.m_root)
      m_root = new KdtreeNode(*kdtree.m_root);
  }

  void buildTree(const glm::vec3* pos, const int* indices, int indexCount);

  Kdtree* cudaCopy();
  static void cudaFree(Kdtree* ptr);

  KdtreeNode* m_root = nullptr;
  AABB m_aabb;
};