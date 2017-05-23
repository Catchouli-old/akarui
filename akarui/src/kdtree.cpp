#include "kdtree.h"

#include "cuda_runtime.h"
#include <functional>
#include <vector>
#include <map>

void Kdtree::buildTree(const glm::vec3* pos, const int* indices, int indexCount)
{
  delete m_root;

  m_root = new KdtreeNode;
  m_aabb = AABB();

  std::vector<prim> idx;
  for (int i = 0; i < indexCount; i += 3) {
    auto tri = prim{ indices[i], indices[i + 1], indices[i + 2] };
    m_aabb.extend(pos[tri.a]);
    m_aabb.extend(pos[tri.b]);
    m_aabb.extend(pos[tri.c]);
    idx.push_back(tri);
  }

  m_root->buildNode(pos, idx, m_aabb, 0);
}

void KdtreeNode::buildNode(const glm::vec3* pos, const std::vector<prim>& idx, const AABB& aabb, int depth)
{
  enum PlaneAxis {
    PlaneAxis_X,
    PlaneAxis_Y,
    PlaneAxis_Z
  };

  const int triTarget = 10;
  const int maxDepth = 10;

  auto terminate = [&, this]() {
    return idx.size() <= triTarget || depth >= maxDepth;
  };

  auto findPlane = [&, this](PlaneAxis& axis, float& pos) {
    axis = PlaneAxis(depth % 3);
    pos = 0.5f;
  };

  // Check if we want to terminate and make this a leaf
  if (terminate()) {
    m_prims = new prim[idx.size()];
    memcpy(m_prims, idx.data(), idx.size() * sizeof(prim));
    m_primCount = (int)idx.size();
    return;
  }

  // Find split axis
  PlaneAxis splitAxis;
  float splitPos;
  findPlane(splitAxis, splitPos);

  // convert the split pos from node aaab space to object space (the aabb is in object space)
  const float min = aabb.getMin()[int(splitAxis)];
  const float max = aabb.getMax()[int(splitAxis)];
  const float objectSpaceSplitPos = min + splitPos * (max - min);

  // the left and right cell's primitives
  std::vector<prim> split[2];

  for (auto p = idx.begin(); p != idx.end(); ++p) {
    const bool v0 = pos[p->a][int(splitAxis)] <= objectSpaceSplitPos;
    const bool v1 = pos[p->b][int(splitAxis)] <= objectSpaceSplitPos;
    const bool v2 = pos[p->c][int(splitAxis)] <= objectSpaceSplitPos;

    // get the side of the first element - left or right
    const int side = v0 ? 0 : 1;

    // if all vertices are on the same side, add it to that node's primitives, otherwise add it to both
    if (v0 == v1 && v1 == v2) {
      split[side].push_back(*p);
    }
    else {
      split[0].push_back(*p);
      split[1].push_back(*p);
    }
  }

  // calculate new nodes aabbs
  AABB aabbs[2];
  aabb.split(int(splitAxis), splitPos, aabbs[0], aabbs[1]);

  // create children
  m_left = new KdtreeNode;
  m_left->buildNode(pos, split[0], aabbs[0], depth + 1);
  m_right = new KdtreeNode;
  m_right->buildNode(pos, split[0], aabbs[0], depth + 1);
}

Kdtree* Kdtree::cudaCopy()
{
  Kdtree* tree = (Kdtree*)malloc(sizeof(Kdtree));
  memcpy(tree, this, sizeof(Kdtree));
  std::vector<std::pair<KdtreeNode*, KdtreeNode*>> nodes;
  std::map<KdtreeNode*, size_t> nodePos;
  std::map<KdtreeNode*, size_t> nodePrimPos;
  
  int prims = 0;
  std::function<void(KdtreeNode*)> walk = [&, this] (KdtreeNode* node) {
    nodePos[node] = nodes.size();
    nodePrimPos[node] = prims;
    KdtreeNode* clone = (KdtreeNode*)malloc(sizeof(KdtreeNode));
    memcpy(clone, node, sizeof(KdtreeNode));
    nodes.push_back(std::pair<KdtreeNode*, KdtreeNode*>(clone, node));
    prims += node->m_primCount;
    if (node->m_left)
      walk(node->m_left);
    if (node->m_right)
      walk(node->m_right);
  };

  // walk tree to find out how many nodes we need to allocate
  walk(m_root);

  // sum up how much space we need for the main tree class + the nodes + the index data
  size_t spaceReq = sizeof(Kdtree) + nodes.size() * sizeof(KdtreeNode);
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    spaceReq += sizeof(prim) * it->first->m_primCount;
  }

  // allocate memory and copy data
  Kdtree* ptr;
  cudaMalloc(&ptr, spaceReq);

  KdtreeNode* nodeStart = reinterpret_cast<KdtreeNode*>(ptr + 1);
  prim* primStart = reinterpret_cast<prim*>(nodeStart + nodes.size());

  // remap pointers
  tree->m_root = nodeStart;
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    if (it->first->m_left)
      it->first->m_left = nodeStart + nodePos.at(it->first->m_left);
    if (it->first->m_right)
      it->first->m_right = nodeStart + nodePos.at(it->first->m_right);
    if (it->first->m_prims)
      it->first->m_prims = primStart + nodePrimPos.at(it->second);
  }

  // copy data
  cudaMemcpy(ptr, &tree, sizeof(Kdtree), cudaMemcpyHostToDevice);

  int i = 0;
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    cudaMemcpy(nodeStart + i, &*it, sizeof(KdtreeNode), cudaMemcpyHostToDevice);
    ++i;
  }
  
  prim* curPrim = primStart;
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    cudaMemcpy(curPrim, it->second->m_prims, it->second->m_primCount * sizeof(prim), cudaMemcpyHostToDevice);
  }

  // free temporary memory
  free(tree);
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    free(it->first);
  }
  nodes.clear();

  return nullptr;
}

void Kdtree::cudaFree(Kdtree* ptr)
{
  // it should all be stored in one big memory block
  ::cudaFree(ptr);
}
