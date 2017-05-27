#include <stdio.h>
#include <GL/glew.h>
#include <SDL2/SDL.h>
#include "window.h"
#include "imgui.h"
#include "kdtree.h"

void cube() {
  glVertex3f(1.0f, 1.0f, 0.0f);
  glVertex3f(0.0f, 1.0f, 0.0f);
  glVertex3f(0.0f, 1.0f, 1.0f);
  glVertex3f(1.0f, 1.0f, 1.0f);
  glVertex3f(1.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 0.0f, 1.0f);
  glVertex3f(1.0f, 0.0f, 1.0f);
  glVertex3f(1.0f, 1.0f, 1.0f);
  glVertex3f(0.0f, 1.0f, 1.0f);
  glVertex3f(0.0f, 0.0f, 1.0f);
  glVertex3f(1.0f, 0.0f, 1.0f);
  glVertex3f(1.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 1.0f, 0.0f);
  glVertex3f(1.0f, 1.0f, 0.0f);
  glVertex3f(0.0f, 1.0f, 1.0f);
  glVertex3f(0.0f, 1.0f, 0.0f);
  glVertex3f(0.0f, 0.0f, 0.0f);
  glVertex3f(0.0f, 0.0f, 1.0f);
  glVertex3f(1.0f, 1.0f, 1.0f);
  glVertex3f(1.0f, 1.0f, 0.0f);
  glVertex3f(1.0f, 0.0f, 0.0f);
  glVertex3f(1.0f, 0.0f, 1.0f);
}

int SDL_main(int argc, char** argv)
{
  // screen res
  dim3 screen_res(800, 600);
  glm::vec3 camPos(-0.5f, 1.0f, 10.5f);// 3.5f);

  // resources
  GLuint tex;
  cudaGraphicsResource_t cudaResource;
  Scene scene;
  Scene* sceneDevPtr;
  GLuint program;

  // ray
  glm::vec3 origin(-2.51f, 3.16f, 7.5f), direction(0.44f, -0.21f, -0.88f);
  //glm::vec3 origin(0.0f, 1.0f, 3.0f), direction(0.0f, 0.0f, -1.0f);
  direction = glm::normalize(direction);

  int maxDepth = 1;

  // init function
  auto init = [&]() {
    // create render target
    tex = createTexture(screen_res.x, screen_res.y);
    glBindTexture(GL_TEXTURE_2D, tex);
    cudaGraphicsGLRegisterImage(&cudaResource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // load scene
    //scene.load("resources/cornell-box", "CornellBox-Original.obj");
    //scene.load("meshes/head", "head.obj");
    //scene.load("resources/testObj", "testObj.obj");
    //scene.load("resources/teapot", "teapot.obj");
    scene.load("stanford", "bunny.obj");
    //scene.load(".", "teapot.obj");

    // load shader
    program = compileShaderProgram("resources/shaders/textured.glsl");

    // add a light to the scene
    scene.Ia = glm::vec3(0.2f);

    Light light;
    light.type = Light::Type_Point;
    light.Id = glm::vec3(0.7f);
    light.Is = glm::vec3(0.3f);
    scene.addLight(light);

    // upload scene to gpu
    sceneDevPtr = scene.cudaCopy();
  };

  std::stringstream ss;

  // render function
  char buf[255] = {};
  auto render = [&](const glm::vec3& camPos, const glm::mat3 camRot) {
    auto time = SDL_GetTicks() / 1000.0f;

    glm::vec3 forward = camRot * glm::vec3(0.0f, 0.0f, -1.0f);

    if (SDL_GetKeyboardState(nullptr)[SDL_SCANCODE_F1]) {
      origin = camPos;
      direction = forward;
    }

    ImGui::Begin("Render stats");
    ss.str(""); ss << "Camera pos: " << camPos.x << " " << camPos.y << " " << camPos.z;
    ImGui::Text(ss.str().c_str());
    ss.str(""); ss << "Facing dir: " << forward.x << " " << forward.y << " " << forward.z;
    ImGui::Text(ss.str().c_str());
    ImGui::SliderInt("Max depth", &maxDepth, 0, 25);
    ImGui::End();

    // update scene
    glm::vec3 lightPos = glm::vec3(0.0f, 1.0f, 1.0f);
    scene.lights[0].pos = lightPos + glm::vec3(sin(time), cos(time), 0.0f) * 0.5f;
    scene.cudaUpdate(sceneDevPtr);

    // map texture into cuda and invoke render kernel
    cudaGraphicsMapResources(1, &cudaResource);
    {
      cudaArray_t cudaArray;
      cudaGraphicsSubResourceGetMappedArray(&cudaArray, cudaResource, 0, 0);
      cudaResourceDesc cudaArrayResourceDesc;
      {
        cudaArrayResourceDesc.resType = cudaResourceTypeArray;
        cudaArrayResourceDesc.res.array.array = cudaArray;
      }
      cudaSurfaceObject_t cudaSurfaceObject;
      cudaCreateSurfaceObject(&cudaSurfaceObject, &cudaArrayResourceDesc);
      if (false)
      {
        // invoke render kernel
        cudaError_t cudaStatus = renderScreen(cudaSurfaceObject, screen_res, time, sceneDevPtr, camPos, camRot);
        if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "renderScreen failed!\n");
        }
      }
      cudaDestroySurfaceObject(cudaSurfaceObject);
    }
    cudaGraphicsUnmapResources(1, &cudaResource);

    // render texture to screen
    glBindTexture(GL_TEXTURE_2D, tex);
    glUseProgram(program);
    drawFullscreenQuad();
    glUseProgram(0);

    // kd tree visualiser
#if 1
    // Opengl renderer
    auto proj = glm::perspective(1.0f, 800.0f / 600.0f, 0.1f, 100.0f);
    //auto view = glm::translate(glm::mat4(1.0f), -camPos) * glm::inverse(glm::mat4(camRot));
    auto view = glm::inverse(glm::mat4(camRot)) * glm::translate(glm::mat4(1.0f), -camPos);
    proj = proj * view;

    glPushAttrib(GL_ALL_ATTRIB_BITS);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf((float*)&proj);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);

    glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glColor3f(1.0f, 1.0f, 1.0f);
    if (false)
    for (int i = 0; i < scene.meshCount; ++i) {
      auto* mesh = scene.meshes[i];
      glBegin(GL_TRIANGLES);
      for (int j = 0; j < mesh->idxCount; ++j) {
        const auto& v = mesh->pos[mesh->idx[j]];
        const auto& n = mesh->nrm[mesh->idx[j]];
        glNormal3f(n.x, n.y, n.z);
        glVertex3f(v.x, v.y, v.z);
      }
      glEnd();
    }

    glBegin(GL_LINES);
    glColor3f(1.0f, 1.0f, 1.0f);
    glVertex3f(origin.x, origin.y, origin.z);
    glm::vec3 end = origin + 100.0f * direction;
    glVertex3f(end.x, end.y, end.z);
    glEnd();

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    std::function<void(KdtreeNode*,int,int)> recurse = [&recurse, &camPos](KdtreeNode* node, int depth, int maxDepth) {
      glm::vec3 min = node->m_aabb.getMin();
      glm::vec3 max = node->m_aabb.getMax();
      glm::vec3 scale = max - min;

      glColor4f(0.0f, 0.0f, 1.0f, 0.01f);
      glPushMatrix();
      glTranslatef(min.x, min.y, min.z);
      glScalef(scale.x, scale.y, scale.z);
      glBegin(GL_QUADS);
      cube();
      glEnd();
      glPopMatrix();

      if (depth >= maxDepth)
        return;

      if (node->m_left)
        recurse(node->m_left, depth + 1, maxDepth);
      if (node->m_right)
        recurse(node->m_right, depth + 1, maxDepth);
    };

    glTranslatef(0.1f, 0.0f, 0.0f);
    Kdtree* kdtree;
    for (int i = 0; i < scene.meshCount; ++i) {
      auto* mesh = scene.meshes[i];
      kdtree = mesh->kdtree;
      recurse(kdtree->m_root, 0, maxDepth);
    }

    // ray/aabb intersect
    auto intersectRayAABB = [](const AABB& b, const glm::vec3& origin, const glm::vec3& dir, glm::vec2& out_t) {
      float& tmin = out_t.x;
      float& tmax = out_t.y;
      out_t = glm::vec2(-INFINITY, INFINITY);

      if (dir.x != 0.0) {
        float tx1 = (b.getMin().x - origin.x) / dir.x;
        float tx2 = (b.getMax().x - origin.x) / dir.x;

        tmin = glm::max(tmin, glm::min(tx1, tx2));
        tmax = glm::min(tmax, glm::max(tx1, tx2));
      }

      if (dir.y != 0.0) {
        float ty1 = (b.getMin().y - origin.y) / dir.y;
        float ty2 = (b.getMax().y - origin.y) / dir.y;

        tmin = glm::max(tmin, glm::min(ty1, ty2));
        tmax = glm::min(tmax, glm::max(ty1, ty2));
      }

      if (dir.z != 0.0) {
        float tz1 = (b.getMin().z - origin.z) / dir.z;
        float tz2 = (b.getMax().z - origin.z) / dir.z;

        tmin = glm::max(tmin, glm::min(tz1, tz2));
        tmax = glm::min(tmax, glm::max(tz1, tz2));
      }

      return tmax >= tmin;
    };

    auto findExit = [](const AABB& b, const glm::vec3& origin, const glm::vec3& dir, float& out_t, int& out_face) {
      out_t = INFINITY;
      out_face = -1;

      if (dir.x != 0.0) {
        float tx1 = (b.getMin().x - origin.x) / dir.x;
        float tx2 = (b.getMax().x - origin.x) / dir.x;

        float t = glm::max(tx1, tx2);
        if (t < out_t) {
          out_t = t;
          out_face = 0 + (dir.x < 0 ? 0 : 1);
        }
      }

      if (dir.y != 0.0) {
        float ty1 = (b.getMin().y - origin.y) / dir.y;
        float ty2 = (b.getMax().y - origin.y) / dir.y;

        float t = glm::max(ty1, ty2);
        if (t < out_t) {
          out_t = t;
          out_face = 2 + (dir.y < 0 ? 0 : 1);
        }
      }

      if (dir.z != 0.0) {
        float tz1 = (b.getMin().z - origin.z) / dir.z;
        float tz2 = (b.getMax().z - origin.z) / dir.z;

        float t = glm::max(tz1, tz2);
        if (t < out_t) {
          out_t = t;
          out_face = 4 + (dir.z < 0 ? 0 : 1);
        }
      }
    };

    // intersect ray with kdtree
    glm::vec2 tmp;
    glm::vec2 entryExit;
    KdtreeNode* node = kdtree->m_root;
    int blarg = 0;
    int flarg = 0;
    if (intersectRayAABB(kdtree->m_aabb, origin, direction, entryExit)) {
      while (entryExit.x < entryExit.y) {
        // down traversal (until we get to a leaf)
        glm::vec3 pEntry = origin + entryExit.x * direction;
        while (node->m_left) {
          ++blarg;
          float pEntrySplitAxis = pEntry[node->m_splitAxis];
          // <=? does it matter?
          node = (pEntrySplitAxis < node->m_splitPosObj ? node->m_left : node->m_right);
        }

        // at a leaf, intersect with the leaf's triangles and exit the leaf
        // todo: intersect leaf triangles, for now just draw the leaf
        glColor4f(1.0f, 0.0f, 0.0f, 0.35f);
        glPushMatrix();
        glm::vec3 min = node->m_aabb.getMin();
        glm::vec3 max = node->m_aabb.getMax();
        glm::vec3 scale = max - min;
        glTranslatef(min.x, min.y, min.z);
        glScalef(scale.x, scale.y, scale.z);
        glBegin(GL_QUADS);
        cube();
        glEnd();
        glPopMatrix();
        flarg++;

        // exit the leaf using a rope
        int exitFace;
        findExit(node->m_aabb, origin, direction, entryExit.x, exitFace);
        node = node->m_ropes[exitFace];

        // no intersection if node is null
        if (node == nullptr)
          break;
      }
    }
    printf("downtraversals: %d, leaves visited: %d\n", blarg, flarg);

    glPopAttrib();
#endif

    cudaStreamSynchronize(0);
  };

  // Run window
  return runInWindow(argc, argv, screen_res, camPos, init, render);
}