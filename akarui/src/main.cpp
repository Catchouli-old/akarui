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

  // render function
  char buf[255] = {};
  auto render = [&](const glm::vec3& camPos, const glm::mat3 camRot) {
    auto time = SDL_GetTicks() / 1000.0f;

    std::stringstream ss;

    ImGui::Begin("Render stats");
    ss.clear(); ss << "Camera pos: " << camPos.x << " " << camPos.y << " " << camPos.z;
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
          return 1;
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
    auto view = glm::translate(glm::mat4(1.0f), -camPos) * glm::inverse(glm::mat4(camRot));
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
    for (int i = 0; i < scene.meshCount; ++i) {
      auto* mesh = scene.meshes[i];
      auto* kdtree = mesh->kdtree;
      recurse(kdtree->m_root, 0, maxDepth);
    }

    glPopAttrib();
#endif

    cudaStreamSynchronize(0);
  };

  // Run window
  return runInWindow(argc, argv, screen_res, camPos, init, render);
}