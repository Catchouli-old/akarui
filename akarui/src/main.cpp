#include <stdio.h>
#include <GL/glew.h>
#include <SDL2/SDL.h>
#include "window.h"
#include "imgui.h"
#include "kdtree.h"

int SDL_main(int argc, char** argv)
{
  // screen res
  dim3 screen_res(800, 600);

  // resources
  GLuint tex;
  cudaGraphicsResource_t cudaResource;
  Scene scene;
  Scene* sceneDevPtr;

  // init function
  auto init = [&]() {
    // create render target
    tex = createTexture(screen_res.x, screen_res.y);
    glBindTexture(GL_TEXTURE_2D, tex);
    cudaGraphicsGLRegisterImage(&cudaResource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    // load scene
    scene.load("resources/cornell-box", "CornellBox-Original.obj");
    //scene.load("meshes/head", "head.obj");
    //scene.load("resources/testObj", "testObj.obj");
    //scene.load("resources/teapot", "teapot.obj");

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
  auto render = [&](const glm::vec3& camPos, const glm::mat3 camRot) -> GLuint {
    auto time = SDL_GetTicks() / 1000.0f;

    ImGui::Begin("a");
    ImGui::LabelText("Test", "eue");
    ImGui::InputText("test", buf, 254);
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

    cudaStreamSynchronize(0);

    return tex;
  };

  // Run window
  return runInWindow(argc, argv, screen_res, init, render);
}