#include <stdio.h>
#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tclap/CmdLine.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "kernel.h"
#include "Scene.h"

#include "tiny_obj_loader.h"

void drawFullscreenQuad();
GLuint createTexture(int width, int height);
GLuint compileShaderProgram(const char* filename);

int main(int argc, char** argv)
{
  TCLAP::CmdLine cmd("akarui", ' ', "0.0");
  TCLAP::ValueArg<bool> vsync("v", "vsync", "Whether to run with vsync on or off", false, true, "bool");
  cmd.add(vsync);
  cmd.parse(argc, argv);

  // build object transform
  glm::mat4 m(1.0f);
  m = glm::rotate(m, 3.14159f, glm::vec3(0.0f, 1.0f, 0.0f));

  // load scene
  Scene scene;
  scene.load("resources/cornell-box/CornellBox-Original.obj", glm::mat3(m));

  // Add a light to the scene
  scene.Ia = glm::vec3(0.1f);

  Light light;
  light.pos = glm::vec3(0.0f, -0.5f, 0.3f);
  light.Id = glm::vec3(0.6f);
  light.Is = glm::vec3(0.3f);
  scene.addLight(light);

  // Upload scene to gpu
  Scene* sceneDevPtr = scene.cudaCopy();

  // screen res
  dim3 screen_res(800, 600);

  // initialise SDL
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window* window;
  SDL_Renderer* renderer;
  if (SDL_CreateWindowAndRenderer(screen_res.x, screen_res.y, SDL_WINDOW_OPENGL, &window, &renderer) != 0) {
    fprintf(stderr, "Failed to create window");
    return 1;
  }
  SDL_SetRelativeMouseMode(SDL_TRUE);

  // initialise opengl
  SDL_GL_CreateContext(window);
  SDL_GL_SetSwapInterval(vsync.getValue() ? 1 : 0);
  glewInit();

  // create an opengl texture
  GLuint tex = createTexture(screen_res.x, screen_res.y);
  glBindTexture(GL_TEXTURE_2D, tex);
  std::vector<glm::vec4> data;
  data.resize(screen_res.x * screen_res.y);
  for (unsigned int i = 0; i < screen_res.x * screen_res.y; ++i) {
    data[i] = glm::vec4((i%10000)/10000.0f, 0.0f, 0.0f, 0.0f);
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, screen_res.x, screen_res.y, 0, GL_RGBA, GL_FLOAT, data.data());
  cudaGraphicsResource_t cudaResource;
  cudaGraphicsGLRegisterImage(&cudaResource, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

  // load shader
  GLuint program = compileShaderProgram("resources/shaders/textured.glsl");

  // fps counter
  int frames = 0;
  int fps = 0;
  uint32_t lastUpdate = SDL_GetTicks();

  // camera params
  glm::vec3 camPos(0.0f, 1.0f, -2.0f);
  glm::vec2 camRot(0.0f);

  bool running = true;
  while (running) {
    // handle events
    SDL_Event evt;
    while (SDL_PollEvent(&evt)) {
      if (evt.type == SDL_QUIT || (evt.type == SDL_KEYDOWN && evt.key.keysym.scancode == SDL_SCANCODE_ESCAPE)) {
        running = false;
      }
      else if (evt.type == SDL_MOUSEMOTION) {
        camRot.y += (float)evt.motion.xrel;
        camRot.x += (float)evt.motion.yrel;
        camRot.x = glm::clamp(camRot.x, -90.0f, 90.0f);
      }
    }

    // update camera
    const Uint8* keys = SDL_GetKeyboardState(nullptr);

    glm::mat4 m(1.0f);
    m = glm::rotate(m, 0.01f * camRot.y, glm::vec3(0.0f, 1.0f, 0.0f));
    m = glm::rotate(m, 0.01f * camRot.x, glm::vec3(1.0f, 0.0f, 0.0f));

    if (keys[SDL_SCANCODE_W])
      camPos += 0.01f * glm::mat3(m) * glm::vec3(0.0f, 0.0f, 1.0f);
    if (keys[SDL_SCANCODE_S])
      camPos -= 0.01f * glm::mat3(m) * glm::vec3(0.0f, 0.0f, 1.0f);
    if (keys[SDL_SCANCODE_A])
      camPos -= 0.01f * glm::mat3(m) * glm::vec3(1.0f, 0.0f, 0.0f);
    if (keys[SDL_SCANCODE_D])
      camPos += 0.01f * glm::mat3(m) * glm::vec3(1.0f, 0.0f, 0.0f);

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
        cudaError_t cudaStatus = renderScreen(cudaSurfaceObject, screen_res, SDL_GetTicks() / 1000.0f, sceneDevPtr, camPos, m);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "renderScreen failed!");
            return 1;
        }
      }
      cudaDestroySurfaceObject(cudaSurfaceObject);
    }
    cudaGraphicsUnmapResources(1, &cudaResource);

    cudaStreamSynchronize(0);

    // render texture to screen
    glBindTexture(GL_TEXTURE_2D, tex);
    glUseProgram(program);
    drawFullscreenQuad();

    SDL_GL_SwapWindow(window);

    // check for opengl errors
    GLuint err = glGetError();
    if (err != GL_NO_ERROR) {
      fprintf(stderr, "OpenGL error: %x\n", err);
    }

    // update fps counter
    ++frames;
    uint32_t time = SDL_GetTicks();
    if (time - lastUpdate >= 1000) {
      // add 1000 to fps, or more than 1000 if more than a second elapsed somehow..
      lastUpdate += 1000 * ((time - lastUpdate) / 1000);
      fps = frames;
      frames = 0;
      printf("fps: %d\n", fps);
    }
  }

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

  // Clean up SDL
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  return 0;
}

void drawFullscreenQuad()
{
  static bool init = false;
  static GLuint vbo;

  // initialise vbo
  if (!init) {
    init = true;

    // 4 values per vertex (x, y, u, v)
    float quad[] = { -1.0f, -1.0f, 0.0f, 1.0f
                   ,  1.0f, -1.0f, 1.0f, 1.0f
                   ,  1.0f,  1.0f, 1.0f, 0.0f
                   , -1.0f,  1.0f, 0.0f, 0.0f
                   };

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
  }

  // Actually draw it
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, false, 4 * sizeof(float), nullptr);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, false, 4 * sizeof(float), reinterpret_cast<void*>(2*sizeof(float)));

  glDrawArrays(GL_QUADS, 0, 4);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);
}

GLuint createTexture(int width, int height)
{
  GLuint id;

  glGenTextures(1, &id);
  glBindTexture(GL_TEXTURE_2D, id);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


  return id;
}

bool checkCompileSuccess(GLuint shader, std::string& err)
{
  GLint compiled;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

  if (compiled) {
    return true;
  }
  else {
    GLint len;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);

    if (len > 1)
    {
      GLint olen;
      char* log = (char*)malloc(len);
      glGetShaderInfoLog(shader, len, &olen, log);
      if (olen > 0)
        err = log;
      free(log);
    }

    return false;
  }
}

bool checkLinkSuccess(GLuint program, std::string& err)
{
  GLint compiled;
  glGetProgramiv(program, GL_LINK_STATUS, &compiled);

  if (compiled) {
    return true;
  }
  else {
    GLint len;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);

    if (len > 1)
    {
      GLint olen;
      char* log = (char*)malloc(len);
      glGetProgramInfoLog(program, len, &olen, log);
      if (olen > 0)
        err = log;
      free(log);
    }

    return false;
  }
}

GLuint compileShaderProgram(const char* filename)
{
  std::string err;

  // Load shader source
  std::ifstream f(filename);
  if (!f) {
    fprintf(stderr, "Failed to open shader file %s\n", filename);
    return 0;
  }
  std::stringstream buf;
  buf << f.rdbuf();
  std::string source = buf.str();

  // Compile vertex shader
  const char* vertexSource[3] = { "#version 330 core\n", "#define COMPILING_VERTEX_SHADER\n", source.c_str() };
  GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex, 3, vertexSource, nullptr);
  glCompileShader(vertex);

  if (!checkCompileSuccess(vertex, err)) {
    fprintf(stderr, "Vertex shader %s failed to compile: %s\n", filename, err.c_str());
    return 0;
  }

  // Compile fragment shader
  const char* fragmentSource[3] = { "#version 330 core\n", "#define COMPILING_FRAGMENT_SHADER\n", source.c_str() };
  GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment, 3, fragmentSource, nullptr);
  glCompileShader(fragment);

  if (!checkCompileSuccess(fragment, err)) {
    fprintf(stderr, "Fragment shader %s failed to compile: %s\n", filename, err.c_str());
    return 0;
  }

  // Link shader program
  GLuint program = glCreateProgram();
  glAttachShader(program, vertex);
  glAttachShader(program, fragment);
  glLinkProgram(program);

  if (!checkLinkSuccess(program, err)) {
    fprintf(stderr, "Shader program %s failed to link: %s\n", filename, err.c_str());
    glDeleteProgram(program);
    return 0;
  }

  return program;
}
