#pragma once

#include <SDL2/SDL_opengl.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tclap/CmdLine.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
//#include "cuda_gl_interop.h"
#include "kernel.h"
#include "Scene.h"
#include "imgui.h"

#include "tiny_obj_loader.h"
#include <functional>

void drawFullscreenQuad();
GLuint createTexture(int width, int height);
GLuint compileShaderProgram(const char* filename);

int runInWindow(int argc, char** argv, glm::ivec3 screen_res, glm::vec3 camPos,
  std::function<void()> initFunc, std::function<void(glm::vec3, glm::mat3)> drawFunc)
{
  // parse command line
  TCLAP::CmdLine cmd("akarui", ' ', "0.0");
  TCLAP::ValueArg<bool> vsync("v", "vsync", "Whether to run with vsync on or off", false, true, "bool");
  cmd.add(vsync);
  cmd.parse(argc, argv);

  // initialise SDL
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window* window;
  SDL_Renderer* renderer;
  if (SDL_CreateWindowAndRenderer(screen_res.x, screen_res.y, SDL_WINDOW_OPENGL, &window, &renderer) != 0) {
    fprintf(stderr, "Failed to create window");
    return 1;
  }
  //SDL_SetRelativeMouseMode(SDL_TRUE);

  // initialise opengl
  SDL_GL_CreateContext(window);
  SDL_GL_SetSwapInterval(vsync.getValue() ? 1 : 0);
  glewInit();

  // initialise - after gl context created
  initFunc();

  // initialise imgui
  {
    // settings
    auto& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)screen_res.x, (float)screen_res.y);
    io.KeyMap[ImGuiKey_Tab] = SDLK_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = SDL_SCANCODE_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = SDL_SCANCODE_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = SDL_SCANCODE_UP;
    io.KeyMap[ImGuiKey_DownArrow] = SDL_SCANCODE_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = SDL_SCANCODE_PAGEUP;
    io.KeyMap[ImGuiKey_PageDown] = SDL_SCANCODE_PAGEDOWN;
    io.KeyMap[ImGuiKey_Home] = SDL_SCANCODE_HOME;
    io.KeyMap[ImGuiKey_End] = SDL_SCANCODE_END;
    io.KeyMap[ImGuiKey_Delete] = SDLK_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = SDLK_BACKSPACE;
    io.KeyMap[ImGuiKey_Enter] = SDLK_RETURN;
    io.KeyMap[ImGuiKey_Escape] = SDLK_ESCAPE;
    io.KeyMap[ImGuiKey_A] = SDLK_a;
    io.KeyMap[ImGuiKey_C] = SDLK_c;
    io.KeyMap[ImGuiKey_V] = SDLK_v;
    io.KeyMap[ImGuiKey_X] = SDLK_x;
    io.KeyMap[ImGuiKey_Y] = SDLK_y;
    io.KeyMap[ImGuiKey_Z] = SDLK_z;
  }

  // load fonts
  GLuint fontTex;
  {
    // textures
    auto& io = ImGui::GetIO();
    fontTex = createTexture(0, 0);
    unsigned char* pix;
    int w, h, bpp;
    io.Fonts->GetTexDataAsRGBA32(&pix, &w, &h, &bpp);
    glBindTexture(GL_TEXTURE_2D, fontTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix);
    io.Fonts->TexID = (void*)(intptr_t)fontTex;
  }

  // gui render function
  {
    auto& io = ImGui::GetIO();
    io.RenderDrawListsFn = [](ImDrawData* d) {
      // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
      ImGuiIO& io = ImGui::GetIO();
      int fb_width = (int)(io.DisplaySize.x * io.DisplayFramebufferScale.x);
      int fb_height = (int)(io.DisplaySize.y * io.DisplayFramebufferScale.y);
      if (fb_width == 0 || fb_height == 0)
        return;
      d->ScaleClipRects(io.DisplayFramebufferScale);

      // We are using the OpenGL fixed pipeline to make the example code simpler to read!
      // Setup render state: alpha-blending enabled, no face culling, no depth testing, scissor enabled, vertex/texcoord/color pointers.
      GLint last_texture; glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);
      GLint last_viewport[4]; glGetIntegerv(GL_VIEWPORT, last_viewport);
      GLint last_scissor_box[4]; glGetIntegerv(GL_SCISSOR_BOX, last_scissor_box);
      glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_TRANSFORM_BIT);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glDisable(GL_CULL_FACE);
      glDisable(GL_DEPTH_TEST);
      glEnable(GL_SCISSOR_TEST);
      glEnableClientState(GL_VERTEX_ARRAY);
      glEnableClientState(GL_TEXTURE_COORD_ARRAY);
      glEnableClientState(GL_COLOR_ARRAY);
      glEnable(GL_TEXTURE_2D);
      //glUseProgram(0); // You may want this if using this code in an OpenGL 3+ context

      // Setup viewport, orthographic projection matrix
      glViewport(0, 0, (GLsizei)fb_width, (GLsizei)fb_height);
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      glOrtho(0.0f, io.DisplaySize.x, io.DisplaySize.y, 0.0f, -1.0f, +1.0f);
      glMatrixMode(GL_MODELVIEW);
      glPushMatrix();
      glLoadIdentity();

      // Render command lists
#define OFFSETOF(TYPE, ELEMENT) ((size_t)&(((TYPE *)0)->ELEMENT))
      for (int n = 0; n < d->CmdListsCount; n++)
      {
        const ImDrawList* cmd_list = d->CmdLists[n];
        const ImDrawVert* vtx_buffer = cmd_list->VtxBuffer.Data;
        const ImDrawIdx* idx_buffer = cmd_list->IdxBuffer.Data;
        glVertexPointer(2, GL_FLOAT, sizeof(ImDrawVert), (const GLvoid*)((const char*)vtx_buffer + OFFSETOF(ImDrawVert, pos)));
        glTexCoordPointer(2, GL_FLOAT, sizeof(ImDrawVert), (const GLvoid*)((const char*)vtx_buffer + OFFSETOF(ImDrawVert, uv)));
        glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(ImDrawVert), (const GLvoid*)((const char*)vtx_buffer + OFFSETOF(ImDrawVert, col)));

        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
        {
          const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
          if (pcmd->UserCallback)
          {
            pcmd->UserCallback(cmd_list, pcmd);
          }
          else
          {
            glBindTexture(GL_TEXTURE_2D, (GLuint)(intptr_t)pcmd->TextureId);
            glScissor((int)pcmd->ClipRect.x, (int)(fb_height - pcmd->ClipRect.w), (int)(pcmd->ClipRect.z - pcmd->ClipRect.x), (int)(pcmd->ClipRect.w - pcmd->ClipRect.y));
            glDrawElements(GL_TRIANGLES, (GLsizei)pcmd->ElemCount, sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT, idx_buffer);
          }
          idx_buffer += pcmd->ElemCount;
        }
      }
#undef OFFSETOF

      // Restore modified state
      glDisableClientState(GL_COLOR_ARRAY);
      glDisableClientState(GL_TEXTURE_COORD_ARRAY);
      glDisableClientState(GL_VERTEX_ARRAY);
      glBindTexture(GL_TEXTURE_2D, (GLuint)last_texture);
      glMatrixMode(GL_MODELVIEW);
      glPopMatrix();
      glMatrixMode(GL_PROJECTION);
      glPopMatrix();
      glPopAttrib();
      glViewport(last_viewport[0], last_viewport[1], (GLsizei)last_viewport[2], (GLsizei)last_viewport[3]);
      glScissor(last_scissor_box[0], last_scissor_box[1], (GLsizei)last_scissor_box[2], (GLsizei)last_scissor_box[3]);
    };
  }

  // fps counter
  int frames = 0;
  int fps = 0;
  uint32_t lastUpdate = SDL_GetTicks();

  // camera params
  glm::vec2 camRot(0.0f);

  // input state
  bool leftMouseDown = false;
  float mouseWheel = 0.0f;

  // main loop
  bool running = true;
  Uint32 lastTime = SDL_GetTicks();
  while (running) {
    // handle events
    SDL_Event evt;
    while (SDL_PollEvent(&evt)) {
      if (evt.type == SDL_QUIT || (evt.type == SDL_KEYDOWN && evt.key.keysym.scancode == SDL_SCANCODE_ESCAPE)) {
        running = false;
      }
      else if (evt.type == SDL_MOUSEMOTION && leftMouseDown) {
        if (abs(evt.motion.xrel) > 10 || abs(evt.motion.yrel) > 10)
          continue;
        camRot.y -= (float)evt.motion.xrel;
        camRot.x -= (float)evt.motion.yrel;
        camRot.x = glm::clamp(camRot.x, -90.0f, 90.0f);
      }
      else if (evt.type == SDL_MOUSEWHEEL) {
        mouseWheel += evt.wheel.y * 0.5f;
      }
      else if (evt.type == SDL_MOUSEBUTTONDOWN) {
        if (evt.button.button == SDL_BUTTON_LEFT && !ImGui::IsMouseHoveringAnyWindow())
          leftMouseDown = true;
      }
      else if (evt.type == SDL_MOUSEBUTTONUP) {
        if (evt.button.button == SDL_BUTTON_LEFT)
          leftMouseDown = false;
      }
      else if (evt.type == SDL_TEXTINPUT) {
        auto& io = ImGui::GetIO();
        io.AddInputCharactersUTF8(evt.text.text);
      }
      else if (evt.type == SDL_KEYDOWN || evt.type == SDL_KEYUP) {
        auto& io = ImGui::GetIO();
        io.KeysDown[evt.key.keysym.sym & ~SDLK_SCANCODE_MASK] = (evt.type == SDL_KEYDOWN);
      }
    }

    // update time delta
    Uint32 time = SDL_GetTicks();
    float timeDelta = (time - lastTime) / 1000.0f;
    lastTime = time;

    // get input state
    int mx, my;
    Uint32 mb = SDL_GetMouseState(&mx, &my);
    int numkeys;
    const Uint8* keys = SDL_GetKeyboardState(&numkeys);
    SDL_Keymod mod = SDL_GetModState();

    // update imgui input
    auto& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)mx, (float)my);
    io.MouseDown[0] = (mb & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0;
    io.MouseDown[1] = (mb & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0;
    io.MouseDown[2] = (mb & SDL_BUTTON(SDL_BUTTON_MIDDLE)) != 0;
    io.MouseDown[3] = (mb & SDL_BUTTON(SDL_BUTTON_X1)) != 0;
    io.MouseDown[4] = (mb & SDL_BUTTON(SDL_BUTTON_X2)) != 0;
    io.MouseWheel = mouseWheel;
    io.KeyCtrl = (mod & KMOD_CTRL) != 0;
    io.KeyShift = (mod & KMOD_SHIFT) != 0;
    io.KeyAlt = (mod & KMOD_ALT) != 0;
    io.KeySuper = (mod & KMOD_LGUI) != 0;

    ImGui::NewFrame();

    // update camera
    glm::mat4 camRotMat(1.0f);
    camRotMat = glm::rotate(camRotMat, 0.01f * camRot.y, glm::vec3(0.0f, 1.0f, 0.0f));
    camRotMat = glm::rotate(camRotMat, 0.01f * camRot.x, glm::vec3(1.0f, 0.0f, 0.0f));

    if (!ImGui::GetIO().WantCaptureKeyboard) {
      if (keys[SDL_SCANCODE_W])
        camPos -= timeDelta * glm::mat3(camRotMat) * glm::vec3(0.0f, 0.0f, 1.0f);
      if (keys[SDL_SCANCODE_S])
        camPos += timeDelta * glm::mat3(camRotMat) * glm::vec3(0.0f, 0.0f, 1.0f);
      if (keys[SDL_SCANCODE_A])
        camPos -= timeDelta * glm::mat3(camRotMat) * glm::vec3(1.0f, 0.0f, 0.0f);
      if (keys[SDL_SCANCODE_D])
        camPos += timeDelta * glm::mat3(camRotMat) * glm::vec3(1.0f, 0.0f, 0.0f);
      if (keys[SDL_SCANCODE_Q])
        camPos -= timeDelta * glm::mat3(camRotMat) * glm::vec3(0.0f, 1.0f, 0.0f);
      if (keys[SDL_SCANCODE_E])
        camPos += timeDelta * glm::mat3(camRotMat) * glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // call render function
    drawFunc(camPos, glm::mat3(camRotMat));

    glUseProgram(0);
    ImGui::Render();

    SDL_GL_SwapWindow(window);

    // check for opengl errors
    GLuint err = glGetError();
    if (err != GL_NO_ERROR) {
      fprintf(stderr, "OpenGL error: %x\n", err);
    }
  }

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  //cudaError_t cudaStatus = cudaDeviceReset();
  //if (cudaStatus != cudaSuccess) {
      //fprintf(stderr, "cudaDeviceReset failed!");
      //return 1;
  //}

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
