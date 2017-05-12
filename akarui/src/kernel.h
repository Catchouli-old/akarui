#pragma once

cudaError_t renderScreen(cudaSurfaceObject_t surface, dim3 screenRes, float time, glm::vec3* vertexBuf, int vertexCount);