#pragma once

#include "Scene.h"
#include <glm/glm.hpp>

cudaError_t renderScreen(cudaSurfaceObject_t surface, dim3 screenRes, float time, Scene*, glm::vec3 camPos, const glm::mat3& viewMat);