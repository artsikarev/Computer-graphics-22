#pragma once

#include "graphics_internal.hpp"
#include <GLFW/glfw3.h>
namespace application {
void setWindow(GLFWwindow* window);

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

} // namespace application