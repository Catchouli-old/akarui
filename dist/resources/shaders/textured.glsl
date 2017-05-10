#extension GL_ARB_explicit_uniform_location : enable

#ifdef COMPILING_VERTEX_SHADER

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;

out vec2 var_uv;

void main() {
  gl_Position = vec4(in_pos, 0.0, 1.0);
  var_uv = in_uv;
}

#endif

#ifdef COMPILING_FRAGMENT_SHADER

in vec2 var_uv;

out vec4 out_colour;

layout(location = 0) uniform sampler2D uni_tex;

void main() {
  out_colour = texture(uni_tex, var_uv);
}

#endif