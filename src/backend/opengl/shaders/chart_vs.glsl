#version 330

uniform mat4 transform;

in vec3 point;

void main(void)
{
   gl_Position = transform * vec4(point.xyz, 1);
}
