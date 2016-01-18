#version 330

uniform mat4 transform;

in vec2 point;
in vec3 color;
in float alpha;

out vec4 pervcol;

void main(void)
{
   pervcol     = vec4(color, alpha);
   gl_Position = transform * vec4(point.xy, 0, 1);
   gl_PointSize= 10;
}
