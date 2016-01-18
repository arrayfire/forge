#version 330

uniform mat4 transform;
uniform vec2 minmaxs[3];

in vec3 point;
in vec3 color;
in float alpha;

out vec4 hpoint;
out vec4 pervcol;

void main(void)
{
   hpoint      = vec4(point.xyz,1);
   pervcol     = vec4(color, alpha);
   gl_Position = transform * vec4(point.xyz, 1);
   gl_PointSize= 10;
}
