#version 330

uniform mat4 transform;
uniform bool isPVROn;
uniform float psize;

in vec3 point;
in vec3 color;
in float alpha;
in float pointsize;

out vec4 hpoint;
out vec4 pervcol;

void main(void)
{
   hpoint      = vec4(point.xyz,1);
   pervcol     = vec4(color, alpha);
   gl_Position = transform * hpoint;
   gl_PointSize = isPVROn ? pointsize : psize;
}
