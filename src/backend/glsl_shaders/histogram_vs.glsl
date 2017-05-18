#version 330

uniform float ymax;
uniform float nbins;
uniform mat4 transform;

in vec2 point;
in float freq;
in vec3 color;
in float alpha;

out vec4 pervcol;

void main(void)
{
   float binId = gl_InstanceID;
   float deltax = 2.0/nbins;
   float deltay = 2.0/ymax;
   float xcurr = -1.0 + binId * deltax;
   if (point.x==1) {
        xcurr  += deltax;
   }
   float ycurr = -1.0;
   if (point.y==1) {
       ycurr += deltay * freq;
   }
   pervcol = vec4(color, alpha);
   gl_Position = transform * vec4(xcurr, ycurr, 0, 1);
}
