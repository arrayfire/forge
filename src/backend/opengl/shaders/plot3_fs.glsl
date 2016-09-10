#version 330

uniform vec2 minmaxs[3];
uniform bool isPVCOn;
uniform bool isPVAOn;

in vec4 hpoint;
in vec4 pervcol;

out vec4 outColor;

vec3 hsv2rgb(vec3 c)
{
   vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
   vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
   return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(void)
{
   bool nin_bounds = (hpoint.x > minmaxs[0].y || hpoint.x < minmaxs[0].x
                   || hpoint.y > minmaxs[1].y || hpoint.y < minmaxs[1].x
                   || hpoint.z < minmaxs[2].x);

   float height = (minmaxs[2].y- hpoint.z)/(minmaxs[2].y-minmaxs[2].x);

   float a  = isPVAOn ? pervcol.w : 1.0;

   if(nin_bounds)
       discard;
   else
       outColor = isPVCOn ? vec4(pervcol.xyz, a) : vec4(hsv2rgb(vec3(height, 1, 1)),a);
}
