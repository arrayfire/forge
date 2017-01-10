#version 330

uniform bool isPVCOn;
uniform bool isPVAOn;
uniform int marker_type;
uniform vec4 marker_color;

in vec4 pervcol;
out vec4 outColor;

void main(void)
{
   vec2 coords = gl_PointCoord;
   float dist  = sqrt((coords.x - 0.5)*(coords.x-0.5) + (coords.y-0.5)*(coords.y-0.5));
   bool in_bounds;

   switch(marker_type) {
       case 1:
           in_bounds = dist<0.1;
           break;
       case 2:
           in_bounds = dist<0.5;
           break;
       case 3:
           in_bounds = ((coords.x > 0.15) || (coords.x < 0.85)) ||
                       ((coords.y > 0.15) || (coords.y < 0.85));
           break;
       case 4:
           in_bounds = (2*(coords.x - 0.25) - (coords.y + 0.5) < 0) &&
                       (2*(coords.x - 0.25) + (coords.y + 0.5) > 1);
           break;
       case 5:
           in_bounds = abs((coords.x - 0.5) + (coords.y - 0.5) ) < 0.13  ||
                       abs((coords.x - 0.5) - (coords.y - 0.5) ) < 0.13  ;
           break;
       case 6:
           in_bounds = abs((coords.x - 0.5)) < 0.07 ||
                       abs((coords.y - 0.5)) < 0.07;
           break;
       case 7:
           in_bounds = abs((coords.x - 0.5) + (coords.y - 0.5) ) < 0.07 ||
                       abs((coords.x - 0.5) - (coords.y - 0.5) ) < 0.07 ||
                       abs((coords.x - 0.5)) < 0.07 ||
                       abs((coords.y - 0.5)) < 0.07;
           break;
       default:
           in_bounds = true;
   }

   if(!in_bounds)
       discard;
   else
       outColor = vec4(isPVCOn ? pervcol.xyz : marker_color.xyz, isPVAOn ? pervcol.w : 1.0);
}
