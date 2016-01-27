#version 330

uniform bool isPVCOn;
uniform bool isPVAOn;
uniform int marker_type;
uniform vec4 marker_color;

in vec4 pervcol;
out vec4 outColor;

void main(void)
{
   float dist = sqrt( (gl_PointCoord.x - 0.5) * (gl_PointCoord.x-0.5) + (gl_PointCoord.y-0.5) * (gl_PointCoord.y-0.5) );
   bool in_bounds;
   switch(marker_type) {
       case 1:
           in_bounds = dist < 0.3;
           break;
       case 2:
           in_bounds = ( (dist > 0.3) && (dist<0.5) );
           break;
       case 3:
           in_bounds = ((gl_PointCoord.x < 0.15) || (gl_PointCoord.x > 0.85)) ||
                       ((gl_PointCoord.y < 0.15) || (gl_PointCoord.y > 0.85));
           break;
       case 4:
           in_bounds = (2*(gl_PointCoord.x - 0.25) - (gl_PointCoord.y + 0.5) < 0) && (2*(gl_PointCoord.x - 0.25) + (gl_PointCoord.y + 0.5) > 1);
           break;
       case 5:
           in_bounds = abs((gl_PointCoord.x - 0.5) + (gl_PointCoord.y - 0.5) ) < 0.13  ||
           abs((gl_PointCoord.x - 0.5) - (gl_PointCoord.y - 0.5) ) < 0.13  ;
           break;
       case 6:
           in_bounds = abs((gl_PointCoord.x - 0.5)) < 0.07 ||
           abs((gl_PointCoord.y - 0.5)) < 0.07;
           break;
       case 7:
           in_bounds = abs((gl_PointCoord.x - 0.5) + (gl_PointCoord.y - 0.5) ) < 0.07 ||
           abs((gl_PointCoord.x - 0.5) - (gl_PointCoord.y - 0.5) ) < 0.07 ||
           abs((gl_PointCoord.x - 0.5)) < 0.07 ||
           abs((gl_PointCoord.y - 0.5)) < 0.07;
           break;
       default:
           in_bounds = true;
   }

   float a  = isPVAOn ? pervcol.w : 1.0;

   if(!in_bounds)
       discard;
   else
       outColor = isPVCOn ? vec4(pervcol.xyz, a) : marker_color;
}
