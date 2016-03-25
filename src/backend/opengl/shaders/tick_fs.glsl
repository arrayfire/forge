#version 330

uniform bool isYAxis;
uniform vec4 tick_color;

out vec4 outputColor;

void main(void)
{
   bool y_axis = isYAxis && abs(gl_PointCoord.y-0.5)>0.12;
   bool x_axis = !isYAxis && abs(gl_PointCoord.x-0.5)>0.12;
   if(y_axis || x_axis)
       discard;
   else
       outputColor = tick_color;
}
