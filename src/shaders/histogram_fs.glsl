#version 330

uniform bool isPVCOn;
uniform bool isPVAOn;
uniform vec4 barColor;

in vec4 pervcol;
out vec4 outColor;

void main(void)
{
   float a  = isPVAOn ? pervcol.w : 1.0;
   outColor = isPVCOn ? vec4(pervcol.xyz, a) : barColor;
}
