#version 330

uniform bool isPVCOn;
uniform vec4 barColor;

in vec4 pervcol;
out vec4 outColor;

void main(void)
{
   outColor = isPVCOn ? pervcol : barColor;
}
