#version 330

uniform bool isPVCOn;
uniform bool isPVAOn;
uniform vec4 barColor;

in vec4 pervcol;
out vec4 outColor;

void main(void)
{
   outColor = vec4(isPVCOn ? pervcol.xyz : barColor.xyz, isPVAOn ? pervcol.w : 1.0);
}
