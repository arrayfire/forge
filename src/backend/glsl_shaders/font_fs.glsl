#version 330

uniform sampler2D tex;
uniform vec4 textColor;

in vec2 texCoord;
out vec4 outputColor;

void main()
{
    vec4 texC   = texture(tex, texCoord);
    outputColor = vec4(textColor.rgb, textColor.a*texC.r);
}
