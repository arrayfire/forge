#version 330

uniform sampler2D tex;
uniform vec4 textColor;

in vec2 texCoord;
out vec4 outputColor;

void main()
{
    vec4 texC = texture(tex, texCoord);
    vec4 alpha = vec4(1.0, 1.0, 1.0, texC.r);
    outputColor = alpha*textColor;
}
