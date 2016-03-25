#version 330

uniform mat4 projectionMatrix;
uniform mat4 modelViewMatrix;

layout (location = 0) in vec2 inPosition;
layout (location = 1) in vec2 inCoord;

out vec2 texCoord;

void main()
{
    gl_Position = projectionMatrix*modelViewMatrix*vec4(inPosition, 0.0, 1.0);
    texCoord = inCoord;
}
