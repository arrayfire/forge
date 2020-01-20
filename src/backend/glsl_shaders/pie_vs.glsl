#version 330

uniform float maxValue;

in vec2 range;
in vec3 color;
in float alpha;

out VS_OUT {
    vec4 color;
    vec2 normRange;
} vs_out;

void main(void)
{
   vs_out.color = vec4(color, alpha);
   vs_out.normRange = range / maxValue;
}
