#version 330

uniform mat4 modelMat;

in vec2 point;
in vec3 color;
in float alpha;
in vec2 direction;

out VS_OUT {
    vec4 color;
    vec2 dir;
} vs_out;

void main(void)
{
   vs_out.color = vec4(color, alpha);
   vs_out.dir   = direction;
   gl_Position  = modelMat * vec4(point.xy, 0.0, 1);
}
