#version 330

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
   gl_Position  = vec4(point.xy, 0.0, 1);
}
