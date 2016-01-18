#version 330

const int size = 259;

layout(std140) uniform ColorMap
{
    vec4 ch[size];
};

uniform float cmaplen;
uniform sampler2D tex;
uniform bool isGrayScale;

in vec2 texcoord;
out vec4 fragColor;

void main()
{
    vec4 tcolor = texture(tex, texcoord);
    vec4 clrs = vec4(1, 0, 0, 1);
    if(isGrayScale)
        clrs = vec4(tcolor.r, tcolor.r, tcolor.r, 1);
    else
        clrs = tcolor;
    vec4 fidx  = (cmaplen-1) * clrs;
    ivec4 idx  = ivec4(fidx.x, fidx.y, fidx.z, fidx.w);
    float r_ch = ch[idx.x].r;
    float g_ch = ch[idx.y].g;
    float b_ch = ch[idx.z].b;
    fragColor = vec4(r_ch, g_ch , b_ch, 1);
}
