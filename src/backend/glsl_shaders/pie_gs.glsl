#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 64) out;

const float Pi = 3.1415926535;
const float radius = 0.75;
const int PointsOnCircle = 42;
const vec4 center = vec4(0.0, 0.0, 0.0, 1.0);

uniform mat4 viewMat;

in VS_OUT {
    vec4 color;
    vec2 normRange;
} gs_in[];

out vec4 pervcol;

void main()
{
    pervcol = gs_in[0].color;

    vec2 pos = gs_in[0].normRange;
    float currAngle = 2 * Pi * pos.x;
    int Points = max(int(PointsOnCircle * pos.y), 1);
    float stepAngle = 2 * Pi * pos.y / Points;

    for (int i = 0; i <= Points; ++i) {
        if ((i & 1) == 0) {
            gl_Position = viewMat * center;
            EmitVertex();
        }

        gl_Position = viewMat * vec4(center.xy + radius * vec2(cos(currAngle), sin(currAngle)), 0.0, 1.0);
        EmitVertex();
        currAngle += stepAngle;
    }
    EndPrimitive();
}
