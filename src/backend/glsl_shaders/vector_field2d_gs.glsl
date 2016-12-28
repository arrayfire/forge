#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 9) out;

const float PiBy2 = 1.57079632679;

uniform mat4 arrowScaleMat;
uniform mat4 viewMat;

in VS_OUT {
    vec4 color;
    vec2 dir;
} gs_in[];

out vec4 pervcol;

mat4 zrotate(float angle)
{
    float s = sin(angle);
    float c = cos(angle);

    return mat4(c, -s, 0, 0,
                s,  c, 0, 0,
                0,  0, 1, 0,
                0,  0, 0, 1);
}

void main()
{
    vec2 dir  = normalize(gs_in[0].dir);

    mat4 arrowModelMat = arrowScaleMat * zrotate(PiBy2 - atan(dir.y, dir.x));

    vec4 pos  = gl_in[0].gl_Position;

    vec4 origin   = viewMat * (pos + arrowModelMat*vec4( 0.000,  0.333, 0.0, 1.0)); // (2/5)
    vec4 leftTip  = viewMat * (pos + arrowModelMat*vec4(-0.333, -0.333, 0.0, 1.0)); // (1)
    vec4 leftTop  = viewMat * (pos + arrowModelMat*vec4(-0.167, -0.333, 0.0, 1.0)); // (3/7)
    vec4 leftBot  = viewMat * (pos + arrowModelMat*vec4(-0.167, -1.000, 0.0, 1.0)); // (9)
    vec4 rightBot = viewMat * (pos + arrowModelMat*vec4( 0.167, -1.000, 0.0, 1.0)); // (8)
    vec4 rightTop = viewMat * (pos + arrowModelMat*vec4( 0.167, -0.333, 0.0, 1.0)); // (6)
    vec4 rightTip = viewMat * (pos + arrowModelMat*vec4( 0.333, -0.333, 0.0, 1.0)); // (4)

    //                             (2/5)
    //                             //\\
    //                           / /  \ \
    //                         /  /    \  \
    //                       /   /      \   \
    //                    (1)__(3/7)____(6)__(4)
    //                          |\       |
    //                          | \      |
    //                          |  \     |
    //                          |   \    |
    //                          |    \   |
    //                          |     \  |
    //                         (9)_____\(8)

    pervcol = gs_in[0].color;

    gl_Position = leftTip;
    EmitVertex();
    gl_Position = origin;
    EmitVertex();
    gl_Position = leftTop;
    EmitVertex();

    gl_Position = rightTip;
    EmitVertex();

    gl_Position = origin;
    EmitVertex();

    gl_Position = rightTop;
    EmitVertex();

    gl_Position = leftTop;
    EmitVertex();

    gl_Position = rightBot;
    EmitVertex();

    gl_Position = leftBot;
    EmitVertex();

    EndPrimitive();
}
