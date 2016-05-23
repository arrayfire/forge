#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 9) out;

uniform mat4 transform;

in VS_OUT {
    vec4 color;
    vec2 dir;
} gs_in[];

out vec4 pervcol;

mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,
                oc * axis.x * axis.y - axis.z * s,
                oc * axis.z * axis.x + axis.y * s,
                0.0,
                oc * axis.x * axis.y + axis.z * s,
                oc * axis.y * axis.y + c,
                oc * axis.y * axis.z - axis.x * s,
                0.0,
                oc * axis.z * axis.x - axis.y * s,
                oc * axis.y * axis.z + axis.x * s,
                oc * axis.z * axis.z + c,
                0.0,
                0.0, 0.0, 0.0, 1.0);
}

void main()
{
    vec4 pos = transform * gl_in[0].gl_Position;
    vec2 dir = gs_in[0].dir;

    mat4 rmat  = rotationMatrix(vec3(0,0,1), atan(dir.y, dir.x));
    mat4 trans = transform * rmat;

    vec4 origin   = pos + trans * vec4(0,0,0,1);                  // (2/5)
    vec4 leftTip  = pos + trans * vec4(-0.333, -0.333, 0.0, 1.0); // (1)
    vec4 leftTop  = pos + trans * vec4(-0.167, -0.333, 0.0, 1.0); // (3/7)
    vec4 leftBot  = pos + trans * vec4(-0.167, -1.000, 0.0, 1.0); // (9)
    vec4 rightBot = pos + trans * vec4( 0.167, -1.000, 0.0, 1.0); // (8)
    vec4 rightTop = pos + trans * vec4( 0.167, -0.333, 0.0, 1.0); // (6)
    vec4 rightTip = pos + trans * vec4( 0.333, -0.333, 0.0, 1.0); // (4)

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
