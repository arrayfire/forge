#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 13) out;

in VS_OUT {
    vec4 color;
    vec3 dir;
} gs_in[];

out vec4 pervcol;

mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    return mat4(oc * axis.x * axis.x + c,          oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c,          oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c,           0.0,
                0.0,                               0.0,                               0.0,                                1.0);
}

void build_house(vec4 position)
{
    pervcol = gs_in[0].color; // gs_in[0] since there's only one input vertex
    gl_Position = position + vec4(-0.2f, -0.2f, 0.0f, 0.0f);    // 1:bottom-left
    EmitVertex();
    gl_Position = position + vec4( 0.2f, -0.2f, 0.0f, 0.0f);    // 2:bottom-right
    EmitVertex();
    gl_Position = position + vec4(-0.2f,  0.2f, 0.0f, 0.0f);    // 3:top-left
    EmitVertex();
    gl_Position = position + vec4( 0.2f,  0.2f, 0.0f, 0.0f);    // 4:top-right
    EmitVertex();
    gl_Position = position + vec4( 0.0f,  0.4f, 0.0f, 0.0f);    // 5:top
    pervcol = vec4(1.0, 1.0, 1.0, 1.0);
    EmitVertex();
    EndPrimitive();
}


void main()
{
    build_house(gl_in[0].gl_Position);
}
