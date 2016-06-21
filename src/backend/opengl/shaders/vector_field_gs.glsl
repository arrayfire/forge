#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 25) out;

uniform mat4 arrowScaleMat;
uniform mat4 viewMat;

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
    vec4 pos    = gl_in[0].gl_Position;
    vec3 dir    = normalize(gs_in[0].dir);

    float theta = acos(dir.z/ 1.0);
    float phi   = atan(dir.y/ dir.x);

    mat4 zrot   = rotationMatrix(vec3(0,0,1), phi);
    vec4 sndAxs = zrot * vec4(0,1,0,1);
    mat4 arot   = rotationMatrix(sndAxs.xyz, theta);
    mat4 trans  = arrowScaleMat * arot * zrot;

    vec4 origin = viewMat * (pos + trans * vec4(0,0,0,1));
    vec4 ltop   = viewMat * (pos + trans * vec4(-0.333, 0.333, 0.333, 1.0));
    vec4 lbot   = viewMat * (pos + trans * vec4(-0.333,-0.333, 0.333, 1.0));
    vec4 rtop   = viewMat * (pos + trans * vec4(-0.333,-0.333,-0.333, 1.0));
    vec4 rbot   = viewMat * (pos + trans * vec4(-0.333, 0.333,-0.333, 1.0));

    vec4 iltop  = viewMat * (pos + trans * vec4(-0.333, 0.167, 0.167, 1.0));
    vec4 ilbot  = viewMat * (pos + trans * vec4(-0.333,-0.167, 0.167, 1.0));
    vec4 irtop  = viewMat * (pos + trans * vec4(-0.333,-0.167,-0.167, 1.0));
    vec4 irbot  = viewMat * (pos + trans * vec4(-0.333, 0.167,-0.167, 1.0));

    vec4 iltop2 = viewMat * (pos + trans * vec4(-1.000, 0.167, 0.167, 1.0));
    vec4 ilbot2 = viewMat * (pos + trans * vec4(-1.000,-0.167, 0.167, 1.0));
    vec4 irtop2 = viewMat * (pos + trans * vec4(-1.000,-0.167,-0.167, 1.0));
    vec4 irbot2 = viewMat * (pos + trans * vec4(-1.000, 0.167,-0.167, 1.0));

    pervcol = gs_in[0].color;

    // Roof top
    gl_Position = ltop;
    EmitVertex();
    gl_Position = origin;
    EmitVertex();
    gl_Position = lbot;
    EmitVertex();
    gl_Position = rbot;
    EmitVertex();
    gl_Position = origin;
    EmitVertex();
    gl_Position = rtop;
    EmitVertex();
    gl_Position = ltop;
    EmitVertex();
    // over the edge areas of roof
    gl_Position = iltop;
    EmitVertex();
    gl_Position = lbot;
    EmitVertex();
    gl_Position = ilbot;
    EmitVertex();
    gl_Position = rbot;
    EmitVertex();
    gl_Position = irbot;
    EmitVertex();
    gl_Position = rtop;
    EmitVertex();
    gl_Position = irtop;
    EmitVertex();
    gl_Position = iltop;
    EmitVertex();
    // Pillar
    gl_Position = iltop2;
    EmitVertex();
    gl_Position = ilbot;
    EmitVertex();
    gl_Position = ilbot2;
    EmitVertex();
    gl_Position = irbot;
    EmitVertex();
    gl_Position = irbot2;
    EmitVertex();
    gl_Position = irtop;
    EmitVertex();
    gl_Position = irtop2;
    EmitVertex();
    gl_Position = iltop2;
    EmitVertex();
    // Pillar base
    gl_Position = irbot2;
    EmitVertex();
    gl_Position = ilbot2;
    EmitVertex();
    EndPrimitive();
}
