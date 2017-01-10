#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 32) out;

const float PiBy2 = 1.57079632679;

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

void emitQuad(vec4 a, vec4 b, vec4 c, vec4 d)
{
    gl_Position = a; EmitVertex();
    gl_Position = b; EmitVertex();
    gl_Position = c; EmitVertex();
    gl_Position = d; EmitVertex();
    EndPrimitive();
}

void main()
{
    vec4 pos    = gl_in[0].gl_Position;
    vec3 dir    = normalize(gs_in[0].dir);

    vec3 originalAxis = vec3(0, 1, 0);
    vec3 planeNormal = cross(originalAxis, dir);
    float angle = acos(dot(originalAxis, dir));
    mat4 rot = rotationMatrix(planeNormal, angle);

    mat4 trans  = arrowScaleMat * rot;

    vec4 t  = viewMat * (pos + trans*vec4( 0.000,  0.300,  0.000, 1.000));

    vec4 a  = viewMat * (pos + trans*vec4(-0.167, -1.000, -0.167, 1.000));
    vec4 b  = viewMat * (pos + trans*vec4( 0.167, -1.000, -0.167, 1.000));
    vec4 c  = viewMat * (pos + trans*vec4(-0.167, -1.000,  0.167, 1.000));
    vec4 d  = viewMat * (pos + trans*vec4( 0.167, -1.000,  0.167, 1.000));

    vec4 i0 = viewMat * (pos + trans*vec4(-0.167, -0.333, -0.167, 1.000));
    vec4 i1 = viewMat * (pos + trans*vec4( 0.167, -0.333, -0.167, 1.000));
    vec4 i2 = viewMat * (pos + trans*vec4(-0.167, -0.333,  0.167, 1.000));
    vec4 i3 = viewMat * (pos + trans*vec4( 0.167, -0.333,  0.167, 1.000));

    vec4 e  = viewMat * (pos + trans*vec4(-0.333, -0.333, -0.333, 1.000));
    vec4 f  = viewMat * (pos + trans*vec4( 0.333, -0.333, -0.333, 1.000));
    vec4 g  = viewMat * (pos + trans*vec4(-0.333, -0.333,  0.333, 1.000));
    vec4 h  = viewMat * (pos + trans*vec4( 0.333, -0.333,  0.333, 1.000));

    pervcol = gs_in[0].color;

    emitQuad(a, b, c, d);
    emitQuad(a, i0, b, i1);
    emitQuad(b, i1, d, i3);
    emitQuad(c, d, i2, i3);
    emitQuad(a, c, i0, i2);
    emitQuad(e, f, g, h);

    gl_Position = g; EmitVertex();
    gl_Position = h; EmitVertex();
    gl_Position = t; EmitVertex();
    gl_Position = f; EmitVertex();
    gl_Position = e; EmitVertex();
    EndPrimitive();

    gl_Position = t; EmitVertex();
    gl_Position = e; EmitVertex();
    gl_Position = g; EmitVertex();
    EndPrimitive();
}
