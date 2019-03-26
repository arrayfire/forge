#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 32) out;

const float PiBy2 = 1.57079632679;

uniform float ArrowSize;

in VS_OUT {
    vec4 color;
    vec3 dir;
} gs_in[];

out vec4 pervcol;

// Quaternion based rotation matrix
mat3 rotate(vec3 axis, float angle)
{
    float t  = angle;
    float s  = sin(t);
    float a  = cos(t);
    float b  = s * axis.x;
    float c  = s * axis.y;
    float d  = s * axis.z;
    float as = a * a;
    float bs = b * b;
    float cs = c * c;
    float ds = d * d;
    float bc = b * c;
    float ad = a * d;
    float bd = b * d;
    float ac = a * c;
    float cd = c * d;
    float ab = a * b;

    return mat3(as+bs-cs-ds, 2*bc-2*ad, 2*bd+2*ac,
                2*bc+2*ad, as-bs+cs-ds, 2*cd-2*ab,
                2*bd-2*ac, 2*cd+2*ab, as-bs-cs+ds);
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
    mat3 smat = mat3(ArrowSize, 0, 0, 0, ArrowSize, 0, 0, 0, ArrowSize);
    vec3 pos  = gl_in[0].gl_Position.xyz;
    vec3 dir  = normalize(gs_in[0].dir);

    vec3 ArrowUp = vec3(0, 1, 0);
    vec3 planeNr  = cross(ArrowUp, dir);

    float angle= acos(dot(ArrowUp, dir));
    mat3 rotor = rotate(planeNr, angle);
    mat3 trans = smat * rotor;

    vec4 t  = vec4(pos + trans * vec3( 0.000,  0.300,  0.000), 1.0);

    vec4 a  = vec4(pos + trans * vec3(-0.167, -1.000, -0.167), 1.0);
    vec4 b  = vec4(pos + trans * vec3( 0.167, -1.000, -0.167), 1.0);
    vec4 c  = vec4(pos + trans * vec3(-0.167, -1.000,  0.167), 1.0);
    vec4 d  = vec4(pos + trans * vec3( 0.167, -1.000,  0.167), 1.0);

    vec4 i0 = vec4(pos + trans * vec3(-0.167, -0.333, -0.167), 1.0);
    vec4 i1 = vec4(pos + trans * vec3( 0.167, -0.333, -0.167), 1.0);
    vec4 i2 = vec4(pos + trans * vec3(-0.167, -0.333,  0.167), 1.0);
    vec4 i3 = vec4(pos + trans * vec3( 0.167, -0.333,  0.167), 1.0);

    vec4 e  = vec4(pos + trans * vec3(-0.333, -0.333, -0.333), 1.0);
    vec4 f  = vec4(pos + trans * vec3( 0.333, -0.333, -0.333), 1.0);
    vec4 g  = vec4(pos + trans * vec3(-0.333, -0.333,  0.333), 1.0);
    vec4 h  = vec4(pos + trans * vec3( 0.333, -0.333,  0.333), 1.0);

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
