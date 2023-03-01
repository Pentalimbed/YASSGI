#include "ReShade.fxh"

#include "Input.fxh"
#include "Poisson.fxh"

namespace YASSGI
{


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Constants
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#define PI 3.14159265358979323846264
#define HALF_PI 1.57079632679

#define EPS 1e-6

#ifndef YASSGI_RENDER_SCALE
#   define YASSGI_RENDER_SCALE 0.5
#endif

#define YASSGI_GI_BUFFER_WIDTH (BUFFER_WIDTH * YASSGI_RENDER_SCALE)
#define YASSGI_GI_BUFFER_HEIGHT (BUFFER_HEIGHT * YASSGI_RENDER_SCALE)
#define YASSGI_GI_BUFFER_SIZE (BUFFER_SIZE * YASSGI_RENDER_SCALE)

#ifndef YASSGI_MIP_LEVEL
#   define YASSGI_MIP_LEVEL 5
#endif

// size of int, don't change.
#define YASSGI_BITMASK_SIZE 32
#define YASSGI_SECTOR_ANGLE (PI / YASSGI_BITMASK_SIZE)

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Uniform Varibales
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

uniform uint  iFrameCount  < source = "framecount"; >;
uniform float fFrameTime   < source = "frametime";  >;

uniform int iViewMode <
	ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "YASSGI\0Depth / Normal\0GI / AO (Raw)\0GI / AO (Accumulated)";
> = 0;

// <---- Input ---->

uniform float2 fDepthRange <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "Weapon/Sky Depth Range";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
> = float2(0.000, 0.999);

uniform float fWeapDepthMult <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "Weapon Depth Multiplier";
    ui_min = 1.0; ui_max = 100.0;
    ui_step = 0.1;
> = 1.0;

// <---- GI ---->

uniform float fZThickness <
    ui_type = "slider";
    ui_category = "GI";
    ui_label = "Z Thickness";
    ui_min = 0.0; ui_max = 20.0;
    ui_step = 0.1;
> = 2.0;

#define iNumSlices 1

// uniform uint iNumSlices <
//     ui_type = "slider";
//     ui_category = "GI";
//      ui_label = "Slice Amount";
//     ui_min = 1; ui_max = 4;
//     ui_step = 1;
// > = 4;

uniform uint iNumSteps <
    ui_type = "slider";
    ui_category = "GI";
    ui_label = "Steps per Slice";
    ui_min = 1; ui_max = 16;
    ui_step = 1;
> = 16;

uniform uint fBaseStride <
    ui_type = "slider";
    ui_category = "GI";
    ui_label = "Base Stride";
    ui_min = 1; ui_max = 10.0;
    ui_step = 0.1;
> = 2.0;

uniform uint fStridePower <
    ui_type = "slider";
    ui_category = "GI";
    ui_label = "Exponential Stride";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.7;

// <---- Temporal Accumulation ---->

uniform int iMaxAccumFrames <
    ui_type = "slider";
    ui_category = "Accumulation";
    ui_label = "Max Accumulated Frames";
    ui_min = 1; ui_max = 64;
    ui_step = 1;
> = 32;

uniform float fZSensitivity <
    ui_type = "slider";
    ui_category = "Accumulation";
    ui_label = "Z Sensitivity";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.2;

uniform float fNormalSensitivity <
    ui_type = "slider";
    ui_category = "Temporal Accumulation";
    ui_label = "Normal Sensitivity";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.4;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Buffers
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// current and history z
texture tex_z  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; MipLevels = YASSGI_MIP_LEVEL;};
sampler samp_z {Texture = tex_z;};

// current and history normal (packed)
texture tex_pk_normal  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_pk_normal {Texture = tex_pk_normal;};

// gi + ao
texture tex_gi_ao  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao {Texture = tex_gi_ao;};

texture tex_gi_ao_accum  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao_accum {Texture = tex_gi_ao_accum;};

texture tex_accum_speed  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = R16F; MipLevels = 1;};
sampler samp_accum_speed {Texture = tex_accum_speed;};

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

float fmod(float a, float b)
{
	float c = frac(abs(a / b)) * abs(b);
	return a < 0 ? -c : c;
}

// src: [Drobot2014] Low Level Optimizations for GCN
float fastsqrt(float x)
{
    return asfloat(0x1FBD1DF5 + (asint(x) >> 1));
}

// src: [Eberly2014] GPGPU Programming for Games and Science
float fastacos(float x)
{
    float res = -0.156583 * abs(x) + HALF_PI;
    res *= fastsqrt(1.0 - abs(x));
    return x >= 0 ? res : PI - res;
}

// return [tangent, bi-tangent, normal]
float3x3 getBasis( float3 normal )
{
    float sz = sign( normal.z );
    float a  = 1.0 / ( sz + normal.z );
    float ya = normal.y * a;
    float b  = normal.x * ya;
    float c  = normal.x * sz;

    float3 T = float3( c * normal.x * a - 1.0, sz * b, c );
    float3 B = float3( b, normal.y * ya - sz, normal.y );

    // Note: due to the quaternion formulation, the generated frame is rotated by 180 degrees,
    // s.t. if N = (0, 0, 1), then T = (-1, 0, 0) and B = (0, -1, 0).
    return float3x3( T, B, normal );
}

float getAngle(float3 v1, float3 v2)
{
    return fastacos(dot(normalize(v1), normalize(v2)));
}
float getCoordAngle(float3 x, float3 y, float3 ivec)
{
    return sign(dot(y, ivec)) * getAngle(x, ivec);
}

/// src https://zhuanlan.zhihu.com/p/390862782
float rand4dTo1d(float4 value, float a, float4 b)
{
    float4 small_val = sin(value);
    float random = dot(small_val, b);
    random = frac(sin(random) * a);
    return random;
}
float3 rand4dTo3d(float4 value){
    return float3(
        rand4dTo1d(value, 14375.5964, float4(15.637, 76.243, 37.168, 83.511)),
        rand4dTo1d(value, 14684.6034, float4(45.366, 23.168, 65.918, 57.514)),
        rand4dTo1d(value, 14985.1739, float4(62.654, 88.467, 25.111, 61.875))
    );
}

void singleSample(
    float2 uv, float3 pos_origin, float3 normal_proj, float3 tangent,
    inout bool bitmask[YASSGI_BITMASK_SIZE],
    out float3 ray_offset, out uint shaded_bits
){
    shaded_bits = 0;

    float3 pos_front = Input::uvToViewSpace(uv.xy);
    float3 pos_back = pos_front + normalize(pos_front) * fZThickness;

    ray_offset = pos_front - pos_origin;

    float2 angles = float2(getCoordAngle(tangent, normal_proj, ray_offset),
                           getCoordAngle(tangent, normal_proj, pos_back - pos_origin));
    [branch]
    if(angles.x < EPS)
        return;
    angles = (min(angles.x, angles.y), max(angles.x, angles.y));

    float sector = PI / YASSGI_BITMASK_SIZE;
    float2 sector_range = float2(0, YASSGI_SECTOR_ANGLE);
    [unroll]
    for(int i = 0; i < YASSGI_BITMASK_SIZE; ++i)
    {
        bool occluded = (sector_range.x + sector * 0.5 <= angles.y) && (sector_range.y > angles.x + sector * 0.5);
        shaded_bits += occluded && !bitmask[i];
        bitmask[i] = bitmask[i] || occluded;
        sector_range += YASSGI_SECTOR_ANGLE;
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pixel Shaders
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void PS_InputSetup(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float2 z : SV_Target0, out float4 pk_normal : SV_Target1)
{
    z = float2(Input::getZ(uv), tex2D(samp_z, uv).x);
    float3 normal = Input::getViewNormalAccurate(uv);

    if(Input::zToLinearDepth(z.x) < fDepthRange.x)
    {
        z.x = ((z.x - 1) * fWeapDepthMult) + 1;
        normal = normalize(normal * float3(1, 1, rcp(fWeapDepthMult)));
    }

    pk_normal = float4(Input::packNormals(normal), tex2D(samp_pk_normal, uv).xy);
}

void PS_BitMaskGI(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 gi_ao : SV_Target0
)
{
    float3 gi = 0;
    float ao = 0;

    // direction, stride, ...
    float3 jitter = rand4dTo3d(float4(uv, iFrameCount / PI, 1));

    float3 pos = Input::uvToViewSpace(uv);
    float3 normal = Input::unpackNormals(tex2D(samp_pk_normal, uv).xy);

    [branch]
    if(Input::zToLinearDepth(pos.z) > fDepthRange.y || Input::zToLinearDepth(pos.z) < 0)
    {
        gi_ao = 0;
        return;
    }

    // Adding more slices makes it way slower to run and compile
    // [unroll]
    // for(int slice = 0; slice < iNumSlices; ++slice)
    // {
        float3 dir = float3(1, 0, 0);
        sincos((jitter.x * 2 + rcp(iNumSlices)) * PI, dir.y, dir.x);
        float2 step = dir.xy * fBaseStride * ReShade::PixelSize * (1 + (jitter.y - 0.5) * 0.5);

        float3 normal_plane = normalize(cross(pos, dir));
        float3 normal_proj = normalize(normal - normal_plane * dot(normal, normal_plane));
        float3 tangent = normalize(cross(normal_plane, normal_proj));

        bool bitmask[YASSGI_BITMASK_SIZE];
        [unroll]
        for(int i = 0; i < YASSGI_BITMASK_SIZE; ++i)
            bitmask[i] = 0;

        float2 uv_offset = 0;
        [loop]
        for(int i = 0; i < iNumSteps * 2; ++i)
        { 
            uv_offset = -uv_offset + (i % 2) * step;  // alternating between + and -
            float2 uv_curr = uv + uv_offset;

            float3 ray_offset;
            uint shaded_bits;
            singleSample(
                uv_curr, pos, normal_proj, tangent,
                bitmask, ray_offset, shaded_bits);

            float3 gi_step = Input::srgbToLinear(tex2Dlod(ReShade::BackBuffer, float4(uv_curr, 0, 0)).rgb);
            gi += gi_step * shaded_bits * saturate(dot(normal, normalize(ray_offset))) * rcp(YASSGI_BITMASK_SIZE);
        }

        [unroll]
        for(int i = 0; i < YASSGI_BITMASK_SIZE; ++i)
            ao += bitmask[i] ? 0 : rcp(YASSGI_BITMASK_SIZE);
    // }
    
    gi_ao = float4(gi * rcp(iNumSlices), ao * rcp(iNumSlices));
}

// void PS_Accumulation(
//     in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
//     out float4 gi_ao_accum : SV_Target0, out float4 o_accum_speed : SV_Target1)
// {
//     float4 gi_accum = tex2D(samp_gi_ao_accum, uv);
//     float accum_speed = tex2Dlod(samp_accum_speed, float4(uv, 1, 0)).x;
//     float4 gi_curr = tex2D(samp_gi_ao, uv);

//     float4 normals = tex2D(samp_pk_normal, uv);
//     float2 zs = tex2D(samp_z, uv).xy;

//     // z & normal disocclusion
//     float4 delta = abs(float4(Input::unpackNormals(normals.zw) - Input::unpackNormals(normals.xy), zs.y - zs.x)) / max(fFrameTime, 1.0);
//     float normal_delta = dot(delta.xyz, delta.xyz);
//     float z_delta = delta.w / zs.x;
//     float quality = exp(-normal_delta * fNormalSensitivity * 1e3 - z_delta * fZSensitivity * 1e3);

//     float accum_speed_new = min(accum_speed * quality + 1, iMaxAccumFrames) ;
//     float4 gi_new = lerp(gi_accum, gi_curr, rcp(accum_speed_new));

//     // finalize
//     gi_ao_accum = gi_new;
//     o_accum_speed = accum_speed_new;
// }

void PS_Display(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target)
{
    [branch]
    if(iViewMode == 0)
    {
        color = tex2D(ReShade::BackBuffer, uv);
    }
    else if(iViewMode == 1)  // Depth / Normal
    {
        if((iFrameCount / 300) % 2)  // Normal
        {
            color = (Input::unpackNormals(tex2D(samp_pk_normal, uv).xy) + 1) * 0.5;
        }
        else  // Depth
        {
            color = Input::zToLinearDepth(tex2D(samp_z, uv).x);
            if(color.r < fDepthRange.x)
                color = float3(color.r / fDepthRange.x, 0, 0);
            else if (color.r > fDepthRange.y)
                color = float3(0.1, 0.5, 1.0);
            color.a = 1;
        }
    }
    else if(iViewMode == 2)  // GI
    {
        color = (iFrameCount / 300) % 2 ? 1 - tex2D(samp_gi_ao, uv).w : Input::linearToSrgb(tex2D(samp_gi_ao, uv).xyz);
    }
    else if(iViewMode == 3)  // GI Accum
    {
        color = (iFrameCount / 300) % 2 ? 1 - tex2D(samp_gi_ao_accum, uv).w : Input::linearToSrgb(tex2D(samp_gi_ao_accum, uv).xyz);
    }
}

technique YASSGI{
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_InputSetup;
        RenderTarget0 = tex_z;
        RenderTarget1 = tex_pk_normal;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_BitMaskGI;
        RenderTarget0 = tex_gi_ao;

        ClearRenderTargets = true;
    }
    // pass {
    //     VertexShader = PostProcessVS;
    //     PixelShader = PS_Accumulation;
    //     RenderTarget0 = tex_gi_ao_accum;
    //     RenderTarget1 = tex_accum_speed;
    // }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Display;
    }
}

}