/*  REFERENCES & CREDITS
    
    Redistribution of source material are maked with <src>,
    with a copy of its license appended if it demands so.

    All other code shall be considered licensed under UNLICENSE,
    either as (re)implementation of their source materials,
    or as the author's original work.

    Free blue noise textures. Christoph Peters.
        url:    http://momentsingraphics.de/BlueNoise.html
        credit: blue noise texture. <src>
    RGB COLOURSPACE TRANSFORMATION MATRIX. Colour Developers.
        url:    https://www.colour-science.org/apps/
        credit: ACEScg <-> sRGB conversion matrices
    Physically Based Rendering. Matt Pharr, Wenzel Jakob, and Greg Humphreys.
        url:    https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler
        credit: radicalInverse & hammersly function <src>
        license:
            Copyright (c) 1998-2015, Matt Pharr, Greg Humphreys, and Wenzel Jakob.

            All rights reserved.

            Redistribution and use in source and binary forms, with or without
            modification, are permitted provided that the following conditions are met:

            1. Redistributions of source code must retain the above copyright notice, this
            list of conditions and the following disclaimer.

            2. Redistributions in binary form must reproduce the above copyright notice,
            this list of conditions and the following disclaimer in the documentation
            and/or other materials provided with the distribution.

            THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
            AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
            IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
            DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
            FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
            DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
            SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
            CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
            OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    Hash Functions for GPU Rendering. Mark Jarzynski, Marc Olano, and UMBC.
        url:    http://www.jcgt.org/published/0009/03/02/
                https://www.shadertoy.com/view/XlGcRh
        credit: pcg3d & pcg4d function <src>
    Practical Realtime Strategies for Accurate Indirect Occlusion. Jorge Jimenez, Xian-Chun Wu, Angelo Pesce, and Adrian Jarabo.
        url:    https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
        credit: GTAO algorithm
    XeGTAO. Intel Corporation.
        url:    https://github.com/GameTechDev/XeGTAO
        credit: details of GTAO implementation
                thin object heuristics
    Horizon-Based Indirect Lighting. Benoît "Patapom" Mayaux.
        url:    https://github.com/Patapom/GodComplex/blob/master/Tests/TestHBIL/2018%20Mayaux%20-%20Horizon-Based%20Indirect%20Lighting%20(HBIL).pdf
        credit: interleaved sampling
                calculation of horizon based indirect light
    Legit Engine. Alexander "Raikiri" Sannikov.
        url:    https://github.com/Raikiri/LegitEngine
                multiple of their youtube videos and comments
        credit: motivation
*/
/*  NOTATION

    raw_z: raw z
    depth: linearized z (0-1)
    z: linearized z
    z direction: + going farther, - coming closer
    dir: normalized direction, same as above
    normal: pointing outwards
    color & gi: ACEScg
    pos_v: view
    angle: radian
*/
/*  TODO

    - il
    o bent normal
    o thickness heuristic
    - alternative bitmask impl (?)
    - hi-z buffer w/ cone tracing (?)
    - remove subtle grid like pattern
    - ibl
*/

#include "ReShade.fxh"
#include "ShaderFastMathLib.h"

namespace YASSGI_SKYRIM
{

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Constants
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#define PI      3.14159265358979323846264
#define HALF_PI 1.5707963267948966
#define RCP_PI  0.3183098861837067

#define MAX_UINT_F 4294967295.0

#define EPS 1e-6

#define BUFFER_SIZE uint2(BUFFER_WIDTH, BUFFER_HEIGHT)

#define NOISE_SIZE 256

#define INTERLEAVED_SIZE_PX 4
#define MAX_MIP 8

#ifndef YASSGI_PREBLUR_SCALE
#   define YASSGI_PREBLUR_SCALE 0.25
#endif

#ifndef YASSGI_DISABLE_IL
#   define YASSGI_DISABLE_IL 0
#endif

static const float3x3 g_sRGBToACEScg = float3x3(
    0.613117812906440,  0.341181995855625,  0.045787344282337,
    0.069934082307513,  0.918103037508582,  0.011932775530201,
    0.020462992637737,  0.106768663382511,  0.872715910619442
);
static const float3x3 g_ACEScgToSRGB = float3x3(
    1.704887331049502,  -0.624157274479025, -0.080886773895704,
    -0.129520935348888,  1.138399326040076, -0.008779241755018,
    -0.024127059936902, -0.124620612286390,  1.148822109913262
);

#define g_colorInputMat g_sRGBToACEScg
#define g_colorOutputMat g_ACEScgToSRGB

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Uniform Varibales
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// <---- Backend ---->

uniform float fFarPlane  < source = "Far"; >;
uniform float fNearPlane < source = "Near"; >;

uniform float4x4 fViewMatrix        < source = "ViewMatrix"; >;
uniform float4x4 fProjMatrix        < source = "ProjMatrix"; >;
uniform float4x4 fViewProjMatrix    < source = "ViewProjMatrix"; >;
uniform float4x4 fInvViewProjMatrix < source = "InvViewProjMatrix"; >;
uniform float4x4 fProjMatrixJit        < source = "ProjMatrix Jittered"; >;
uniform float4x4 fViewProjMatrixJit    < source = "ViewProjMatrix Jittered"; >;
uniform float4x4 fInvViewProjMatrixJit < source = "InvViewProjMatrix Jittered"; >;

uniform float fFov < source = "FieldOfView"; >;
uniform float3 fCamPos < source = "Position"; >;

uniform int   iFrameCount < source = "FrameCount"; >;
uniform float fTimer      < source = "TimerReal"; >;
uniform float fFrameTime  < source = "TimingsReal"; >;

// <---- UI ---->

uniform float fDebug <
    ui_type = "slider";
    ui_min = 0; ui_max = 1;
> = 1;

uniform int iViewMode <
	ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "YASSGI\0AO\0IL\0";
> = 0;

const static float2 fZRange = float2(0, 20000);
// uniform float2 fZRange <
//     ui_type = "slider";
//     ui_category = "Input";
//     ui_label = "Weapon/Sky Z Range";
//     ui_min = 0.0; ui_max = 25000;
//     ui_step = 0.1;
// > = float2(100, 20000);

uniform uint iDirCount <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Slices";
    ui_min = 1; ui_max = 4;
> = 2;  

uniform float fBaseStridePx <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Base Stride (px)";
    ui_min = 1; ui_max = 64;
    ui_step = 1;
> = 16;

uniform float fSpreadExp <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Spread Exponent";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.01;
> = 2;

uniform float fStrideJitter <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Stride Jitter";
    ui_min = 0; ui_max = 1;
    ui_step = 0.01;
> = 0.3;

uniform float fMaxSampleDistPx <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Sample Distance (px)";
    ui_min = 2; ui_max = BUFFER_WIDTH;
    ui_step = 1;
> = 400;

uniform float fLodRangePx <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "LOD Range (px)";
    ui_min = 2; ui_max = 64;
    ui_step = 1;
> = 48;

// uniform float fAngleJitterScale <
//     ui_type = "slider";
//     ui_category = "Sampling";
//     ui_label = "Angle Jitter Scale";
//     ui_min = 0; ui_max = 1;
//     ui_step = 0.01;
// > = 0.65;

uniform float fFxRange <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Effect Reach";
    ui_min = 0; ui_max = 100.0;
    ui_step = 0.1;
> = 50.0;

uniform float fFxFalloff <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Effect Falloff";
    ui_min = 0; ui_max = 1.0;
    ui_step = 0.001;
> = 0.7;

uniform float fThinOccluderCompensation <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Thin Obj Compensation";
    ui_min = 0; ui_max = 0.7;
    ui_step = 0.01;
> = 0.7;

uniform float fLightSrcThres <
	ui_type = "slider";
    ui_label = "Light Source Threshold";
    ui_tooltip = "Only pixels brighter than this are considered light-emitting.";
	ui_category = "Visual";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.1;

uniform float fAlbedoSatPower <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Albedo Saturation Power";
    ui_tooltip = "Since ReShade has no way of knowing the true albedo of a surface separate from lighting,\n"
        "any shader has to guess. A value of 0.0 tells the shader that everything is monochrome, and its\n"
        "hue is the result of lighting. Greater value yields more saturated output on colored surfaces.\n";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.01;
> = 1.0;

uniform float fAlbedoNorm <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Albedo Normalization";
    ui_tooltip = "Since ReShade has no way of knowing the true albedo of a surface separate from lighting,\n"
        "any shader has to guess. A value of 0.0 tells the shader that there is no lighting in the scene,\n"
        "so dark surfaces are actually black. 1.0 says that all surfaces are in fact colored brightly, and\n"
        "the variation in brightness are the result of illumination, rather than the texture pattern itself.";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.8;

uniform float fAoStrength <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "AO";
    ui_tooltip = "Negative value for non-physical-accurate exponential mixing.";
    ui_min = -2.0; ui_max = 1.0;
    ui_step = 0.01;
> = -1.5;

uniform float fIlStrength <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "IL";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.01;
> = 2.0;

}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Buffers
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

namespace Skyrim
{
texture tex_normal : NORMAL_TAAMASK_SSRMASK;
sampler samp_normal { Texture = tex_normal; };

texture tex_depth : TARGET_MAIN_DEPTH;
sampler samp_depth { Texture = tex_depth; };
}

namespace YASSGI_SKYRIM
{
texture tex_blue <source = "YASSGI_bleu.png";> {Width = NOISE_SIZE; Height = NOISE_SIZE; Format = RGBA8;};
sampler samp_blue                              {Texture = tex_blue; AddressU = REPEAT; AddressV = REPEAT; AddressW = REPEAT;};

// downscaled normal (RGB) raw_z (A)
// tbf orig paper uses interleaved buffers (not blurred) but no sampler spamming for dx9.
//                                          (or perhaps a separate shader(s) for those?)
// Edit: I do think we need those for maximal performance.
texture tex_blur_normal_z  {Width = BUFFER_WIDTH * YASSGI_PREBLUR_SCALE; Height = BUFFER_HEIGHT * YASSGI_PREBLUR_SCALE; Format = RGBA32F; MipLevels = MAX_MIP;};
sampler samp_blur_normal_z {Texture = tex_blur_normal_z;};

texture tex_blur_color  {Width = BUFFER_WIDTH * YASSGI_PREBLUR_SCALE; Height = BUFFER_HEIGHT * YASSGI_PREBLUR_SCALE; Format = RGBA16F; MipLevels = MAX_MIP;};
sampler samp_blur_color {Texture = tex_blur_color;};

// texture tex_bent_normal  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
// sampler samp_bent_normal {Texture = tex_bent_normal;};

texture tex_il_ao  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = 1;};
sampler samp_il_ao {Texture = tex_il_ao;};

// texture tex_il_ao_ac1  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = 1;};
// sampler samp_il_ao_ac1 {Texture = tex_il_ao_ac1;};

// texture tex_il_ao_ac2  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = 1;};
// sampler samp_il_ao_ac2 {Texture = tex_il_ao_ac2;};
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// <---- Input ---->

float3 unpackNormal(float2 enc)
{
	float2 fenc = enc * 4 - 2;
    float f = dot(fenc, fenc);
    float g = sqrt(1 - f * 0.25);
    float3 n = float3(fenc * g, 1 - f * 0.5);
    return n * float3(1, -1, -1);  // for my habit
}

float rawZToLinear01(float raw_z)
{
    return raw_z / (fFarPlane - raw_z * (fFarPlane - fNearPlane));
}

float3 uvzToWorld(float2 uv, float raw_z)
{
    float4 pos_s = float4((uv * 2 - 1) * float2(1, -1) * raw_z, raw_z, 1);
    float4 pos = mul(transpose(fInvViewProjMatrixJit), pos_s);
    return pos.xyz / pos.w;
}
float3 uvzToView(float2 uv, float raw_z)
{
    float4x4 inv_proj = mul(fInvViewProjMatrixJit, fViewMatrix);
    float4 pos_view = mul(transpose(inv_proj), float4((uv * 2 - 1) * float2(1, -1) * raw_z, raw_z, 1));
    return pos_view.xyz / pos_view.w;
}
float3 viewToUvz(float3 pos){
    float4 pos_clip = mul(transpose(fProjMatrixJit), float4(pos, 1));
    pos_clip.xyz /= pos_clip.w;
    pos_clip.xy = (pos_clip.xy / pos_clip.z * float2(1, -1) + 1) * 0.5;
    return pos_clip.xyz;
}

bool isNear(float z){return z < fNearPlane;}
bool isFar(float z){return z > fFarPlane;}
bool isWeapon(float z){return z < fZRange.x;}
bool isSky(float z){return z > fZRange.y;}

float3 fakeAlbedo(float3 color)
{
    float3 albedo = pow(max(0, color), fAlbedoSatPower * length(color));  // length(color) suppress saturation of darker colors
    albedo = saturate(lerp(albedo, normalize(albedo), fAlbedoNorm));
    return albedo;
}

// <---- Sampling ---->

float radicalInverse(uint i)
{
    uint bits = (i << 16u) | (i >> 16u);
    bits = (bits & 0x55555555u) << 1u | (bits & 0xAAAAAAAAu) >> 1u;
    bits = (bits & 0x33333333u) << 2u | (bits & 0xCCCCCCCCu) >> 2u;
    bits = (bits & 0x0F0F0F0Fu) << 4u | (bits & 0xF0F0F0F0u) >> 4u;
    bits = (bits & 0x00FF00FFu) << 8u | (bits & 0xFF00FF00u) >> 8u;
    return bits / MAX_UINT_F;  // can't use 0xffffffff for some reason
}
float2 hammersley(uint i, uint N) {return float2(float(i) / N, radicalInverse(i));}

uint3 pcg3d(uint3 v) {

    v = v * 1664525u + 1013904223u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v ^= v >> 16u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    return v;
}
uint4 pcg4d(uint4 v)
{
    v = v * 1664525u + 1013904223u;
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    v ^= v >> 16u;
    
    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;
    
    return v;
}

// <---- Misc ---->

bool isInScreen(float2 uv) { return all(uv >= 0) && all(uv < 1); }

// box = minx, miny, maxx, maxy
// retval = nearest / farthest traversal distance (can be negative)
// use retval.x < retval.y to check if truly intersected
float2 intersectBox(float2 orig, float2 dir, float4 box)
{
    float4 dists = (box - orig.xyxy) * rcp(dir).xyxy;
    return float2(max(min(dists.x, dists.z), min(dists.y, dists.w)),
                  min(max(dists.x, dists.z), max(dists.y, dists.w)));
}

float3 projectToPlane(float3 v, float3 normal)
{
    return v - dot(v, normal) * normal;
}

// HBIL pp.29
float ilIntegral(float nx, float ny, float cos_prev, float cos_new)
{
    float delta_angle = acosFast4(cos_prev) - acosFast4(cos_new);
    float sin_prev = sqrt(1 - cos_prev * cos_prev);
    float sin_new = sqrt(1 - cos_new * cos_new);
    return 0.5 * nx * (delta_angle + sin_prev * cos_prev - sin_new * cos_new) + 0.5 * ny * (sin_prev * sin_prev - sin_new * sin_new);
}

float computeHorizonContribution(float3 eyeDir, float3 eyeTangent, float3 viewNorm, float minAngle, float maxAngle)
{
  return
    +0.25 * dot(eyeDir, viewNorm) * (- cos(2.0 * maxAngle) + cos(2.0 * minAngle))
    +0.25 * dot(eyeTangent, viewNorm) * (2.0 * maxAngle - 2.0 * minAngle - sin(2.0 * maxAngle) + sin(2.0 * minAngle));
}

float luminance(float3 color)
{
    return dot(color, float3(0.21267291505, 0.71515223009, 0.07217499918));
}

float3 giPolyFit(float3 albedo, float visibility)
{
    float3 a = 2.0404 * albedo - 0.3324;
    float3 b = 4.7951 * albedo - 0.6417;
    float3 c = 2.7552 * albedo + 0.6903;
    float vis2 = visibility * visibility;
    float vis3 = vis2 * visibility;
    return a * vis3 - b * vis2 + c * visibility;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pixel Shaders
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void PS_PreBlur(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 blur_normal_z : SV_Target0, out float4 blur_color : SV_Target1
)
{
    float3 sum_normal = unpackNormal(tex2D(Skyrim::samp_normal, uv).xy);
    float sum_z = tex2D(Skyrim::samp_depth, uv).x;
    float3 sum_color = mul(g_colorInputMat, tex2D(ReShade::BackBuffer, uv).rgb);
    float sum_w = 1;
    [unroll]
    for(uint i = 0; i < 8; ++i)
    {
        float2 offset_px; sincos(i * 0.25 * PI, offset_px.y, offset_px.x);  // <-sincos here!
        const float2 offset_uv = offset_px / YASSGI_PREBLUR_SCALE * 0.5 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
        const float2 uv_sample = uv + offset_uv;

        const float w = exp(-0.66 * length(offset_px)) * isInScreen(uv_sample);
        sum_normal += unpackNormal(tex2D(Skyrim::samp_normal, uv_sample).xy) * w;
        sum_z += tex2D(Skyrim::samp_depth, uv_sample).x * w;
        sum_color += mul(g_colorInputMat, tex2D(ReShade::BackBuffer, uv_sample).rgb) * w;
        sum_w += w;
    }
    blur_normal_z = float4(sum_normal, sum_z) / sum_w;
    blur_color = float4(sum_color / sum_w, 1);
}

void PS_GI(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 il_ao : SV_Target0
)
{
    float4 temp = 0;  // temp vec for debug

    il_ao = 0;
    // normal_bent = 0;

    const uint2 px_coord = uv * BUFFER_SIZE;

    const float3 color = mul(g_colorInputMat, tex2Dfetch(ReShade::BackBuffer, px_coord).rgb);
    const float3 albedo = fakeAlbedo(color);
    const float3 normal_v = unpackNormal(tex2Dfetch(Skyrim::samp_normal, px_coord).xy);
    const float raw_z = tex2Dfetch(Skyrim::samp_depth, px_coord).x;
    const float3 pos_v = uvzToView(uv, raw_z) * 0.99995;  // closer to the screen bc we're using blurred geometry
    const float3 dir_v_view = -normalize(pos_v);

    [branch]
    if(isWeapon(pos_v.z) || isSky(pos_v.z))  // leave sky alone
        return;
    
    const uint3 rand = pcg3d(uint3(px_coord, iFrameCount));
    const float4 blue = tex2Dfetch(samp_blue, (px_coord + rand.xy) % NOISE_SIZE);
    // // interleaved sampling (enough for ao)
    // uint2 px_coord_shifted = px_coord + blue.xy * 8;  // de-grid
    // const uint px_idx = (dot(px_coord_shifted % INTERLEAVED_SIZE_PX, uint2(1, INTERLEAVED_SIZE_PX)) + (iFrameCount % 16)) % 16;
    // const float2 distrib = hammersley(px_idx, INTERLEAVED_SIZE_PX * INTERLEAVED_SIZE_PX);
    // // ^^^ x for angle, y for stride

    // some consts
    const float rcp_dir_count = 1.0 / iDirCount;
    const float angle_sector = PI * rcp_dir_count;  // may confuse with bitmask il sectors; angle_increment? angle_pizza_slice?
    const float stride_px = max(1, fBaseStridePx * (0.5 + blue.x * fStrideJitter));
    const float log2_stride_px = log2(stride_px);
    const float falloff_start_px = fFxRange * (1 - fFxFalloff);
    const float falloff_mul = -rcp(fFxRange);
    const float falloff_add = falloff_start_px / fFxFalloff + 1;

    // per slice
    float4 sum = 0;  // visibility
    [loop]  // unroll?
    for(uint idx_dir = 0; idx_dir < iDirCount; ++idx_dir)
    {
        // slice directions
        const float angle_slice = (idx_dir + blue.y) * angle_sector;
        float2 dir_px_slice; sincos(angle_slice, dir_px_slice.y, dir_px_slice.x);  // <-sincos here!
        const float2 dir_uv_slice = normalize(dir_px_slice * float2(BUFFER_WIDTH * BUFFER_RCP_HEIGHT, 1));

        const float3 dir_v_slice_screen = float3(dir_px_slice.x, dir_px_slice.y, 0);
        const float3 dir_v_slice_local = projectToPlane(dir_v_slice_screen, dir_v_view);
        const float3 normal_slice = normalize(cross(dir_v_slice_local, dir_v_view));
        const float3 normal_proj = projectToPlane(normal_v, normal_slice);  // not unit vector
        const float3 normal_proj_normalized = normalize(normal_proj);
        
        const float sign_n = sign(dot(dir_v_slice_local, normal_proj));
        const float cos_n = saturate(dot(normal_proj_normalized, dir_v_view));
        const float n = sign_n * acosFast4(cos_n);
        const float sin_n = sin(n);
        
        // Algorithm 1 in the GTAO paper
        // 0 for -dir_px_slice, 1 for +dir_px_slice
        const float2 dists = intersectBox(px_coord, dir_px_slice, float4(0, 0, BUFFER_SIZE)).y;
        float h0, h1;
        [unroll]
        for(uint side = 0; side <= 1; ++side)
        {
            const int side_sign = side * 2 - 1;
            const float max_dist = min(fMaxSampleDistPx, side ? dists.y : abs(dists.x));
            const float min_hor_cos = cos(n + side_sign * HALF_PI);

            // marching
            uint step = 0;
            float dist_px = stride_px;
            float hor_cos = min_hor_cos;
            float3 radiance_sample = 0;
            [loop]
            while(dist_px < max_dist && step < 64)  // preventing infinite loop when you tweak params in ReShade
            {
                const float2 offset_px = dir_px_slice * dist_px;
                const float2 px_coord_sample = px_coord + side_sign * offset_px;
                const float2 uv_sample = (px_coord_sample + 0.5) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

                const uint mip_level = clamp(log2(dist_px) - log2_stride_px, 0, MAX_MIP);

                const float4 geo_sample = tex2Dlod(samp_blur_normal_z, float4(uv_sample, mip_level, 0));
                const float3 pos_v_sample = uvzToView(uv_sample, geo_sample.w);
                const float3 offset_v = pos_v_sample - pos_v;

                [branch]
                if(!isWeapon(pos_v_sample.z))
                {
                    // thin obj heuristics
                    const float falloff = length(offset_v * float3(1, 1, 1 + fThinOccluderCompensation));
                    const float weight = saturate(falloff * falloff_mul + falloff_add);

                    const float3 dir_v_hor = normalize(offset_v);
                    float hor_cos_sample = dot(dir_v_hor, dir_v_view);
                    hor_cos_sample = lerp(min_hor_cos, hor_cos_sample, weight);

#if YASSGI_DISABLE_IL
                    hor_cos = max(hor_cos, hor_cos_sample);
#else
                    [branch]
                    if(hor_cos_sample > hor_cos)
                    {
                        float3 radiance_sample_new = tex2Dlod(samp_blur_color, float4(uv_sample, mip_level, 0)).rgb;
                        radiance_sample_new = luminance(radiance_sample_new) > fLightSrcThres ? radiance_sample_new : 0;
                        // radiance_sample_new *= ilIntegral(cos_n * side_sign, sin_n, hor_cos, hor_cos_sample);
                        radiance_sample_new *= computeHorizonContribution(dir_v_view, dir_v_slice_local, normal_v, acosFast4(hor_cos_sample), acosFast4(hor_cos));

                        // depth filtering. HBIL pp.38
                        float t = smoothstep(0, 1, dot(geo_sample.xyz, offset_v));
                        // float t = dot(geo_sample.xyz, offset_v) > -EPS;
                        // float t = 1;
                        radiance_sample = lerp(radiance_sample, radiance_sample_new, t);

                        sum.rgb += max(0, radiance_sample * albedo);

                        hor_cos = hor_cos_sample;
                    }
#endif
                }

                dist_px += stride_px * exp2(step * fSpreadExp);
                ++step;  // 2 same stride at the start. more precise perhaps (?)
            }

            const float h = acosFast4(hor_cos);
            const float angle_hor = n + clamp(side_sign * h - n, -HALF_PI, HALF_PI);  // XeGTAO suggested skipping clamping. Hmmm...
            sum.w += saturate(length(normal_proj) * 0.25 * (cos_n + 2 * angle_hor * sin_n - cos(2 * angle_hor - n)));
            
            side ? h1 : h0 = h;
        }

        // // bent normal (Algorithm 2)
        // const float t0 = (6 * sin(h0 - n) - sin(3 * h0 - n) +
        //             6 * sin(h1 - n) - sin(3 * h1 - n) + 
        //             16 * sin(n) - 3 * (sin(h0 + n) + sin(h1 + n))) * 0.083333333333333333333;  // rcp 12
        // const float t1 = (-cos(3 * h0 - n) - cos(3 * h1 - n) +
        //             8 * cos(n) - 3 * (cos(h0 + n) +cos(h1 + n))) * 0.083333333333333333333;
        // const float3 normal_bent_local = float3(dir_px_slice * t0, -t1);
        // normal_bent.xyz += normal_bent_local;
    }

    // temp = normal_bent.xyz * 0.5 + 0.5;

    // normal_bent.xyz = normalize(normal_bent.xyz);
    // normal_bent.w = 1;

    sum /= iDirCount;
    il_ao.w = clamp(1 - sum.w, 0, 0.95);  // got -inf here...
    il_ao.rgb = sum.rgb;
    // il_ao = temp;
}

// void PS_Accum(
//     in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
//     out float4 il_ao_accum : SV_Target)
// {
//     const uint2 px_coord = uv * BUFFER_SIZE;

//     float4 il_ao_curr = tex2Dfetch(samp_il_ao, px_coord);
//     float4 il_ao_hist = tex2Dfetch(samp_il_ao_ac2, px_coord);
//     il_ao_accum = lerp(il_ao_hist, il_ao_curr, rcp(1 + fDebug));
// }

// void PS_AccumCopy(
//     in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
//     out float4 il_ao_hist : SV_Target)
// {
//     il_ao_hist = tex2Dfetch(samp_il_ao_ac1, uv * BUFFER_SIZE);
// }

void PS_Display(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target)
{
    color = tex2D(ReShade::BackBuffer, uv);
    float4 il_ao = tex2Dlod(samp_il_ao, float4(uv, 1, 0));  // no need for any filter, 3 slices and it's good enough w/ vanilla TAA
    float ao_mult = fAoStrength > 0 ?
        lerp(1, 1 - il_ao.a, fAoStrength) :  // normal mixing
        exp2(il_ao.a * fAoStrength);  // exponential mixing

    if(iViewMode == 0)  // None
    {
        color.rgb = mul(g_colorInputMat, color.rgb);
        color.rgb += il_ao.rgb * fIlStrength;
        color.rgb = color.rgb * ao_mult;
        color.rgb = mul(g_colorOutputMat, color.rgb);
    }
    else if(iViewMode == 1)  // AO
    {
        color.rgb = ao_mult;
    }
    else if(iViewMode == 2)  // IL
    {
        color.rgb = mul(g_colorOutputMat, il_ao.rgb * fIlStrength);
    }
}



technique YASSGI_Skyrim
{
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_PreBlur;
        RenderTarget0 = tex_blur_normal_z;
        RenderTarget1 = tex_blur_color;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_GI;
        RenderTarget0 = tex_il_ao;
    }
    // pass {
    //     VertexShader = PostProcessVS;
    //     PixelShader = PS_Accum;
    //     RenderTarget0 = tex_il_ao_ac1;
    // }
    // pass {
    //     VertexShader = PostProcessVS;
    //     PixelShader = PS_AccumCopy;
    //     RenderTarget0 = tex_il_ao_ac2;
    // }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Display;
    }
}

}
