/*
    raw_z: raw z
    depth: linearized z (0-1)
    z: linearized z
    z direction: + going farther, - coming closer
    dir: normalized direction, same as above
    normal: pointing outwards
    color & gi: ACEScg
    pos_v: view
*/

/*  TODO
    - il
    - bent normal
    - thickness heuristic
    - alternative bitmask impl (?)
*/

#include "ReShade.fxh"
#include "ShaderFastMathLib.h"

namespace YASSGI_SKYRIM_TEST
{

#define PI      3.14159265358979323846264
#define HALF_PI 1.5707963267948966
#define RCP_PI  0.3183098861837067

#define MAX_UINT_F 4294967295.0

#define EPS 1e-6

#define BUFFER_SIZE uint2(BUFFER_WIDTH, BUFFER_HEIGHT)

#define INTERLEAVED_SIZE_PX 4
#define MAX_MIP 8

// color space conversion matrices
// src: https://www.colour-science.org/apps/  using CAT02
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


uniform float fFarPlane  < source = "Far"; >;
uniform float fNearPlane < source = "Near"; >;

uniform float4x4 fViewMatrix        < source = "ViewMatrix"; >;
uniform float4x4 fProjMatrix        < source = "ProjMatrix"; >;
uniform float4x4 fViewProjMatrix    < source = "ViewProjMatrix"; >;
uniform float4x4 fInvViewProjMatrix < source = "InvViewProjMatrix"; >;

uniform float fFov < source = "FieldOfView"; >;
uniform float3 fCamPos < source = "Position"; >;

uniform int   iFrameCount < source = "FrameCount"; >;
uniform float fTimer      < source = "TimerReal"; >;
uniform float fFrameTime  < source = "TimingsReal"; >;


uniform float fDebug = 1;

uniform int iViewMode <
	ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "YASSGI\0AO\0";
> = 0;

uniform float2 fZRange <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "Weapon/Sky Z Range";
    ui_min = 0.0; ui_max = 25000;
    ui_step = 0.1;
> = float2(0.1, 20000);

uniform uint iDirCount <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Slices";
    ui_min = 1; ui_max = 4;
> = 1;  

uniform float fBaseStridePx <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Base Stride (px)";
    ui_min = 1; ui_max = 64;
    ui_step = 1;
> = 4;

uniform float fSpreadExp <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Spread Exponent";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.5;

// Separate AO and IL perhaps
uniform float fMaxDistPx <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Max Distance (px)";
    ui_min = 2; ui_max = BUFFER_WIDTH;
    ui_step = 1;
> = 200;

uniform float fSampleRangePx <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "LOD Range (px)";
    ui_min = 2; ui_max = 64;
    ui_step = 1;
> = 8;

uniform float fIlStrength <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "IL";
    ui_min = 0.0; ui_max = 2.0;
    ui_step = 0.01;
> = 1.0;

uniform float fAoStrength <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "AO";
    ui_min = 0.0; ui_max = 4.0;
    ui_step = 0.01;
> = 1.0;

}

namespace Skyrim
{
texture tex_normal : NORMAL_TAAMASK_SSRMASK;
sampler samp_normal { Texture = tex_normal; };

texture tex_depth : TARGET_MAIN_DEPTH;
sampler samp_depth { Texture = tex_depth; };

texture tex_motion_vector : MOTION_VECTOR;
sampler samp_motion_vector { Texture = tex_motion_vector; };

texture tex_cubemap : REFLECTIONS;
sampler samp_cubemap { Texture = tex_cubemap; };
}

namespace YASSGI_SKYRIM_TEST
{
// downscaled normal (RGB) raw_z (A)
// blurred normal only used for rough backface verification as in the hbil paper.
// tbf orig paper uses interleaved buffers (not blurred) but no sampler spamming for dx9.
//                                          (or perhaps a separate shader(s) for those?)
texture tex_blur_normal_z  {Width = BUFFER_WIDTH / INTERLEAVED_SIZE_PX; Height = BUFFER_HEIGHT / INTERLEAVED_SIZE_PX; Format = RGBA32F; MipLevels = MAX_MIP;};
sampler samp_blur_normal_z {Texture = tex_blur_normal_z;};

texture tex_blur_color  {Width = BUFFER_WIDTH / INTERLEAVED_SIZE_PX; Height = BUFFER_HEIGHT / INTERLEAVED_SIZE_PX; Format = RGBA16F; MipLevels = MAX_MIP;};
sampler samp_blur_color {Texture = tex_blur_color;};

texture tex_gi_ao  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao {Texture = tex_gi_ao;};

texture tex_gi_ao_blur1  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao_blur1 {Texture = tex_gi_ao_blur1;};

texture tex_gi_ao_blur2  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao_blur2 {Texture = tex_gi_ao_blur2;};

// <---- Utility ---->

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
    return raw_z / fFarPlane - raw_z * (fFarPlane - fNearPlane);
}

float3 uvzToWorld(float2 uv, float raw_z)
{
    float4 pos_s = float4((uv * 2 - 1) * float2(1, -1) * raw_z, raw_z, 1);
    float4 pos = mul(transpose(fInvViewProjMatrix), pos_s);
    return pos.xyz / pos.w;
}
float3 uvzToView(float2 uv, float raw_z)
{
    float4x4 inv_proj = mul(fInvViewProjMatrix, fViewMatrix);
    float4 pos_view = mul(transpose(inv_proj), float4((uv * 2 - 1) * float2(1, -1) * raw_z, raw_z, 1));
    return pos_view.xyz / pos_view.w;
}
float3 viewToUvz(float3 pos){
    float4 pos_clip = mul(transpose(fProjMatrix), float4(pos, 1));
    pos_clip.xyz /= pos_clip.w;
    pos_clip.xy = (pos_clip.xy / pos_clip.z * float2(1, -1) + 1) * 0.5;
    return pos_clip.xyz;
}

bool isNear(float z){return z < fNearPlane;}
bool isFar(float z){return z > fFarPlane;}
bool isWeapon(float z){return z < fZRange.x;}
bool isSky(float z){return z > fZRange.y;}

// <---- Sampling ---->

// algo: https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler
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

/*
    Source:
    Hash Functions for GPU Rendering 
        http://www.jcgt.org/published/0009/03/02/
        https://www.shadertoy.com/view/XlGcRh
*/
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


/*
    Reference:
    LegitEngine
        https://github.com/Raikiri/LegitEngine
    
    No source code included
*/
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
        const float2 offset_uv = offset_px * INTERLEAVED_SIZE_PX * 0.5 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
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


/*  
    Reference:
    Horizon-Based Indirect Lighting
        https://github.com/Patapom/GodComplex/blob/master/Tests/TestHBIL/2018%20Mayaux%20-%20Horizon-Based%20Indirect%20Lighting%20(HBIL).pdf
    LegitEngine
        https://github.com/Raikiri/LegitEngine
    XeGTAO
        https://github.com/GameTechDev/XeGTAO/tree/master
*/
void PS_GI(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 gi_ao : SV_Target0
)
{
    // float4 temp = 0;  // temp vec for debug

    gi_ao = 0;

    // const float3 rand = pcg3d(uint3(uv * BUFFER_SIZE, iFrameCount)) / MAX_UINT_F;
    float3 rand = 0;

    const uint2 px_coord = uv * BUFFER_SIZE;
    
    float3 normal_v = unpackNormal(tex2Dfetch(Skyrim::samp_normal, px_coord).xy);
    const float raw_z = tex2Dfetch(Skyrim::samp_depth, px_coord).x;
    const float3 pos_v = uvzToView(uv, raw_z) * 0.9995;  // closer to the screen bc we're using blurred geometry
    const float3 dir_v_view = -normalize(pos_v);

    [branch]
    if(isWeapon(pos_v.z) || isSky(pos_v.z))  // leave sky alone
        return;

    // interleaved sampling
    const float2 distrib = hammersley(
        dot(px_coord % INTERLEAVED_SIZE_PX, uint2(1, INTERLEAVED_SIZE_PX)),
        INTERLEAVED_SIZE_PX * INTERLEAVED_SIZE_PX);
    // ^^^ x for angle, y for stride

    // some consts
    const float rcp_dir_count = 1.0 / iDirCount;
    const float angle_sector = 2.0 * PI * rcp_dir_count;  // may confuse with bitmask il sectors; angle_increment? angle_pizza_slice?
    const float stride_px = max(1, fBaseStridePx + fBaseStridePx * (distrib.y + rand.x - 1) * 0.5);
    const float log2_stride_px = log2(stride_px);

    // per slice
    float4 sum = 0;
    [loop]  // unroll?
    for(uint idx_dir = 0; idx_dir < iDirCount; ++idx_dir)
    {
        // slice directions
        const float angle_slice = (idx_dir + distrib.x + rand.y) * angle_sector;
        float2 dir_px_slice; sincos(angle_slice, dir_px_slice.y, dir_px_slice.x);  // <-sincos here!
        const float2 dir_uv_slice = normalize(dir_px_slice * float2(BUFFER_WIDTH * BUFFER_RCP_HEIGHT, 1));

        const float3 dir_v_slice_screen = float3(dir_px_slice.x, dir_px_slice.y, 0);
        const float3 dir_v_slice_local = projectToPlane(dir_v_slice_screen, dir_v_view);
        const float3 normal_slice = normalize(cross(dir_v_slice_local, dir_v_view));
        const float3 normal_proj = projectToPlane(normal_v, normal_slice);  // not unit vector

        const float sign_n = sign(dot(dir_v_slice_local, normal_proj));
        const float cos_n = saturate(dot(normalize(normal_proj), dir_v_view));
        const float n = sign_n * acosFast4(cos_n);
        const float sin_n = sin(n);
        
        // 0 for -dir_px_slice, 1 for +dir_px_slice
        const float2 dists = intersectBox(px_coord, dir_px_slice, float4(0, 0, BUFFER_SIZE)).y;
        [unroll]
        for(uint side = 0; side <= 1; ++side)
        {
            const int side_sign = side * 2 - 1;
            const float max_dist = min(fMaxDistPx, side ? dists.y : abs(dists.x));
            const float min_hor_cos = cos(n + side_sign * HALF_PI);

            // marching
            uint step = 0;
            float dist_px = stride_px;
            float hor_cos = min_hor_cos;
            [loop]
            while(dist_px < max_dist && step < 64)
            {
                const float2 offset_px = dir_px_slice * dist_px;
                const float2 px_coord_sample = px_coord + side_sign * offset_px;
                const float2 uv_sample = (px_coord_sample + 0.5) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

                const uint mip_level = clamp(log2(dist_px) - log2_stride_px, 0, MAX_MIP);

                const float raw_z_sample = tex2Dlod(samp_blur_normal_z, float4(uv_sample, mip_level, 0)).w;
                const float3 pos_v_sample = uvzToView(uv_sample, raw_z_sample);
                const float3 dir_v_hor = normalize(pos_v_sample - pos_v);
                hor_cos = max(hor_cos, dot(dir_v_hor, dir_v_view));

                float3 color_sample = tex2Dlod(samp_blur_color, float4(uv_sample, mip_level, 0)).rgb;
                
                dist_px += stride_px * exp2((step + distrib.y) * fSpreadExp);
                ++step;  // 2 same stride at the start. more precise perhaps (?)
            }

            const float angle_hor = n + clamp(side_sign * acosFast4(hor_cos) -n, -HALF_PI, HALF_PI);
            sum.w += length(normal_proj) * 0.25 * (cos_n + 2 * angle_hor * sin_n - cos(2 * angle_hor - n));
        }
    }

    // temp = rawZToLinear01(raw_z) * fFarPlane - pos_v.z;

    gi_ao.w = saturate(1 - sum.w);
    // gi_ao = temp;
}


/*
    Reference:
    Edge-avoiding À-trous
        https://jo.dreggn.org/home/2010_atrous.pdf
    SVGF (impl without variance)
        https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf
        https://www.shadertoy.com/view/tlXfRX
*/
float4 atrous(sampler samp, float2 uv, float radius)
{
    const float rand = pcg4d(uint4(uv * BUFFER_SIZE, iFrameCount, radius)).x / MAX_UINT_F;

    const uint2 px_coord = uv * BUFFER_SIZE;

    const float3 normal = unpackNormal(tex2Dfetch(Skyrim::samp_normal, px_coord).xy);
    const float z = rawZToLinear01(tex2Dfetch(Skyrim::samp_depth, px_coord).x) * fFarPlane;
    const float2 zgrad = float2(ddx(z), ddy(z));
    const float4 color = tex2Dfetch(samp, px_coord);

    float weightsum = 1;
    float4 sum = color;
    [unroll] for (int i = 0; i < 8; ++i)
    {
        float2 offset_px; sincos((i + rand) * 0.25 * PI, offset_px.y, offset_px.x); // <-sincos here!
        float2 uv_sample = uv + offset_px * radius * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

        const float3 normal_samp = unpackNormal(tex2D(Skyrim::samp_normal, uv_sample).xy);
        const float z_samp = rawZToLinear01(tex2D(Skyrim::samp_depth, uv_sample).x) * fFarPlane;
        const float4 color_sample = tex2D(samp, uv_sample);

        float w = pow(max(0, dot(normal, normal_samp)), 64);                          // normal
        w *= exp(-abs(z - z_samp) / (1 * abs(dot(zgrad, offset_px * radius)) + EPS)); // depth
        // w *= exp(-abs(lum_curr - lum[i]) / (fVarianceWeight * variance + EPS));    // luminance (should except ao)
        w = saturate(w) * exp(-0.66 * length(offset_px)) * isInScreen(uv_sample);     // gaussian kernel
        // ^^^ -four- three horsemen of the SVGF sampling weights

        weightsum += w;
        sum += color_sample * w;
    }
    return (sum / weightsum);
}

void PS_Blur1(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0
)
{
    color = atrous(samp_gi_ao, uv, 2);
}

void PS_Blur2(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0
)
{
    color = atrous(samp_gi_ao_blur1, uv, 4);
}

void PS_Blur3(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0
)
{
    color = atrous(samp_gi_ao_blur2, uv, 8);
}

void PS_Display(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target)
{
    color = tex2D(ReShade::BackBuffer, uv);
    float4 gi_ao = tex2D(samp_gi_ao_blur2, uv);
    
    if(iViewMode == 0)  // None
    {
        color.rgb = mul(g_colorInputMat, color.rgb);
        // color.rgb += gi_ao.rgb * fIlStrength;
        color.rgb = color.rgb * exp2(-gi_ao.a * fAoStrength);
        color.rgb = saturate(mul(g_colorOutputMat, color.rgb));
    }
    else if(iViewMode == 1)  // AO
    {
        color.rgb = exp2(-gi_ao.a * fAoStrength);
    }
}

technique YASSGI_TEST
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
        RenderTarget0 = tex_gi_ao;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Blur1;
        RenderTarget0 = tex_gi_ao_blur1;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Blur2;
        RenderTarget0 = tex_gi_ao_blur2;
    }
    // pass {
    //     VertexShader = PostProcessVS;
    //     PixelShader = PS_Blur3;
    //     RenderTarget0 = tex_gi_ao_blur1;
    // }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Display;
    }
}

}