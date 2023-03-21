/*
    Unspecified Attribution:
    Horizon-Based Indirect Lighting (mostly following this pdf)
        https://github.com/Patapom/GodComplex/blob/master/Tests/TestHBIL/2018%20Mayaux%20-%20Horizon-Based%20Indirect%20Lighting%20(HBIL).pdf
    LegitEngine by Alexander Sannikov | MIT
        https://github.com/Raikiri/LegitEngine
*/

/*
    raw_z: raw z
    depth: linearized z (0-1)
    z: linearized z
    z direction: + going farther, - coming closer
    dir: normalized direction, same as above
    normal: pointing outwards
    color & gi: ACEScg
    pos_v: view
    pos_l: local camera plane
    pos_s: 2d slice
*/

#include "ReShade.fxh"

namespace YASSGI_SKYRIM_TEST
{

#define PI      3.14159265358979323846264
#define HALF_PI 1.5707963267948966
#define RCP_PI  0.3183098861837067

#define EPS 1e-6

#define BUFFER_SIZE uint2(BUFFER_WIDTH, BUFFER_HEIGHT)


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


uniform uint iDirCount = 4;  // half side direction, same as Sannikov, bc we march different steps on each size
uniform uint iInterleavedSizePx = 4;
uniform float fSampleRangePx = 8;
uniform float fBaseStridePx = 1;
uniform float fSpreadExp = 0.5;

uniform float fDebug = 1;

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
texture tex_blur_normal_z  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA32F; MipLevels = 10;};
sampler samp_blur_normal_z {Texture = tex_blur_normal_z;};

texture tex_blur_color  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = 10;};
sampler samp_blur_color {Texture = tex_blur_color;};

texture tex_gi_ao  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao {Texture = tex_gi_ao;};

// <---- Utility ---->

bool isInScreen(float2 uv) { return all(uv >= 0) && all(uv < 1); }

float maxOf(float4 v) {return max(max(v.x, v.y), max(v.z, v.w));}
float minOf(float4 v) {return min(min(v.x, v.y), min(v.z, v.w));}

// box = minx, miny, maxx, maxy
// retval = nearest / farthest traversal distance (can be negative)
// use retval.x < retval.y to check if truly intersected
float2 intersectBox(float2 orig, float2 dir, float4 box)
{
    float4 dists = (box - orig.xyxy) * rcp(dir).xyxy;
    return float2(minOf(dists), maxOf(dists));
}

// <---- Input ---->

float3 unpackNormal(float2 enc)
{
	float2 fenc = enc * 4 - 2;
    float f = dot(fenc, fenc);
    float g = sqrt(1 - f * 0.25);
    float3 n = float3(fenc * g, 1 - f * 0.5);
    return n;
}

float3 uvToWorld(float2 uv, float raw_z)
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

// <---- Sampling ---->

// src: https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/The_Halton_Sampler
float radicalInverse(uint i)
{
    uint bits = (i << 16u) | (i >> 16u);
    bits = (bits & 0x55555555u) << 1u | (bits & 0xAAAAAAAAu) >> 1u;
    bits = (bits & 0x33333333u) << 2u | (bits & 0xCCCCCCCCu) >> 2u;
    bits = (bits & 0x0F0F0F0Fu) << 4u | (bits & 0xF0F0F0F0u) >> 4u;
    bits = (bits & 0x00FF00FFu) << 8u | (bits & 0xFF00FF00u) >> 8u;
    return bits / float(0xFFFFFFFFu);
}
float2 hammersley(uint i, uint N) {return float2(float(i) / N, radicalInverse(i));}



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
        const float2 offset_uv = offset_px * iInterleavedSizePx * 0.5 * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
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
    out float4 gi_ao : SV_Target0
)
{
    float4 temp = 0;  // temp vec for debug

    const uint2 px_coord = uv * BUFFER_SIZE;
    
    const float3 normal_v = unpackNormal(tex2Dfetch(Skyrim::samp_normal, px_coord).xy);
    const float raw_z = tex2Dfetch(Skyrim::samp_depth, px_coord).x;
    const float3 pos_v = uvzToView(uv, raw_z);
    const float3 dir_v_view = -normalize(pos_v);

    // local camera plane
    const float3 dir_v_local_x = normalize(cross(float3(0, -1, 0), dir_v_view));
    const float3 dir_v_local_y = cross(dir_v_view, dir_v_local_x);

    // interleaved sampling
    const float2 distrib = hammersley(
        dot(px_coord % iInterleavedSizePx, uint2(1, iInterleavedSizePx)),
        iInterleavedSizePx * iInterleavedSizePx);
    // ʌʌʌ x for angle, y for stride

    // per slice
    const float rcp_dir_count = 1.0 / iDirCount;
    const float angle_sector = 2.0 * PI * rcp_dir_count;  // may confuse with bitmask il sectors; angle_increment? angle_pizza_slice_with_tomato_mozzarella_and_basil?
    float4 sum = 0;
    [loop]  // unroll?
    for(uint idx_dir = 0; idx_dir < iDirCount; ++idx_dir)
    {
        // slice directions
        const float angle_slice = (idx_dir + distrib.x) * angle_sector;
        float2 dir_l_slice; sincos(angle_slice, dir_l_slice.y, dir_l_slice.x);  // <-sincos here!
        const float3 dir_v_slice = dir_l_slice.x * dir_v_local_x + dir_l_slice.y * dir_v_local_y;
        const float sin_alpha = dir_v_slice.z;
        const float cos_alpha = sqrt(1 - sin_alpha * sin_alpha) * 0.99;  // 0.99 from hbil code, prevent NaN

        // proj normal
        const float2 normal_s_proj = normalize(float2(dot(normal_v, dir_v_slice), dot(normal_v, dir_v_view)));
        float t = -normal_s_proj.x / normal_s_proj.y;

        // march distance
        const float2 dir_uv_slice = viewToUvz(pos_v + dir_v_slice * EPS).xy - viewToUvz(pos_v).xy;  // sign f*ck-up prevention
        const float2 dir_px_slice = normalize(dir_uv_slice * BUFFER_SIZE);
        const float dist_to_screen = intersectBox(px_coord, dir_px_slice, float4(0, 0, BUFFER_SIZE)).y;

        // marching
        uint step = 0;
        float dist_px = fBaseStridePx;
        float max_cos_theta = t * rsqrt(1 + t * t);
        [loop]
        while(dist_px < dist_to_screen)
        {
            float2 offset_px = dir_px_slice * dist_px;
            float2 px_coord_sample = px_coord + offset_px;
            float2 uv_sample = (px_coord_sample + 0.5) * float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
            // ʌʌʌ does this account for camera distortion? (SLAM PTSD)

            uint mip_level = log2(dist_px / fBaseStridePx);
            [branch]  
            if(mip_level >= 10) break;

            float raw_z_sample = tex2Dlod(samp_blur_normal_z, float4(uv_sample, mip_level, 0)).w;
            float3 pos_v_sample = uvzToView(uv_sample, raw_z_sample);
            float3 pos_v_offset = pos_v_sample - pos_v;

            float cos_theta = 
                dot(float2(sin_alpha, cos_alpha), float2(abs(pos_v_offset.x), -pos_v_offset.z)) *
                rsqrt(pos_v_offset.x * pos_v_offset.x - pos_v_offset.z * pos_v_offset.z);
            max_cos_theta = max(cos_theta, max_cos_theta);

            // float3 color_sample = tex2Dlod(samp_blur_color, float4(uv_sample, mip_level, 0)).rgb;
            
            dist_px += fBaseStridePx * exp2(step * fSpreadExp);
            ++step;  // 2 same stride at the start. more precise perhaps (?)
        }

        sum.w += max_cos_theta * rcp_dir_count;

        temp.xy = normal_s_proj * 0.5 + 0.5;
    }

    gi_ao = 1 - sum.w;
    // gi_ao = temp;
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
        // RenderTarget0 = tex_gi_ao;
    }
}

}