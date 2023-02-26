#include "ReShade.fxh"

#include "Poisson.fxh"

#ifndef YASSGI_RENDER_SCALE
#   define YASSGI_RENDER_SCALE 0.5
#endif

#ifndef YASSGI_MIP_LEVEL
#   define YASSGI_MIP_LEVEL 4
#endif

#define YASSGI_BUFFER_WIDTH BUFFER_WIDTH * YASSGI_RENDER_SCALE
#define YASSGI_BUFFER_HEIGHT BUFFER_HEIGHT * YASSGI_RENDER_SCALE

#define NOISE_SIZE 512

#define PI 3.14159265359
#define SQRT2 1.41421356237

#define EPS 1e-6

namespace YASSGI{
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Buffers
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
texture ZTex     { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; MipLevels = YASSGI_MIP_LEVEL; };
sampler ZSampler { Texture = ZTex; };

texture ZPrevTex     { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; MipLevels = YASSGI_MIP_LEVEL; };
sampler ZPrevSampler { Texture = ZPrevTex; };

// Alpha channel = hit number ~= AO
texture GITex     { Width = YASSGI_BUFFER_WIDTH; Height = YASSGI_BUFFER_HEIGHT; Format = RGBA16F; };
sampler GISampler { Texture = GITex; };

texture GIHitDistTex     { Width = YASSGI_BUFFER_WIDTH; Height = YASSGI_BUFFER_HEIGHT; Format = R16F; };
sampler GIHitDistSampler { Texture = GIHitDistTex; };

texture GIPreBlurTex     { Width = YASSGI_BUFFER_WIDTH; Height = YASSGI_BUFFER_HEIGHT; Format = RGBA16F; };
sampler GIPreBlurSampler { Texture = GIPreBlurTex; };

texture GIAccumTex     { Width = YASSGI_BUFFER_WIDTH; Height = YASSGI_BUFFER_HEIGHT; Format = RGBA16F; };
sampler GIAccumSampler { Texture = GIAccumTex; };

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Uniform Varibales
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// @source https://www.shadertoy.com/view/WltSRB
static const float3x3 sRGBtoXYZ = float3x3
(
	0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
);
static const float3x3 XYZtoSRGB = float3x3
(
    3.2404542, -1.5371385, -0.4985314,
    -0.9692660, 1.8760108, 0.0415560,
    0.0556434, -0.2040259, 1.0572252
);

uniform int   RANDOM < source = "random"; min = 0; max = 1 << 15; >;
uniform uint  FRAMECOUNT  < source = "framecount"; >;
uniform float FRAMETIME   < source = "frametime";  >;

uniform int iDebugView <
	ui_type = "combo";
    ui_label = "Debug View";
    ui_items = "None\0Initial Sample\0Hit Intensity\0Pre-Blur\0Accumulated GI\0";
> = 0;

// <---- Depth and Normal ---->

uniform int iVerticalFOV <
    ui_type = "slider";
    ui_category = "Depth & Normal";
    ui_label = "Vertical FOV";
    ui_min = 60; ui_max = 140;
    ui_step = 1;
> = 90;

uniform float2 fDepthRange <
    ui_type = "slider";
    ui_category = "Depth & Normal";
    ui_label = "Depth Range";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
> = float2(0.0, 0.999);

// <---- Tracing ---->

uniform int iRayAmount <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Ray Amount";
    ui_min = 1; ui_max = 32;
    ui_step = 1;
> = 3;

uniform int iMaxRayStep <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Max Ray Step";
    ui_min = 1; ui_max = 40;
    ui_step = 1;
> = 16;

uniform float fBaseStride <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Base Stride";
    ui_min = 0.0; ui_max = 4.0;
    ui_step = 0.01;
> = 0.3;

uniform float fStrideJitter <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Stride Jitter";
    ui_min = 0.0; ui_max = 0.9;
    ui_step = 0.01;
> = 0.2;

uniform float fSpreadStep <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Step/Spread Ratio";
    ui_min = 1.0; ui_max = 40.0;
    ui_step = 0.1;
> = 2.0;

uniform float fZThickness <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Z Thickness";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
> = 0.4;

uniform bool bBackfaceLighting <
    ui_label = "Backface Lighting";
    ui_category = "Tracing";
> = false;

// <---- Denoising ---->

uniform float fPreBlurRadius <
    ui_type = "slider";
    ui_category = "Denoising";
    ui_label = "Pre-Blur Radius";
    ui_min = 0.0; ui_max = 50.0;
    ui_step = 1.0;
> = 30.0;

static const float fPreBlurNonlinearAccumSpeed = ( 1.0 / ( 1.0 + 8.0 ) );
static const float fLobeAngleFraction = 0.13f;

uniform float4 fHitDistParams <
    ui_type = "input";
    ui_category = "Denoising";
    ui_label = "Hitdist Scaling Params";
    ui_tooltip = "Most don't need changes except perhaps z scale.\n"
                 "1) bias 2) z scale (ratio to 1 meter) 3) roughness scale 4) roughness power";
    ui_min = 0.1; ui_max = 10.0;
    ui_step = 0.01;
> = float4(3.0, 0.1, 20.0, -25.0);

uniform float fGeometrySensitivity <
    ui_type = "slider";
    ui_category = "Denoising";
    ui_label = "Geometry Sensitivity";
    ui_tooltip = "maximum allowed deviation from local tangent plane.";
    ui_min = 0.001; ui_max = 0.1;
    ui_step = 0.001;
> = 0.005f;

uniform float fAccumMult <
    ui_type = "slider";
    ui_category = "Denoising";
    ui_label = "Accumulation Multiplier";
    ui_min = 0.1; ui_max = 10.0;
    ui_step = 0.1;
> = 4.0;

// <---- Color & Mixing ---->

uniform int iInputColorSpace <
	ui_type = "combo";
    ui_label = "Input Color Space";
    ui_items = "sRGB\0";
	ui_category = "Color & Mixing";
> = 0;

uniform float3 fAmbientLight <
    ui_type = "color";
    ui_category = "Color & Mixing";
    ui_label = "Ambient Light";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.1;
> = 0.0;

uniform float fMixMult <
    ui_type = "slider";
    ui_category = "Color & Mixing";
    ui_label = "Mix Strength";
    ui_min = 0.1; ui_max = 10.0;
    ui_step = 0.1;
> = 4.0;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Vertex Shader
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// MXAO.fx
struct VSOUT
{
    float4 position    : SV_Position;
    float2 texcoord    : TEXCOORD0;
    float2 scaledcoord : TEXCOORD1;
    float3 uvtoviewADD : TEXCOORD4;
    float3 uvtoviewMUL : TEXCOORD5;
};

VSOUT VS_YASSGI(in uint id : SV_VertexID)
{
        VSOUT vsout;

        vsout.texcoord.x = (id == 2) ? 2.0 : 0.0;
        vsout.texcoord.y = (id == 1) ? 2.0 : 0.0;
        vsout.scaledcoord.xy = vsout.texcoord.xy / YASSGI_RENDER_SCALE;
        vsout.position = float4(vsout.texcoord.xy * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);

        // MXAO.uvtoviewADD = float3(-1.0,-1.0,1.0);
        // MXAO.uvtoviewMUL = float3(2.0,2.0,0.0);
        //uncomment to enable perspective-correct position recontruction. Minor difference for common FoV's
        vsout.uvtoviewADD = float3(-tan(radians(iVerticalFOV * 0.5)).xx, 1.0);
        vsout.uvtoviewADD.y *= BUFFER_ASPECT_RATIO;
        vsout.uvtoviewMUL = float3(-2.0 * vsout.uvtoviewADD.xy, 0.0);

        return vsout;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// <---- Math ---->
float2x2 getRotateMatrix(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return float2x2(c, -s, s, c);
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

// <---- Depth and Normal ---->
float z2Depth(in float z){ return (z - 1.0) * rcp(RESHADE_DEPTH_LINEARIZATION_FAR_PLANE); }
float depth2Z(in float depth){ return depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE + 1.0; }

float3 coords2WorldPos(in float2 coord, in VSOUT vsout)
{
    return (coord.xyx * vsout.uvtoviewMUL + vsout.uvtoviewADD) * depth2Z(ReShade::GetLinearizedDepth(coord.xy));
}
float2 worldPos2Coords(in float3 world_pos, in VSOUT vsout)
{
    const float4 projtouv = float4(rcp(vsout.uvtoviewMUL.xy), -rcp(vsout.uvtoviewMUL.xy) * vsout.uvtoviewADD.xy);
    return (world_pos.xy / world_pos.z) * projtouv.xy + projtouv.zw;
}

float3 getScreenNormal(in float2 uv, in VSOUT vsout)
{
    float3 center_position = coords2WorldPos(uv, vsout);

    float3 delta_x, delta_y;
    float4 neighbour_uv;
    
    neighbour_uv = uv.xyxy + float4(BUFFER_PIXEL_SIZE.x, 0, -BUFFER_PIXEL_SIZE.x, 0);

    float3 delta_right = coords2WorldPos(neighbour_uv.xy, vsout) - center_position;
    float3 delta_left  = center_position - coords2WorldPos(neighbour_uv.zw, vsout);

    delta_x = abs(delta_right.z) > abs(delta_left.z) ? delta_left : delta_right;

    neighbour_uv = uv.xyxy + float4(0, BUFFER_PIXEL_SIZE.y, 0, -BUFFER_PIXEL_SIZE.y);
        
    float3 delta_bottom = coords2WorldPos(neighbour_uv.xy, vsout) - center_position;
    float3 delta_top  = center_position - coords2WorldPos(neighbour_uv.zw, vsout);

    delta_y = abs(delta_bottom.z) > abs(delta_top.z) ? delta_top : delta_bottom;

    float3 normal = cross(delta_y, delta_x);
    normal *= rsqrt(dot(normal, normal)); //no epsilon, will cause issues for some reason

    return normal;
}   

bool isInScreen(in float2 coord)
{
    return coord.x > 0.0 && coord.x < 1.0 && coord.y > 0.0 && coord.y < 1.0;
}
bool isNear(in float world_z)
{
    return world_z < depth2Z(fDepthRange.x);
}
bool isSky(in float world_z)
{
    return world_z > depth2Z(fDepthRange.y);
}

// <---- Random and Sampling ---->

/// @source https://zhuanlan.zhihu.com/p/390862782
float rand4dTo1d(float4 value, float a, float4 b)
{
    float4 small_val = sin(value);
    float random = dot(small_val, b);
    random = frac(sin(random) * a);
    return random;
}
float3 rand4dTo3d(float4 value){
    return float3(
        rand4dTo1d(value, 14375.5964, float4(15.637,76.243,37.168,83.511)),
        rand4dTo1d(value, 14684.6034, float4(45.366, 23.168,65.918,57.514)),
        rand4dTo1d(value, 14985.1739, float4(62.654, 88.467,25.111,61.875))
    );
}

/// @source http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
float3 uniformLambert(float2 rand, float3 normal)
{
    float cos_theta = rand.x;
    float sin_theta = sqrt(1 - cos_theta * cos_theta);
    float phi = 2 * PI * rand.y;

    float3 h = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    float3 up_vec = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tan_x = normalize(cross(up_vec, normal));
    float3 tan_y = cross(normal, tan_x);
    return normalize(tan_x * h.x + tan_y * h.y + normal * h.z);
}

// <---- Tracing ---->

struct RayInfo
{
    float3 orig;
    float3 dir;
    float spread_fact;

    float3 pos;
    float2 uv;
    bool hit;
    float travel_dist;
    float mip_level;
};

// Simple tracer
// you can control stride by providing a non-unit ray_dir
void traceRay(inout RayInfo ray, VSOUT vsout)
{
    ray.hit = false;
    ray.mip_level = YASSGI_MIP_LEVEL;
    ray.travel_dist = 0;

    float3 stride = ray.dir * fBaseStride;
    float len_stride = length(stride);
    ray.pos = ray.orig + stride * 0.008;
    ray.uv = worldPos2Coords(ray.pos, vsout);

    if(isNear(ray.orig.z) || isSky(ray.orig.z))
        return;

    [loop]
    for(int step = 0; step < iMaxRayStep; step++)
    {
        ray.mip_level = min(float(step) / ray.spread_fact, YASSGI_MIP_LEVEL);
        float len_mult = exp2(ray.mip_level);
        float3 new_pos = ray.pos + stride * len_mult;
        float2 new_coord = worldPos2Coords(new_pos, vsout);
        ray.travel_dist += len_stride * len_mult;

        if(!isInScreen(new_coord) || isNear(new_pos.z) || isSky(new_pos.z))
            break;

        float new_z = tex2Dlod(ZSampler, float4(new_coord, int(ray.mip_level), 0)).x;
        
        if(new_pos.z > new_z && new_pos.z < new_z + fZThickness * len_mult)
        {
            ray.hit = true;
            break;
        }
        
        ray.pos = new_pos;
        ray.uv = new_coord;
    }
}

// <---- Denoising ---->

// reblur
float getFrustumSize(float z)
{
    return 2.0f * z * tan(radians(iVerticalFOV) * 0.5);
}
float getSpecularLobeHalfAngle(float roughness, float precent_volume)
{
    return atan( roughness * roughness * precent_volume / ( 1.0 - precent_volume ) );
}
float specMagicCurve(float roughness, float precent_volume)
{
    float angle = getSpecularLobeHalfAngle(roughness, precent_volume);
    float almostHalfPi = getSpecularLobeHalfAngle(1.0, precent_volume);

    return saturate(angle / almostHalfPi);
}

float getHitDistScale(float z, float roughness)
{
    return (fHitDistParams.x + abs(z) * fHitDistParams.y) * lerp(1.0, fHitDistParams.z, saturate(exp2(fHitDistParams.w * roughness * roughness)));
}
float getHitDistRadiusFactor(float hit_dist, float frustum_size)
{
    return saturate(hit_dist / frustum_size);
}
float2 getGeometryWeightParams(float frustum_size, float3 world_pos, float3 normal, float nonlinear_accum_speed)
{
    float relaxation = lerp(1.0, 0.25, nonlinear_accum_speed);
    float a = relaxation / (fGeometrySensitivity * frustum_size);
    float b = -dot(normal, world_pos) * a;

    return float2(a, b);
}
float getNormalWeightParams(float nonlinear_accum_speed, float fraction, float roughness)
{
    float angle = getSpecularLobeHalfAngle(roughness, 0.75);
    angle *= lerp(saturate(fraction), 1.0, nonlinear_accum_speed);

    return 1.0 / max(angle, 2.0 / 255.0);
}
float2 getHitDistanceWeightParams(float hit_dist, float nonlinear_accum_speed, float roughness)
{
    float smc = specMagicCurve(roughness, 0.987);
    float norm = lerp(EPS, 1.0, min(nonlinear_accum_speed, smc));
    float a = 1.0 / norm;
    float b = hit_dist * a;

    return float2(a, -b);
}
float getCombinedWeight(
    float3 normal,
    float3 sample_pos, float3 sample_normal, float sample_roughness,
    float2 geom_params, float normal_params, float2 rough_params)
{
    float3 a = float3(geom_params.x, normal_params, rough_params.x);
    float3 b = float3(geom_params.y, 0, rough_params.y);

    float3 t = float3(dot(normal, sample_pos), acos(saturate(dot(normal, sample_normal))), sample_roughness);
    
    float3 w = exp(-3.0 * abs(t * a + b));

    return w.x * w.y * w.z;
}

float4 spatialBlur(sampler gi_sampler, sampler gi_hitdist_sampler, VSOUT vsout)
{
    float4 gi = tex2D(gi_sampler, vsout.texcoord);

    // pre blur
    float3 world_pos = coords2WorldPos(vsout.texcoord, vsout);
    float3 normal = getScreenNormal(vsout.texcoord, vsout);
    float frustum_size = getFrustumSize(world_pos.z);
    float nonlinear_accum_speed = fPreBlurNonlinearAccumSpeed;

    // determine radius
    float hit_dist = tex2D(gi_hitdist_sampler, vsout.texcoord).x;
    float hit_dist_factor = getHitDistRadiusFactor(hit_dist * getHitDistScale(world_pos.z, 1.0), frustum_size);

    float blur_radius = fPreBlurRadius * hit_dist_factor;

    // weight params
    float2 geom_params = getGeometryWeightParams(frustum_size, world_pos, normal, nonlinear_accum_speed);
    float normal_params = getNormalWeightParams(nonlinear_accum_speed, fLobeAngleFraction, 1.0);
    float2 hit_dist_params = getHitDistanceWeightParams(hit_dist, nonlinear_accum_speed, 1.0);
    float min_hit_dist_weight = 0.1 * fLobeAngleFraction;

    // normal dir skew
    float3x3 kernel_basis = getBasis(normal);
    float world_radius = blur_radius * frustum_size / YASSGI_BUFFER_HEIGHT;
    kernel_basis[0] *= world_radius;
    kernel_basis[1] *= world_radius;

    float weightsum = gi.w > EPS;
    float4 blurred_color = weightsum * gi;
    [unroll]
    for(uint n = 0; n < 8; ++n)
    {
        float3 offset = g_Poisson8[n];

        float3 tangent = kernel_basis[0];
        float3 bitangent = kernel_basis[1];
        
        // TODO handle specular skew when specular available

        float2 rotated_offset = mul(getRotateMatrix(FRAMECOUNT/PI), offset.xy);
        float3 world_offset = tangent * rotated_offset.x + bitangent * rotated_offset.y;
        float3 sample_pos = world_pos + world_offset;
        float2 sample_uv = worldPos2Coords(sample_pos, vsout);

        float4 sample_gi = tex2D(gi_sampler, sample_uv);
        float3 sample_normal = getScreenNormal(sample_uv, vsout);

        // weighting
        // no need for material comparison i guess
        float w = exp(-0.66 * offset.z * offset.z);  // Base gaussian weight
        w *= getCombinedWeight(normal, sample_pos, sample_normal, 1.0, geom_params, normal_params, 0.0);
        w *= lerp(min_hit_dist_weight, 1.0, exp(-3.0 * abs(hit_dist * hit_dist_params.x + hit_dist_params.y)));

        // screen check
        w = isInScreen(sample_uv) ? w : 0.0;
        sample_gi = w ? sample_gi : 0.0;

        // accumulate
        weightsum += w;
        blurred_color += sample_gi * w;
    }
    blurred_color /= weightsum;

    return blurred_color;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pixel Shaders
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void PS_InputBufferSetup(in VSOUT vsout, out float4 z : SV_Target0, out float4 z_prev : SV_Target1)
{
    z_prev = tex2Dfetch(ZSampler, vsout.texcoord * ReShade::PixelSize);
    z = depth2Z(ReShade::GetLinearizedDepth(vsout.texcoord));
}

void PS_Trace(in VSOUT vsout, out float4 color: SV_Target0, out float hit_dist: SV_Target1)
{
    float3 world_pos = coords2WorldPos(vsout.texcoord, vsout);
    float3 normal = getScreenNormal(vsout.texcoord, vsout);

    float max_dist = fBaseStride * (1 << (iMaxRayStep / fSpreadStep) - 1) * fSpreadStep;

    color = 0;
    
    [loop]
    for(int r = 0; r < iRayAmount; r++)
    {
        RayInfo ray;

        ray.orig = world_pos;
        float3 rand3 = rand4dTo3d(float4(vsout.texcoord, frac(FRAMECOUNT / PI), r / SQRT2));
        // ray.dir = normalize(normal + uniformHemisphere(rand3.x, rand3.y)) * (1 + (rand3.z - 1) * fStrideJitter);
        ray.dir = uniformLambert(rand3.xy, normal) * (1 + (rand3.z - 1) * fStrideJitter);
        ray.spread_fact = fSpreadStep;

        traceRay(ray, vsout);

        float3 hit_color = 0.0;
        if(!ray.hit)
        {
            color.rgb += mul(fAmbientLight, sRGBtoXYZ);
            continue;
        }

        color.w += 1;
        
        // normal check
        float3 normal_end = getScreenNormal(ray.uv, vsout);
        bool is_backface = (sign(ray.pos.z - ray.orig.z) != sign(ray.dir.z)) || (dot(normal_end, ray.dir) > -0.02);  // the first one should be mitigated w/ more precise hit detection
        if(!bBackfaceLighting && is_backface)
            continue;

        hit_color = tex2Dlod(ReShade::BackBuffer, float4(ray.uv, int(ray.mip_level), 0)).rgb;
        hit_color = mul(hit_color, sRGBtoXYZ);
        hit_color *= abs(dot(normal, ray.dir));
        color.rgb += hit_color * saturate(1 - ray.travel_dist / max_dist);

        hit_dist += ray.travel_dist / iRayAmount;
    }

    color.rgb /= iRayAmount;
}

void PS_PreBlur(in VSOUT vsout, out float4 o: SV_Target0)
{
    o = spatialBlur(GISampler, GIHitDistSampler, vsout);
}

void PS_Accumulation(in VSOUT vsout, out float4 o: SV_Target0)
{
    float4 gi_accum = tex2D(GIAccumSampler, vsout.texcoord);
    float4 gi_curr = tex2D(GIPreBlurSampler, vsout.texcoord);
    o = lerp(gi_accum, gi_curr, rcp(1 + gi_accum.w * fAccumMult / min(FRAMETIME, 1.0)));
}

// void PS_Blur(in VSOUT vsout, out float4 o: SV_Target0)
// {
//     o = poissonBlur(GIAccumSampler, vsout.texcoord, 2);
// }

// void PS_PostBlur(in VSOUT vsout, out float4 o: SV_Target0)
// {
//     o = poissonBlur(GIAccumSampler, vsout.texcoord, 2);
// }

void PS_Display(in VSOUT vsout, out float4 color: SV_Target)
{
    if(iDebugView == 0)
    {
        color = tex2D(ReShade::BackBuffer, vsout.texcoord);
        color = mul(color.rgb, sRGBtoXYZ);
        color += tex2D(GIAccumSampler, vsout.texcoord).rgb * fMixMult;
        color = mul(color.rgb, XYZtoSRGB);
    }
    else if(iDebugView == 1)  // initial sample
        color = mul(tex2D(GISampler, vsout.texcoord).rgb, XYZtoSRGB);
    else if(iDebugView == 2)  // hit intensity
        color = tex2D(GISampler, vsout.texcoord).a / iRayAmount;
    else if(iDebugView == 3)  // pre-blur gi
        color = mul(tex2D(GIPreBlurSampler, vsout.texcoord).rgb, XYZtoSRGB);
    else if(iDebugView == 4)  // accumulated gi
        color = mul(tex2D(GIAccumSampler, vsout.texcoord).rgb, XYZtoSRGB);
}

technique YASSGI{
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_InputBufferSetup;
        RenderTarget0 = ZTex;
        RenderTarget1 = ZPrevTex;
    }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_Trace;
        RenderTarget0 = GITex;
        RenderTarget1 = GIHitDistTex;

        ClearRenderTargets = true;
    }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_PreBlur;
        RenderTarget0 = GIPreBlurTex;
    }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_Accumulation;
        RenderTarget0 = GIAccumTex;
    }
    // pass {
    //     VertexShader = VS_YASSGI;
    //     PixelShader = PS_Blur;
    //     RenderTarget0 = GIAccumTex;
    // }
    // pass {
    //     VertexShader = VS_YASSGI;
    //     PixelShader = PS_PostBlur;
    //     RenderTarget0 = GIAccumTex;
    // }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_Display;
    }
}
}