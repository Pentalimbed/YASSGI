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

namespace YASSGI{
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Buffers
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
texture ZTex	    { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; MipLevels = YASSGI_MIP_LEVEL; };
sampler ZTexSampler { Texture = ZTex; };

texture ZPrevTex	    { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; MipLevels = YASSGI_MIP_LEVEL; };
sampler ZPrevTexSampler { Texture = ZPrevTex; };

// Alpha channel = hit number ~= AO
texture GITex	     { Width = YASSGI_BUFFER_WIDTH; Height = YASSGI_BUFFER_HEIGHT; Format = RGBA16F; };
sampler GITexSampler { Texture = GITex; };

texture GIAccumTex        { Width = YASSGI_BUFFER_WIDTH; Height = YASSGI_BUFFER_HEIGHT; Format = RGBA16F; };
sampler GIAccumTexSampler { Texture = GIAccumTex; };

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
    ui_items = "None\0Initial Sample\0Hit Intensity\0Accumulated GI\0";
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

float3 normal_from_depth(in float2 uv, in VSOUT vsout)
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

/// @source https://www.ronja-tutorials.com/post/024-white-noise/#:~:text=For%20many%20effects%20we%20want%20random%20numbers%20to,other%20tutorials%2C%20for%20example%20perlin%20and%20voronoi%20noise.
float rand4dTo1d(float4 value, float4 dotDir)
{
    //make value smaller to avoid artefacts
    float4 smallValue = sin(value);
    //get scalar value from 3d vector
    float random = dot(smallValue, dotDir);
    //make value more random by making it bigger and then taking teh factional part
    random = frac(sin(random) * 143758.5453);
    return random;
}
float3 rand4dTo3d(float4 value){
    return float3(
        rand4dTo1d(value, float4(86.3528267, 92.075745, 67.4930429, 7.370599)),
        rand4dTo1d(value, float4(26.669095, 86.7327491, 57.2284329, 22.1987259)),
        rand4dTo1d(value, float4(10.2522207, 32.5525956, 25.4438045, 45.0742175))
    );
}

/// @source http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
float3 uniformHemisphere(float2 rand, float3 normal)
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
        float len_mult = pow(2, ray.mip_level);
        float3 new_pos = ray.pos + stride * len_mult;
        float2 new_coord = worldPos2Coords(new_pos, vsout);
        ray.travel_dist += len_stride * len_mult;

        if(!isInScreen(new_coord) || isNear(new_pos.z) || isSky(new_pos.z))
            break;

        float new_z = tex2Dlod(ZTexSampler, float4(new_coord, int(ray.mip_level), 0)).x;
        
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

float2 rotate2d(float2 vec, float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mul(float2x2(c, -s, s, c), vec);
}

float4 poissonBlur(sampler tex, float2 uv, float radius)
{
    float4 color = tex2D(tex, uv);
    [loop]
    for(int i=0; i<8; ++i)
    {
        float2 poisson_pt = g_Poisson8[i].xy * radius * ReShade::PixelSize / YASSGI_RENDER_SCALE;
        color += tex2D(tex, uv + rotate2d(poisson_pt, FRAMECOUNT / 10.0));
    }
    return color / 9;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pixel Shaders
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void PS_InputBufferSetup(in VSOUT vsout, out float4 z : SV_Target0, out float4 z_prev : SV_Target1)
{
    z_prev = tex2Dfetch(ZTexSampler, vsout.texcoord * ReShade::PixelSize);
    z = depth2Z(ReShade::GetLinearizedDepth(vsout.texcoord));
}

void PS_Trace(in VSOUT vsout, out float4 color: SV_Target0)
{
    float3 world_pos = coords2WorldPos(vsout.texcoord, vsout);
    float3 normal = normal_from_depth(vsout.texcoord, vsout);

    float max_dist = fBaseStride * (1 << (iMaxRayStep / fSpreadStep) - 1) * fSpreadStep;

    color = 0;
    
    [loop]
    for(int r = 0; r < iRayAmount; r++)
    {
        RayInfo ray;

        ray.orig = world_pos;
        float3 rand3 = rand4dTo3d(float4(vsout.texcoord, RANDOM / PI, (r + FRAMECOUNT) / SQRT2));
        // ray.dir = normalize(normal + uniformHemisphere(rand3.x, rand3.y)) * (1 + (rand3.z - 1) * fStrideJitter);
        ray.dir = uniformHemisphere(rand3.xy, normal) * (1 + (rand3.z - 1) * fStrideJitter);
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
        float3 normal_end = normal_from_depth(ray.uv, vsout);
        bool is_backface = (sign(ray.pos.z - ray.orig.z) != sign(ray.dir.z)) || (dot(normal_end, ray.dir) > -0.02);  // the first one should be mitigated w/ more precise hit detection
        if(!bBackfaceLighting && is_backface)
            continue;

        hit_color = tex2Dlod(ReShade::BackBuffer, float4(ray.uv, int(ray.mip_level), 0)).rgb;
        hit_color = mul(hit_color, sRGBtoXYZ);
        hit_color *= abs(dot(normal, ray.dir));
        color.rgb += hit_color * saturate(1 - distance(ray.orig, ray.pos) / max_dist);
    }

    color.rgb /= iRayAmount;
}

void PS_PreBlur_Accumulation(in VSOUT vsout, out float4 o: SV_Target0)
{
    // pre blur
    float4 gi_curr = tex2D(GITexSampler, vsout.texcoord);
    float4 gi_accum = poissonBlur(GIAccumTexSampler, vsout.texcoord, 2);

    // accum
    o = lerp(gi_accum, gi_curr, rcp(1 + gi_accum.w * fAccumMult / min(FRAMETIME, 1.0)));
}

void PS_Blur(in VSOUT vsout, out float4 o: SV_Target0)
{
    o = poissonBlur(GIAccumTexSampler, vsout.texcoord, 2);
}

void PS_PostBlur(in VSOUT vsout, out float4 o: SV_Target0)
{
    o = poissonBlur(GIAccumTexSampler, vsout.texcoord, 2);
}

void PS_Display(in VSOUT vsout, out float4 color: SV_Target)
{
    if(iDebugView == 0)
    {
        color = tex2D(ReShade::BackBuffer, vsout.texcoord);
        color = mul(color.rgb, sRGBtoXYZ);
        color += tex2D(GIAccumTexSampler, vsout.texcoord).rgb * fMixMult;
        color = mul(color.rgb, XYZtoSRGB);
    }
    else if(iDebugView == 1)  // initial sample
        color = mul(tex2D(GITexSampler, vsout.texcoord).rgb, XYZtoSRGB);
    else if(iDebugView == 2)  // hit intensity
        color = tex2D(GITexSampler, vsout.texcoord).a / iRayAmount;
    else if(iDebugView == 3)  // accumulated gi
        color = mul(tex2D(GIAccumTexSampler, vsout.texcoord).rgb, XYZtoSRGB);
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

        ClearRenderTargets = true;
    }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_PreBlur_Accumulation;
        RenderTarget0 = GIAccumTex;
    }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_Blur;
        RenderTarget0 = GIAccumTex;
    }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_PostBlur;
        RenderTarget0 = GIAccumTex;
    }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_Display;
    }
}
}