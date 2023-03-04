/*
    z: raw z
    depth: linearized z
    z direction: + going farther, - coming closer
    normal: pointing outwards
    color & gi: ACEScg
    pos: view space coordinates
*/

#include "ReShade.fxh"

namespace YASSGI
{

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Constants
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#define PI      3.14159265358979323846264
#define HALF_PI 1.5707963267948966
#define RCP_PI  0.3183098861837067

#define EPS 1e-6

#define BUFFER_SIZE uint2(BUFFER_WIDTH, BUFFER_HEIGHT)

#define YASSGI_NOISE_SIZE 128

// 0 - Simple Tracing
// (1) - Hi-Z Tracing
// (2) - Bitmask IL https://arxiv.org/abs/2301.11376
#ifndef YASSGI_TECHNIQUE
#   define YASSGI_TECHNIQUE 0
#endif

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

// color space conversion matrices
// src: https://www.colour-science.org/apps/  using CAT02
static const float3x3 g_sRGBToACEScg = float3x3(
    0.613117812906440,  0.341181995855625,  0.045787344282337,
    0.069934082307513,  0.918103037508582,  0.011932775530201,
    0.020462992637737,  0.106768663382511,  0.872715910619442
);
static const float3x3 g_ACEScgToSRGB = float3x3(
    1.704887331049502, -0.624157274479025, -0.080886773895704,
    -0.129520935348888,  1.138399326040076, -0.008779241755018,
    -0.024127059936902, -0.124620612286390,  1.148822109913262
);

#define g_colorInputMat g_sRGBToACEScg
#define g_colorOutputMat g_ACEScgToSRGB

// @source https://github.com/NVIDIAGameWorks/RayTracingDenoiser/blob/master/Shaders/Include/Poisson.hlsli
// samples = 8, min distance = 0.5, average samples on radius = 2
static const float3 g_Poisson8[8] =
{
    float3( -0.4706069, -0.4427112, +0.6461146 ),
    float3( -0.9057375, +0.3003471, +0.9542373 ),
    float3( -0.3487388, +0.4037880, +0.5335386 ),
    float3( +0.1023042, +0.6439373, +0.6520134 ),
    float3( +0.5699277, +0.3513750, +0.6695386 ),
    float3( +0.2939128, -0.1131226, +0.3149309 ),
    float3( +0.7836658, -0.4208784, +0.8895339 ),
    float3( +0.1564120, -0.8198990, +0.8346850 )
};

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Uniform Varibales
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

uniform uint  iFrameCount  < source = "framecount"; >;
uniform float fFrameTime   < source = "frametime";  >;

uniform int iViewMode <
	ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "YASSGI\0Depth / Normal\0GI / AO (Raw)\0GI / AO (Accumulated)\0";
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

// <---- Sampling ---->

uniform uint iNumSample <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Sample Count";
    ui_min = 1; ui_max = 32;
    ui_step = 1;
> = 4;

uniform uint iNumSteps <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Max Steps";
    ui_min = 1; ui_max = 32;
    ui_step = 1;
> = 12;

uniform float fBaseStride <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Base Stride";
    ui_min = 0.01; ui_max = 10.0;
    ui_step = 0.01;
> = 1.0;

uniform float fSpreadExp <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Spread Exponent";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.3;

uniform float fZThickness <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Z Thickness";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
> = 4.0;

// <---- Shading ---->

uniform float fMatRoughness <
    ui_type = "slider";
    ui_category = "Shading";
    ui_label = "Material: Roughness";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.9;

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
    ui_category = "Accumulation";
    ui_label = "Normal Sensitivity";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.4;

// <---- Mixing ---->

uniform float fLightSrcThres <
	ui_type = "slider";
    ui_label = "Light Source Threshold";
    ui_tooltip = "Only pixels with greater brightness are considered light-emitting.";
	ui_category = "Mixing";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.5;

uniform float fBackfaceLightMult <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "Backface Lighting";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.0;

uniform float fBounceMult <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "Bounce Strength";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.01;
> = 0.9;

uniform float fIlStrength <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "IL Strength";
    ui_min = 0.0; ui_max = 100.0;
    ui_step = 0.1;
> = 1.0;

uniform float fAoStrength <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "AO Strength";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
> = 1.0;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Buffers
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// blue noise
texture tex_blue_noise   <source ="YASSGI_bleu.png";> { Width = YASSGI_NOISE_SIZE; Height = YASSGI_NOISE_SIZE; Format = RGBA8; };
sampler samp_blue_noise                               { Texture = tex_blue_noise; AddressU = REPEAT; AddressV = REPEAT; AddressW = REPEAT;};

// color
texture tex_color  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = YASSGI_MIP_LEVEL;};
sampler samp_color {Texture = tex_color;};

// normal & z
texture tex_g  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = YASSGI_MIP_LEVEL;};
sampler samp_g {Texture = tex_g;};

// normal & z (previous frame)
texture tex_g_prev  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_g_prev {Texture = tex_g_prev;};

// gi & ao
texture tex_gi_ao  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao {Texture = tex_gi_ao;};

// gi & ao, accumulated
texture tex_gi_ao_accum  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = RGBA16F; MipLevels = YASSGI_MIP_LEVEL;};
sampler samp_gi_ao_accum {Texture = tex_gi_ao_accum;};

texture tex_accum_speed  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = R16F; MipLevels = 1;};
sampler samp_accum_speed {Texture = tex_accum_speed;};

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// <---- Util & Math ---->


// <---- Depth & Normal ---->

float zToLinearDepth(float z) {return (z - 1) * rcp(RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);}
float linearDepthToZ(float depth) {return depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE + 1;}

float getLinearDepth(float2 uv) {return ReShade::GetLinearizedDepth(uv);}
float getZ(float2 uv) {return linearDepthToZ(getLinearDepth(uv));}

float3 uvToViewSpace(float2 uv, float z){return float3(uv * 2 - 1, 1) * z;}
float3 uvToViewSpace(float2 uv){return uvToViewSpace(uv, getZ(uv));}
float2 viewSpaceToUv(float3 pos){return (pos.xy / pos.z + 1) * 0.5;}

// src: https://gist.github.com/bgolus/a07ed65602c009d5e2f753826e8078a0
// src fr: https://atyuwen.github.io/posts/normal-reconstruction
// normal points outwards from the hull
float3 getViewNormalAccurate(float2 uv)
{
    float3 view_pos = uvToViewSpace(uv);

    float2 off1 = float2(BUFFER_RCP_WIDTH, 0);
    float2 off2 = float2(0, BUFFER_RCP_HEIGHT);

    float3 view_pos_l = uvToViewSpace(uv - off1);
    float3 view_pos_r = uvToViewSpace(uv + off1);
    float3 view_pos_d = uvToViewSpace(uv - off2);
    float3 view_pos_u = uvToViewSpace(uv + off2);

    float3 l = view_pos - view_pos_l;
    float3 r = view_pos_r - view_pos;
    float3 d = view_pos - view_pos_d;
    float3 u = view_pos_u - view_pos;

    // get depth values at 1 & 2 px offset along both axis
    float4 H = float4(
        getZ(uv - off1),
        getZ(uv + off1),
        getZ(uv - 2 * off1),
        getZ(uv + 2 * off1)
    );
    float4 V = float4(
        getZ(uv - off2),
        getZ(uv + off2),
        getZ(uv - 2 * off2),
        getZ(uv + 2 * off2)
    );
    
    // current pixel's depth difference
    float2 he = abs((2 * H.xy - H.zw) - view_pos.z);
    float2 ve = abs((2 * V.xy - V.zw) - view_pos.z);

    // pick horizontal and vertical diff with smallest depth difference from slopes
    float3 h_deriv = he.x < he.y ? l : r;
    float3 v_deriv = ve.x < ve.y ? d : u;

    return normalize(cross(v_deriv, h_deriv));
}

bool isNear(float z){return z < 1;}
bool isFar(float z){return z > RESHADE_DEPTH_LINEARIZATION_FAR_PLANE + 1;}
bool isWeapon(float z){return z < linearDepthToZ(fDepthRange.x);}
bool isSky(float z){return z > linearDepthToZ(fDepthRange.y);}

// techniquely not depth but I just wanna put it here
bool isInScreen(float2 uv)
{
    return uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0;
}

// <---- Random & Sampling ---->

// src http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
float2 r2(int n)
{
	return frac(n * float2(0.754877666246692760049508896358532874940835564978799543103, 0.569840290998053265911399958119574964216147658520394151385));
}

// return r, theta
float2 sampleDisk(float2 rand2)
{
    return float2(sqrt(rand2.x), 2 * PI * rand2.y);
}
float3 sampleHemisphereUniform(float2 rand2, float3 normal)
{
    float cos_theta = rand2.x;
    float sin_theta = sqrt(1 - cos_theta * cos_theta);
    float phi = 2 * PI * rand2.y;

    float3 h = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    float3 up_vec = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tan_x = normalize(cross(up_vec, normal));
    float3 tan_y = cross(normal, tan_x);
    return normalize(tan_x * h.x + tan_y * h.y + normal * h.z);
}

// <---- GI Methods ---->

struct RayInfo
{
    float3 orig;
    float3 stride;
    float spread_exp;

    float3 pos;
    float2 uv;
    bool hit;
    float spread_level;
};

void simpleRayMarch(inout RayInfo ray)
{
    ray.hit = false;

    ray.pos = ray.orig;
    [loop]
    for(int step = 0; step < iNumSteps; ++step)
    {
        ray.spread_level = step * ray.spread_exp;

        float len_mult = exp2(ray.spread_level);
        ray.pos += ray.stride * len_mult;
        ray.uv = viewSpaceToUv(ray.pos);
        
        [branch]
        if(!isInScreen(ray.uv) || isNear(ray.pos.z) || isSky(ray.pos.z))
            break;

        float z_curr = tex2Dlod(samp_g, float4(ray.uv, ray.spread_level, 0)).w;

        [branch]
        if(ray.pos.z > z_curr && ray.pos.z < z_curr + fZThickness * len_mult)
        {
            ray.hit = true;
            break;
        }
    }
}

float bsdf_OrenNayar(float roughness, float nov, float nol, float voh)
{
	float a = roughness * roughness;
	float s = a;// / ( 1.29 + 0.5 * a );
	float s2 = s * s;
	float vol = 2 * voh * voh - 1;		// double angle identity
	float cosri = vol - nov * nol;
	float c1 = 1 - 0.5 * s2 / (s2 + 0.33);
	float c2 = 0.45 * s2 / (s2 + 0.09) * cosri * ( cosri >= 0 ? rcp( max( nol, nov ) ) : 1 );
	return ( c1 + c2 ) * ( 1 + roughness * 0.5 ) * RCP_PI;
}

// <---- Blur ---->
float4 spatialBlur(sampler gi_ao_sampler, float2 uv, float radius, float accum_speed)
{
    float4 gi_ao = tex2D(gi_ao_sampler, uv);

    float weightsum = 0;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pixel Shaders
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void PS_SavePrev(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 g_prev : SV_Target0)
{
    g_prev = tex2Dfetch(samp_g, int2(uv * BUFFER_SIZE));
}

void PS_InputSetup(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0, out float4 g : SV_Target1)
{
    // color
    color.rgb = mul(tex2D(ReShade::BackBuffer, uv).rgb, g_colorInputMat);
    color.a = 1;

    // g
    float z = getZ(uv);
    float3 normal = getViewNormalAccurate(uv);

    if(isWeapon(z))
    {
        z = ((z - 1) * fWeapDepthMult) + 1;
        normal = normalize(normal * float3(1, 1, rcp(fWeapDepthMult)));
    }

    g = float4(normal, z);
}

void PS_GI(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 gi_ao : SV_Target0
)
{
    gi_ao = 0;

    float4 g = tex2D(samp_g, uv);

    float3 pos_orig = uvToViewSpace(uv, g.w);
    [branch]
    if(isNear(pos_orig.z) || isSky(pos_orig.z))  // leave sky alone
        return;

    float3 normal_orig = g.xyz;
    float3 viewdir_orig = normalize(pos_orig);

    float rcp_numsample = rcp(iNumSample);

    [loop]
    for(uint i = 0; i < iNumSample; ++i)
    {
        float3 rand3 = tex2Dfetch(samp_blue_noise,
            int2((uv * YASSGI_GI_BUFFER_SIZE + r2(iFrameCount + i) * YASSGI_NOISE_SIZE) % YASSGI_NOISE_SIZE)).xyz;

        RayInfo ray;
        ray.orig = pos_orig;
        float3 raydir = sampleHemisphereUniform(rand3.xy, normal_orig);
        ray.stride = raydir * fBaseStride * (1 + (rand3.z - 1) * 0.5);
        ray.spread_exp = fSpreadExp;
        
        simpleRayMarch(ray);
        [branch]
        if(!ray.hit)
            continue;

        // AO
        gi_ao.w += rcp_numsample * RCP_PI * 0.5;  // TODO consider differnt sampling scheme

        // normal check
        float3 normal_end = tex2Dlod(samp_g, float4(ray.uv, 0, 0)).xyz;
        bool is_backface = dot(normal_end, raydir) > -EPS;

        // determine source color
        float3 hit_color = tex2Dlod(samp_color, float4(ray.uv, ray.spread_level, 0)).rgb;
        hit_color += tex2Dlod(samp_gi_ao_accum, float4(ray.uv, ray.spread_level, 0)).rgb * fBounceMult;
        hit_color = length(hit_color) > fLightSrcThres ? hit_color : 0;
        hit_color *= is_backface ? fBackfaceLightMult : 1;

        // brdf
        hit_color *= saturate(bsdf_OrenNayar(fMatRoughness,
                                             max(dot(normal_orig, -viewdir_orig), EPS),
                                             max(dot(normal_orig, raydir), EPS),
                                             max(dot(raydir, -viewdir_orig), EPS)));
                                        
        // sampling
        hit_color *= RCP_PI * 0.5;
        
        gi_ao.rgb += hit_color * rcp_numsample;
    }
}

void PS_Accumulation(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 gi_ao_accum : SV_Target0, out float4 o_accum_speed : SV_Target1)
{
    float4 gi_accum = tex2D(samp_gi_ao_accum, uv);
    float accum_speed = tex2Dlod(samp_accum_speed, float4(uv, 1, 0)).x;
    float4 gi_curr = tex2D(samp_gi_ao, uv);
    
    float4 g_curr = tex2D(samp_g, uv);
    float4 g_prev = tex2D(samp_g_prev, uv);

    // z & normal disocclusion
    float4 delta = abs(g_curr - g_prev) / max(fFrameTime, 1.0);
    float normal_delta = dot(delta.xyz, delta.xyz);
    float z_delta = delta.w / g_curr.w;
    float quality = exp(-normal_delta * fNormalSensitivity * 1e3 - z_delta * fZSensitivity * 1e3);

    float accum_speed_new = min(accum_speed * quality + 1, iMaxAccumFrames) ;
    float4 gi_new = lerp(gi_accum, gi_curr, rcp(accum_speed_new));

    // finalize
    gi_ao_accum = gi_new;
    o_accum_speed = accum_speed_new;
}

void PS_Display(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target)
{
    [branch]
    if(iViewMode == 0)  // None
    {
        float4 gi_ao = tex2D(samp_gi_ao_accum, uv);
        float4 albedo = tex2D(samp_color, uv);
        color.rgb = albedo.rgb;
        color.rgb += gi_ao.rgb * fIlStrength;
        color.rgb /= 1 + gi_ao.w * fAoStrength;
        color.rgb = saturate(mul(color.rgb, g_colorOutputMat));
    }
    else if(iViewMode == 1)  // Depth / Normal
    {
        float4 g = tex2D(samp_g, uv);
        if((iFrameCount / 300) % 2)  // Normal
        {
            color = g.xyz * 0.5 * float3(1, 1, -1) + 0.5;  // for convention
        }
        else  // Depth
        {
            color = zToLinearDepth(g.w);
            if(color.r < fDepthRange.x)
                color = float3(color.r / fDepthRange.x, 0, 0);
            else if (color.r > fDepthRange.y)
                color = float3(0.1, 0.5, 1.0);
            color.a = 1;
        }
    }
    else if(iViewMode == 2)  // GI
    {
        color = (iFrameCount / 300) % 2 ?
            rcp(1 + tex2D(samp_gi_ao, uv).w * fAoStrength) :
            mul(tex2D(samp_gi_ao, uv).xyz * fIlStrength, g_colorOutputMat);
    }
    else if(iViewMode == 3)  // GI Accum
    {
        color = (iFrameCount / 300) % 2 ?
            rcp(1 + tex2D(samp_gi_ao_accum, uv).w * fAoStrength) :
            mul(tex2D(samp_gi_ao_accum, uv).xyz * fIlStrength, g_colorOutputMat);
    }
}

technique YASSGI{
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_SavePrev;
        RenderTarget0 = tex_g_prev;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_InputSetup;
        RenderTarget0 = tex_color;
        RenderTarget1 = tex_g;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_GI;
        RenderTarget0 = tex_gi_ao;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Accumulation;
        RenderTarget0 = tex_gi_ao_accum;
        RenderTarget1 = tex_accum_speed;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Display;
    }
}

}