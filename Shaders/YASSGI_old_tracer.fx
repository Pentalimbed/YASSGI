/*
    z: raw z
    depth: linearized z
    z direction: + going farther, - coming closer
    normal: pointing outwards
    color & gi: ACEScg
    pos: view space coordinates
*/

/*  TODO
    ? firefly suppression
    o optical flow reprojection  
    - material properties
    o sky (kinda)
    x use that blue noise somehow
    X bitmask il
    o licence sanitization
    - dx9 check to toggle bit ops
*/

#include "ReShade.fxh"

namespace YASSGI_Old_Tracer
{

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Constants
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#define PI      3.14159265358979323846264
#define HALF_PI 1.5707963267948966
#define RCP_PI  0.3183098861837067

#define EPS 1e-6

#define BUFFER_SIZE uint2(BUFFER_WIDTH, BUFFER_HEIGHT)

#ifndef YASSGI_RENDER_SCALE
#   define YASSGI_RENDER_SCALE 0.5
#endif

#define YASSGI_GI_BUFFER_WIDTH (BUFFER_WIDTH * YASSGI_RENDER_SCALE)
#define YASSGI_GI_BUFFER_HEIGHT (BUFFER_HEIGHT * YASSGI_RENDER_SCALE)
#define YASSGI_GI_BUFFER_SIZE (BUFFER_SIZE * YASSGI_RENDER_SCALE)

#ifndef YASSGI_MIP_LEVEL
#   define YASSGI_MIP_LEVEL 5
#endif

#define YASSGI_BITMASK_SIZE 16
#define YASSGI_SECTOR_ANGLE (PI / YASSGI_BITMASK_SIZE)

#ifndef YASSGI_USE_MOTION
#   define YASSGI_USE_MOTION 0
#endif

// color space conversion matrices
// data src: https://www.colour-science.org/apps/  using CAT02
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

// in case someone discovered a better color space
#define g_colorInputMat g_sRGBToACEScg
#define g_colorOutputMat g_ACEScgToSRGB

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Uniform Varibales
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

uniform uint  iFrameCount  < source = "framecount"; >;
uniform float fFrameTime   < source = "frametime";  >;

uniform int iViewMode <
	ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "YASSGI\0Depth / Normal\0Fake Albedo\0GI / AO (Raw)\0GI / AO (Accumulated)\0Accumulated Frames\0";
> = 0;

// <---- Input ---->

uniform int iFov <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "Vertical FOV";
    ui_min = 60; ui_max = 150;
    ui_step = 1;
> = 90;

uniform float2 fDepthRange <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "Weapon/Sky Depth Range";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
> = float2(0.001, 0.999);

uniform float fWeapDepthMult <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "Weapon Depth Multiplier";
    ui_tooltip = "Many FPS games squash their weapon into a super flat pancake.\n"
        "You can check it in the depth/normal debug view. Red parts are the weapons.\n"
        "Crank this up to free them from oppression and bring them back to illumination.";
    ui_min = 1.0; ui_max = 100.0;
    ui_step = 0.1;
> = 1.0;

uniform float fLightSrcThres <
	ui_type = "slider";
    ui_label = "Light Source Threshold";
    ui_tooltip = "Only pixels brighter than this are considered light-emitting.";
	ui_category = "Input";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.1;

uniform float fAlbedoSatPower <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "Albedo Saturation Power";
    ui_tooltip = "Since ReShade has no way of knowing the true albedo of a surface separate from lighting,\n"
        "any shader has to guess. A value of 0.0 tells the shader that everything is monochrome, and its\n"
        "hue is the result of lighting. Greater value yields more saturated output on colored surfaces.\n";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.01;
> = 1.0;

uniform float fAlbedoNorm <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "Albedo Normalization";
    ui_tooltip = "Since ReShade has no way of knowing the true albedo of a surface separate from lighting,\n"
        "any shader has to guess. A value of 0.0 tells the shader that there is no lighting in the scene,\n"
        "so dark surfaces are actually black. 1.0 says that all surfaces are in fact colored brightly, and\n"
        "the variation in brightness are the result of illumination, rather than the texture pattern itself.";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.8;

// <---- Sampling ---->

uniform uint iNumSample <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Sample Count";
    ui_min = 1; ui_max = 32;
    ui_step = 1;
> = 6;

uniform uint iNumSteps <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Max Steps";
    ui_min = 1; ui_max = 32;
    ui_step = 1;
> = 8;

uniform float fBaseStride <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Base Stride";
    ui_min = 0.01; ui_max = 10.0;
    ui_step = 0.01;
> = 8.0;

uniform float fDepthScaledStride <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Depth Scaled Stride";
    ui_min = 0.00; ui_max = 1.0;
    ui_step = 0.01;
> = 1.0;

uniform float fSpreadExp <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Spread Exponent";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.7;

uniform float fStrideJitter <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Stride Jitter";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.75;

uniform float fZThickness <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Z Thickness";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
> = 2.5;

// <---- Temporal Accumulation ---->

uniform int iMaxAccumFrames <
    ui_type = "slider";
    ui_category = "Accumulation";
    ui_label = "Max Accumulated Frames";
    ui_min = 1; ui_max = 64;
    ui_step = 1;
> = 12;

uniform float fDisocclThres <
    ui_type = "slider";
    ui_category = "Accumulation";
    ui_label = "Disocclusion Threshold";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.5;

// <---- Mixing ---->

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
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.01;
> = 2.0;

uniform float fBackfaceLightMult <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "Backface Lighting";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.05;

uniform float fSkylightMult <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "Skylight";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.2;

uniform float fBounceMult <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "Bounce";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.1;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Buffers
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

}

#if YASSGI_USE_MOTION
// motion vectors via other fx
texture texMotionVectors          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; };
sampler sMotionVectorTex         { Texture = texMotionVectors;  };
#endif

namespace YASSGI_Old_Tracer
{
// color
texture tex_color  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = YASSGI_MIP_LEVEL;};
sampler samp_color {Texture = tex_color;};

// normal (RGB) & z (A)
texture tex_g  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = YASSGI_MIP_LEVEL;};
sampler samp_g {Texture = tex_g;};

// normal & z (previous frame)
texture tex_g_prev  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_g_prev {Texture = tex_g_prev;};

// gi (RGB) & ao (A)
texture tex_gi_ao  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao {Texture = tex_gi_ao;};

// gi & ao, accumulated
texture tex_gi_ao_accum  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao_accum {Texture = tex_gi_ao_accum;};

// hist len (R)
texture tex_temporal  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = R16F;};
sampler samp_temporal {Texture = tex_temporal;};

texture tex_gi_ao_accum_1  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = RGBA16F; MipLevels = YASSGI_MIP_LEVEL;};
sampler samp_gi_ao_accum_1 {Texture = tex_gi_ao_accum_1;};

texture tex_temporal_1  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = R16F;};
sampler samp_temporal_1 {Texture = tex_temporal_1;};

// gi & ao blur
texture tex_gi_ao_blur1  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao_blur1 {Texture = tex_gi_ao_blur1;};

texture tex_gi_ao_blur2  {Width = YASSGI_GI_BUFFER_WIDTH; Height = YASSGI_GI_BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_gi_ao_blur2 {Texture = tex_gi_ao_blur2;};

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// <---- Input ---->

float zToLinearDepth(float z) {return z * rcp(RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);}
float linearDepthToZ(float depth) {return depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;}

float getLinearDepth(float2 uv) {return ReShade::GetLinearizedDepth(uv);}
float getZ(float2 uv) {return linearDepthToZ(getLinearDepth(uv));}

float getHalfFrustumSize(float z) {return z * tan(radians(iFov) * 0.5);}
float3 uvToViewSpace(float2 uv, float z){ return float3((uv * 2 - 1) * getHalfFrustumSize(z) * float2(BUFFER_WIDTH * BUFFER_RCP_HEIGHT, 1), z);}
float3 uvToViewSpace(float2 uv){return uvToViewSpace(uv, getZ(uv));}
float2 viewSpaceToUv(float3 pos){return (pos.xy / getHalfFrustumSize(pos.z) * float2(BUFFER_HEIGHT * BUFFER_RCP_WIDTH, 1) + 1) * 0.5;}

// src: WorldNormalFromDepthTexture.shader by Ben Golus https://gist.github.com/bgolus/a07ed65602c009d5e2f753826e8078a0
// algo: Accurate Normal Reconstruction from Depth Buffer by atyuwen https://atyuwen.github.io/posts/normal-reconstruction
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
bool isFar(float z){return z > RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;}
bool isWeapon(float z){return z < linearDepthToZ(fDepthRange.x);}
bool isSky(float z){return z > linearDepthToZ(fDepthRange.y);}

bool isInScreen(float2 uv){return all(uv > 0) && all(uv < 1);}

float3 fakeAlbedo(float3 color)
{
    float3 albedo = pow(max(0, color), fAlbedoSatPower * length(color));  // length(color) suppress saturation of darker colors
    albedo = saturate(lerp(albedo, normalize(albedo), fAlbedoNorm));
    return albedo;
}

// srgb to luminance: https://ninedegreesbelow.com/photography/srgb-luminance.html
// mult by conversion matrix
float luminance(float3 color)
{
    return dot(color, float3(0.21267291505, 0.71515223009, 0.07217499918));
}

// <---- Random & Sampling ---->

/// src: https://gist.github.com/hao5119266/f5b22c71a01310b5dd0db345bf4b5bfb
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

// return r, theta
float2 sampleDisk(float2 rand2)
{
    return float2(sqrt(rand2.x), 2 * PI * rand2.y);
}
// algo: https://alexanderameye.github.io/notes/sampling-the-hemisphere/
float3 sampleHemisphereUniform(float2 rand2, float3 normal, out float pdf)
{
    float cos_theta = rand2.x;
    float sin_theta = sqrt(1 - cos_theta * cos_theta);
    float phi = 2 * PI * rand2.y;

    float3 h = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    float3 up_vec = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tan_x = normalize(cross(up_vec, normal));
    float3 tan_y = cross(normal, tan_x);

    pdf = 0.5 * RCP_PI;

    return normalize(tan_x * h.x + tan_y * h.y + normal * h.z);
}
float3 sampleHemisphereCosWeighted(float2 rand2, float3 normal, out float pdf)
{
    float cos_theta = sqrt(rand2.x);
    float sin_theta = sqrt(1 - cos_theta * cos_theta);
    float phi = 2 * PI * rand2.y;

    float3 h = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    float3 up_vec = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    float3 tan_x = normalize(cross(up_vec, normal));
    float3 tan_y = cross(normal, tan_x);

    pdf = cos_theta * RCP_PI;

    return normalize(tan_x * h.x + tan_y * h.y + normal * h.z);
}

// <---- GI Methods ---->

struct RayInfo
{
    float3 orig;
    float3 stride;
    float spread_exp;

    float3 pos;
    float3 normal;
    float2 uv;
    bool hit;
    float spread_level;
};

// super naive intersection scheme with exponential steps
// started(deviated much) from: https://github.com/jebbyk/SSRT-for-reshade
// exponential steps and mip level inspired by: https://zhuanlan.zhihu.com/p/97886108 i.e. UE4 SSGI
// ^^^ to EPIC: not src!
void simpleRayMarch(inout RayInfo ray)
{
    ray.hit = false;

    ray.pos = ray.orig;
    [loop]
    for(int step = 0; step < iNumSteps; ++step)
    {
        ray.spread_level = step * ray.spread_exp;

        // make sure marched one pixel at least
        float2 min_len = ReShade::PixelSize * ray.pos.z / (ray.stride.xy);
        float len_mult = max(min(min_len.x, min_len.y), exp2(ray.spread_level));
        ray.pos += ray.stride * len_mult;
        ray.uv = viewSpaceToUv(ray.pos);
        
        [branch]
        if(!isInScreen(ray.uv) || isNear(ray.pos.z) || isSky(ray.pos.z))
            break;

        float4 g_curr = tex2Dlod(samp_g, float4(ray.uv, ray.spread_level, 0));
        ray.normal = g_curr.xyz;
        float z_curr = g_curr.w;
        ray.hit = ray.pos.z > z_curr && ray.pos.z < z_curr + fZThickness * len_mult;
        [branch]
        if(ray.hit)
            break;
    }
}

// <---- Blur ---->

// orig src (missing the latter parts): https://alain.xyz/blog/ray-tracing-denoising
// src: https://www.yuque.com/isumiai/cg/efwkig#Sf1rG
float3 fireflyRejectionVariance(float3 radiance, float3 variance, float3 mean)
{
    float3 stddev = sqrt(max(1e-5, variance));
    float3 high_thres = 0.1 + mean + stddev * 1.0;
    return min(radiance, high_thres);
}

// algo: Edge-avoiding À-trous https://jo.dreggn.org/home/2010_atrous.pdf
//     svgf (impl without variance) https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf
// referenced: https://www.shadertoy.com/view/tlXfRX
float4 atrous(sampler samp, float2 uv, float radius)
{
    float4 gi_curr = tex2D(samp, uv);

    float4 g_curr = tex2D(samp_g, uv);
    float2 zgrad = float2(ddx(g_curr.w), ddy(g_curr.w));

    float weightsum = 1;
    float4 sum = gi_curr;
    [unroll] for (int i = 0; i < 8; ++i)
    {
        float2 offset_px; sincos(i * 0.25 * PI, offset_px.y, offset_px.x); // <-sincos here!
        float2 uv_sample = uv + offset_px.xy * radius / YASSGI_GI_BUFFER_SIZE;

        float4 g_sample = tex2D(samp_g, uv_sample);
        float4 gi_sample = tex2D(samp, uv_sample);

        float w = pow(max(0, dot(g_curr.xyz, g_sample.xyz)), 64);                                   // normal
        w *= exp(-abs(g_curr.w - g_sample.w) / (1 * abs(dot(zgrad, offset_px.xy * radius)) + EPS)); // depth
        // w *= exp(-abs(lum_curr - lum[i]) / (fVarianceWeight * variance + EPS));                     // luminance
        w = saturate(w) * exp(-0.66 * length(offset_px)) * isInScreen(uv_sample);                   // gaussian kernel
        // ^^^ -four- three horsemen of the SVGF sampling weights

        weightsum += w;
        sum += gi_sample * w;
    }
    return (sum / weightsum);
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
    color.rgb = mul(g_colorInputMat, tex2D(ReShade::BackBuffer, uv).rgb);
    color.a = 1;

    // g
    float z = getZ(uv);
    float3 normal = getViewNormalAccurate(uv);

    // inspired by weapon-related stuff in AstrayFX/RadiantGI
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
    
    [branch]
    if(isNear(g.w) || isSky(g.w))  // leave sky alone
        return;

    float3 pos_orig = uvToViewSpace(uv, g.w);
    float3 normal_orig = g.xyz;
    float3 viewdir_orig = normalize(pos_orig);

    float3 color_orig = tex2D(samp_color, uv).rgb;
    float3 albedo = fakeAlbedo(color_orig);

    float rcp_numsample = rcp(iNumSample);

    float weightsum = 0;
    [loop]
    for(uint i = 0; i < iNumSample; ++i)
    {
        float3 rand3 = rand4dTo3d(float4(uv, iFrameCount * RCP_PI, i * RCP_PI));

        float pdf;

        RayInfo ray;
        ray.orig = pos_orig;
        float3 raydir = sampleHemisphereCosWeighted(rand3.xy, normal_orig, pdf);
        ray.stride = raydir * fBaseStride * (1 + (rand3.z - 1) * fStrideJitter);  // banding avoiding
        ray.stride *= lerp(1, max(EPS, zToLinearDepth(pos_orig.z)), fDepthScaledStride);  // sampling faraway pixels breaks cache coherency
        ray.spread_exp = fSpreadExp;
        
        simpleRayMarch(ray);
        [branch]
        if(!ray.hit)
        {
            // super cheese sky but looking good
            gi_ao.rgb += isInScreen(ray.uv) && isSky(tex2Dlod(samp_g, float4(ray.uv, 0, 0)).w) ?
                tex2Dlod(samp_color, float4(ray.uv, ray.spread_level, 0)).rgb * albedo * PI * fSkylightMult * rcp_numsample :
                0;
            continue;
        } 

        // AO
        gi_ao.w += rcp_numsample;  // TODO consider differnt sampling scheme

        // normal check
        bool is_backface = dot(ray.normal, raydir) > -EPS;

        // determine source color
        float3 hit_color = tex2Dlod(samp_color, float4(ray.uv, ray.spread_level, 0)).rgb;
        hit_color += tex2Dlod(samp_gi_ao_accum_1, float4(ray.uv, ray.spread_level, 0)).rgb * fBounceMult;  // lazy multi bounce
        hit_color = luminance(hit_color) > fLightSrcThres ? hit_color : 0;  // straight up length() is ok too I guess but maybe costlier?
        hit_color *= is_backface ? fBackfaceLightMult : 1;  // a.k.a. neon sign fix if you go to that club in rp_southside

        // brdf
        hit_color *= albedo;
        hit_color *= PI;  // cos-weighted sampling + lambert = chill; oren-nayar = black spots, not chill
        hit_color = max(hit_color, 0);
        
        gi_ao.rgb += hit_color * rcp_numsample;
    }
}

void PS_Accumulation(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 gi_ao_accum : SV_Target0, out float temporal_info : SV_Target1
)
{
#if YASSGI_USE_MOTION
    float2 uv_prev = uv + tex2D(sMotionVectorTex, uv).xy;

    [branch]
    if(!isInScreen(uv_prev))
    {
        gi_ao_accum = tex2D(samp_gi_ao, uv);
        temporal_info = 1;
        return;
    }
#else
    float2 uv_prev = uv;
#endif

    float hist_len_prev = tex2Dfetch(samp_temporal_1, uv_prev * YASSGI_GI_BUFFER_SIZE).x;

    float4 gi_ao_prev = tex2Dfetch(samp_gi_ao_accum_1, uv_prev * YASSGI_GI_BUFFER_SIZE);
    float4 gi_ao_curr = tex2Dfetch(samp_gi_ao, uv * YASSGI_GI_BUFFER_SIZE);
    
    float4 g_curr = tex2D(samp_g, uv);
    float4 g_prev = tex2D(samp_g_prev, uv_prev);

    // disocclusion. geometry deviation and hist_len(accum speed) update method from that REBLUR paper.
    // ^^^ to NV: not src!
    float z_delta = abs(g_curr.w - g_prev.w) / g_curr.w / max(fFrameTime, 1.0);  // depth and frame interval weighted
    float delta = z_delta * abs(dot(g_curr.xyz, normalize(uvToViewSpace(uv, g_curr.w))));  // geometry: compare deviation of plane instead of point
    // can't tweak the params for a smooth transition from valid to disoccluded, so just discard à la à-trous/svgf
    bool occluded = delta > fDisocclThres * 0.01;
    
    float hist_len_new = min(hist_len_prev * (!occluded) + 1, iMaxAccumFrames);
    float4 gi_ao_new = occluded ? gi_ao_curr : lerp(gi_ao_prev, gi_ao_curr, rcp(hist_len_new));

    // finalize

    // - here lies what used to be some luminance-based firefly suppression code -

    gi_ao_accum = gi_ao_new;
    temporal_info = hist_len_new;
}

// algo: Ray Tracing Gems Chapter 25
// orig src (missing the latter parts): https://alain.xyz/blog/ray-tracing-denoising
// src: https://www.yuque.com/isumiai/cg/efwkig#Sf1rG
// does this really have any use other than swapping buffer?
void PS_Firefly(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 gi_ao : SV_Target0, out float temporal_info : SV_Target1
)
{
    float4 gi_ao_curr = tex2D(samp_gi_ao_accum, uv);
    float hist_len = tex2D(samp_temporal, uv).x;

    float3 mean = gi_ao_curr.rgb;
    float3 mean_2 = gi_ao_curr.rgb * gi_ao_curr.rgb;
    for(int i = 0; i < 8; ++i)
    {
        float2 offset_px; sincos(i * 0.25 * PI, offset_px.y, offset_px.x); // <-sincos here!
        float2 uv_sample = uv + offset_px.xy * 3 / YASSGI_GI_BUFFER_SIZE;

        float3 gi_sample = tex2D(samp_gi_ao_accum, uv_sample).rgb;
        mean += gi_sample;
        mean_2 += gi_sample * gi_sample;
    }
    mean /= 9;
    mean_2 /= 9;
    
    gi_ao = float4(fireflyRejectionVariance(gi_ao_curr.rgb, mean_2 - mean * mean, mean), gi_ao_curr.a);
    temporal_info = hist_len;
}

// algo: Edge-avoiding À-trous https://jo.dreggn.org/home/2010_atrous.pdf
//     svgf (impl without variance) https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf
// referenced: https://www.shadertoy.com/view/tlXfRX
void PS_Blur1(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0
)
{
    color = atrous(samp_gi_ao_accum_1, uv, 1);
}

void PS_Blur2(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0
)
{
    color = atrous(samp_gi_ao_blur1, uv, 2);
}

void PS_Blur3(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0
)
{
    color = atrous(samp_gi_ao_blur2, uv, 4);
}

void PS_Blur4(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0
)
{
    color = atrous(samp_gi_ao_blur1, uv, 8);
}

void PS_Blur5(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0
)
{
    color = atrous(samp_gi_ao_blur2, uv, 16);
}

void PS_Display(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target)
{
    color = tex2D(ReShade::BackBuffer, uv);
    
    if(iViewMode == 0)  // None
    {
        float4 gi_ao = tex2D(samp_gi_ao_accum_1, uv);
        float4 albedo = tex2D(samp_color, uv);
        color.rgb = albedo.rgb;
        color.rgb += gi_ao.rgb * fIlStrength;
        color.rgb = color.rgb * exp2(-gi_ao.a * fAoStrength);
        color.rgb = saturate(mul(g_colorOutputMat, color.rgb));
    }
    else if(iViewMode == 1)  // Depth / Normal
    {
        float4 g = tex2D(samp_g, uv);
        if((iFrameCount / 300) % 2)  // Normal
        {
            color = -g.xyz * 0.5 + 0.5;  // for convention
        }
        else  // Depth
        {
            color = zToLinearDepth(g.w);
            if(color.r < fDepthRange.x)
                color = float3(color.r / fDepthRange.x, 0, 0);
            else if (isSky(g.w))
                color = float3(0.1, 0.5, 1.0);
        }
    }
    else if(iViewMode == 2)  // Fake Albedo
    {
        color.rgb = saturate(mul(g_colorOutputMat, fakeAlbedo(tex2D(samp_color, uv).rgb)));
    }
    else if(iViewMode == 3)  // GI
    {
        color = (iFrameCount / 300) % 2 ?
            exp2(-tex2D(samp_gi_ao, uv).a * fAoStrength) :
            mul(g_colorOutputMat, tex2D(samp_gi_ao, uv).rgb * fIlStrength);
    }
    else if(iViewMode == 4)  // GI Accum
    {
        color = (iFrameCount / 300) % 2 ?
            exp2(-tex2D(samp_gi_ao_accum_1, uv).a * fAoStrength) :
            mul(g_colorOutputMat, tex2D(samp_gi_ao_accum, uv).rgb * fIlStrength);
    }
    else if(iViewMode == 5)  // Accum speed
    {
        color = tex2D(samp_temporal_1, uv).x / iMaxAccumFrames;
    }
}

technique YASSGI_Old_Tracer <
    ui_tooltip = "!: This shader is slower with performance mode on dx9!"; >
{
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
        RenderTarget1 = tex_temporal;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Firefly;
        RenderTarget0 = tex_gi_ao_accum_1;
        RenderTarget1 = tex_temporal_1;
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
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Blur3;
        RenderTarget0 = tex_gi_ao_accum_1;
    }  // variable pass number
    // pass {
    //     VertexShader = PostProcessVS;
    //     PixelShader = PS_Blur4;
    //     RenderTarget0 = tex_gi_ao_accum_1;
    // }
    // pass {
    //     VertexShader = PostProcessVS;
    //     PixelShader = PS_Blur5;
    //     RenderTarget0 = tex_gi_ao_accum_1;
    // }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Display;
    }
}

}
