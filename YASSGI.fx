#include "ReShade.fxh"

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

texture BlueNoiseTex < source = "dh_rt_noise.png"; > { Width = NOISE_SIZE; Height = NOISE_SIZE; MipLevels = 1; Format = RGBA8; };
sampler BlueNoiseTexSampler                          { Texture = BlueNoiseTex; AddressU = REPEAT; AddressV = REPEAT; AddressW = REPEAT; };

texture ZTex	    { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F; MipLevels = YASSGI_MIP_LEVEL; };
sampler ZTexSampler { Texture = ZTex; };

texture NormalTex	     { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };
sampler NormalTexSampler { Texture = NormalTex; };

texture GITex	     { Width = YASSGI_BUFFER_WIDTH; Height = YASSGI_BUFFER_HEIGHT; Format = RGBA16F; };
sampler GITexSampler { Texture = GITex; };

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Uniform Varibales
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// Matrix src: https://www.shadertoy.com/view/WltSRB
static const float3x3 sRGBtoAP1 = float3x3
(
	0.613097, 0.339523, 0.047379,
	0.070194, 0.916354, 0.013452,
	0.020616, 0.109570, 0.869815
);
static const float3x3 AP1toSRGB = float3x3
(
     1.704859, -0.621715, -0.083299,
    -0.130078,  1.140734, -0.010560,
    -0.023964, -0.128975,  1.153013
);

uniform int   RANDOM < source = "random"; min = 0; max = 1 << 15; >;
uniform uint  FRAMECOUNT  < source = "framecount"; >;
uniform float FRAMETIME   < source = "frametime";  >;

uniform int iDebugView <
	ui_type = "combo";
    ui_label = "Debug View";
    ui_items = "None\0Initial Sample\0";
> = 0;

uniform int iInputColorSpace <
	ui_type = "combo";
    ui_label = "Input Color Space";
    ui_items = "sRGB\0";
	ui_category = "Color";
> = 0;

uniform int iVerticalFOV <
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Vertical FOV";
    ui_min = 60; ui_max = 140;
    ui_step = 1;
> = 90;

uniform float2 fDepthRange <
    ui_type = "slider";
    ui_category = "Depth";
    ui_label = "Depth Range";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
> = float2(0.0, 0.999);

uniform int iRayAmount <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Ray Amount";
    ui_min = 1; ui_max = 32;
    ui_step = 1;
> = 2;

uniform int iMaxRayStep <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Max Ray Step";
    ui_min = 1; ui_max = 40;
    ui_step = 1;
> = 20;

uniform float fBaseStride <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Base Stride";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
> = 2.0;

uniform float fStrideJitter <
    ui_type = "slider";
    ui_category = "Tracing";
    ui_label = "Stride Jitter";
    ui_min = 0.0; ui_max = 0.9;
    ui_step = 0.01;
> = 0.4;

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
> = 2.0;

uniform bool bBackfaceLighting <
    ui_label = "Backface Lighting";
    ui_category = "Tracing";
> = false;

uniform float3 fAmbientLight <
    ui_type = "color";
    ui_category = "Environment";
    ui_label = "Ambient Light";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.1;
> = 0.0;

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
bool isInScreen(in float3 world_pos, in VSOUT vsout)
{
    return isInScreen(worldPos2Coords(world_pos, vsout));
}
bool isNear(in float world_z)
{
    return world_z < depth2Z(fDepthRange.x);
}
bool isSky(in float world_z)
{
    return world_z > depth2Z(fDepthRange.y);
}

// https://www.ronja-tutorials.com/post/024-white-noise/#:~:text=For%20many%20effects%20we%20want%20random%20numbers%20to,other%20tutorials%2C%20for%20example%20perlin%20and%20voronoi%20noise.
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

float3 uniformHemisphere(float ep0, float ep1)
{
    float theta = acos(ep0);
    float phi = 2 * PI * ep1;
    return float3(cos(phi) * sin(theta), sin(phi) * sin(theta), ep0);
}

// Simple tracer
// you can control stride by providing a non-unit ray_dir
// hit: 0-hit nothing; 1-hit and valid; 2-hit and invalid;
void traceRay(in float3 ray_orig, in float3 ray_dir, in VSOUT vsout, out int hit, out float3 hit_pos, out float mip_level)
{
    hit = 0;
    hit_pos = 0.0;
    mip_level = YASSGI_MIP_LEVEL;

    if(isNear(ray_orig.z) || isSky(ray_orig.z))
        return;
    
    float3 stride = ray_dir * fBaseStride;
    float3 old_pos = ray_orig + stride * 0.008;
    for(int step = 0; step < iMaxRayStep; ++step)
    {
        mip_level = min(float(step) / fSpreadStep, YASSGI_MIP_LEVEL);
        float len_mult = pow(2, mip_level);
        float3 new_pos = old_pos + stride * len_mult;
        float2 new_coord = worldPos2Coords(new_pos, vsout);

        if(!isInScreen(new_coord) || isNear(new_pos.z) || isSky(new_pos.z))
            return;

        float new_z = tex2Dlod(ZTexSampler, float4(new_coord, int(mip_level), 0)).x;
            
        if(new_pos.z > new_z && new_pos.z < new_z + fZThickness * len_mult)
        {
            if(sign(new_z - ray_orig.z) != sign(ray_dir.z))
            {
                hit = 2;
                return;
            }

            hit = 1; hit_pos = new_pos;
            
            return;
        }
        
        old_pos = new_pos;
    }
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pixel Shaders
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void PS_InputBufferSetup(in VSOUT vsout, out float4 normal : SV_Target0, out float4 z : SV_Target1)
{
    // float3 offs = float3(BUFFER_PIXEL_SIZE,0);

    // float3 f 	     =       coords2WorldPos(vsout.texcoord.xy, vsout);
    // float3 gradx1 	 = - f + coords2WorldPos(vsout.texcoord.xy + offs.xz, vsout);
    // float3 gradx2 	 =   f - coords2WorldPos(vsout.texcoord.xy - offs.xz, vsout);
    // float3 grady1 	 = - f + coords2WorldPos(vsout.texcoord.xy + offs.zy, vsout);
    // float3 grady2 	 =   f - coords2WorldPos(vsout.texcoord.xy - offs.zy, vsout);

    // gradx1 = lerp(gradx1, gradx2, abs(gradx1.z) > abs(gradx2.z));
    // grady1 = lerp(grady1, grady2, abs(grady1.z) > abs(grady2.z));

    // normal = float4(normalize(cross(grady1, gradx1)) * 0.5 + 0.5, 1.0);
    normal = float4(normal_from_depth(vsout.texcoord, vsout), 1.0);
    z = depth2Z(ReShade::GetLinearizedDepth(vsout.texcoord));
}

void PS_Trace(in VSOUT vsout, out float4 color: SV_Target)
{
    float3 ray_orig = coords2WorldPos(vsout.texcoord, vsout);
    float3 normal = tex2D(NormalTexSampler, vsout.texcoord).xyz;

    float max_dist = fBaseStride * (1 << (iMaxRayStep / fSpreadStep) - 1) * fSpreadStep;
    
    [loop]
    for(int r = 0; r < iRayAmount; ++r)
    {
        float3 rand3 = rand4dTo3d(float4(vsout.texcoord, RANDOM / PI, (r + FRAMECOUNT) / SQRT2));

        float3 ray_dir = normalize(normal + uniformHemisphere(rand3.x, rand3.y)) * (1 + (rand3.z - 1) * fStrideJitter);

        // tracing
        int hit = 0;
        float3 ray_end;
        float mip_level;
        traceRay(ray_orig, ray_dir, vsout, hit, ray_end, mip_level);

        float3 hit_color = 0.0;

        if(hit == 2)
            continue;
        else if(hit == 0)
            hit_color = fAmbientLight;
        else
        {
            // normal check
            float2 coord_end = worldPos2Coords(ray_end, vsout);
            float3 normal_end = tex2Dlod(NormalTexSampler, float4(coord_end, 0, 0)).rgb;
            float normal_dot = dot(normal_end, ray_dir);
            if(!bBackfaceLighting && normal_dot > -0.02)
                continue;

            hit_color = tex2Dlod(ReShade::BackBuffer, float4(coord_end, int(mip_level), 0)).rgb * abs(normal_dot);
        }
        
        // color mixing
        hit_color = mul(hit_color, sRGBtoAP1);
        color += hit_color * saturate(1 - distance(ray_orig, ray_end) / max_dist);
    }

    color /= iRayAmount;
    color.w = 1.0;
}

void PS_Display(in VSOUT vsout, out float4 color: SV_Target)
{
    if(iDebugView == 0)
        color = tex2D(ReShade::BackBuffer, vsout.texcoord);
    else if(iDebugView == 1)
        color = mul(tex2D(GITexSampler, vsout.texcoord).rgb, AP1toSRGB);
}

technique YASSGI{
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_InputBufferSetup;
        RenderTarget0 = NormalTex;
        RenderTarget1 = ZTex;
    }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_Trace;
        RenderTarget = GITex;

        ClearRenderTargets = true;
    }
    pass {
        VertexShader = VS_YASSGI;
        PixelShader = PS_Display;
    }
}
}