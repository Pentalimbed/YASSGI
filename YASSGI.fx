/*
    z: raw z
    depth: linearized z
    z direction: + going farther, - coming closer
    normal: pointing outwards
*/

#include "ReShade.fxh"

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

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Buffers
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// normal and z
texture tex_g  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_g {Texture = tex_g;};

// normal and z (previous frame)
texture tex_g_prev  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_g_prev {Texture = tex_g_prev;};

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


float zToLinearDepth(float z) {return (z - 1) * rcp(RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);}
float linearDepthToZ(float depth) {return depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE + 1;}

float getLinearDepth(float2 uv) {return ReShade::GetLinearizedDepth(uv);}
float getZ(float2 uv) {return linearDepthToZ(getLinearDepth(uv));}

float3 uvToViewSpace(float2 uv, float z){return float3(uv * 2 - 1, 1) * z;}
float3 uvToViewSpace(float2 uv){return uvToViewSpace(uv, getZ(uv));}
float2 viewSpaceToUv(float3 pos){return (pos.xy / pos.z + 1) / 2;}

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

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Pixel Shaders
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void PS_SavePrev(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 g_prev : SV_Target0)
{
    g_prev = tex2D(samp_g, uv);
}

void PS_InputSetup(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 g : SV_Target0)
{
    float z = getZ(uv);
    float3 normal = getViewNormalAccurate(uv);

    if(zToLinearDepth(z.x) < fDepthRange.x)
    {
        z.x = ((z.x - 1) * fWeapDepthMult) + 1;
        normal = normalize(normal * float3(1, 1, rcp(fWeapDepthMult)));
    }

    g = float4(normal, z);
}

void PS_Display(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target)
{
    [branch]
    if(iViewMode == 0)  // None
    {
        color.rgb = tex2D(ReShade::BackBuffer, uv).rgb;
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
    // else if(iViewMode == 2)  // GI
    // {
    //     color = (iFrameCount / 300) % 2 ? linearToSrgb(1 - tex2D(samp_gi_ao, uv).w) : linearToSrgb3(tex2D(samp_gi_ao, uv).xyz * fIlStrength);
    // }
    // else if(iViewMode == 3)  // GI Accum
    // {
    //     color = (iFrameCount / 300) % 2 ? linearToSrgb(1 - tex2D(samp_gi_ao_accum, uv).w) : linearToSrgb3(tex2D(samp_gi_ao_accum, uv).xyz * fIlStrength);
    // }
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
        RenderTarget0 = tex_g;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Display;
    }
}

}