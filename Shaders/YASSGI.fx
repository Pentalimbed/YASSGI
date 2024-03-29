/*  REFERENCES & CREDITS
    
    Redistribution of source material are maked with <src>,
    with a copy of its license appended if it demands so.

    All other code shall be considered licensed under UNLICENSE,
    either as (re)implementation of their source materials,
    or as the author's original work.

    Free blue noise textures. Christoph Peters.
        url:    http://momentsingraphics.de/BlueNoise.html
        credit: blue noise texture. <src CC0>
    RGB COLOURSPACE TRANSFORMATION MATRIX. Colour Developers.
        url:    https://www.colour-science.org/apps/
        credit: ACEScg <-> sRGB conversion matrices
    Accurate Normal Reconstruction from Depth Buffer. atyuwen.
        url:    https://atyuwen.github.io/posts/normal-reconstruction
        credit: view normal reconstruction algorithm
    The Halton Sampler, Physically Based Rendering. Matt Pharr, Wenzel Jakob, and Greg Humphreys.
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
                thin object heuristics <src>
        license:
            MIT License

            Copyright (C) 2016-2021, Intel Corporation 

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
    Horizon-Based Indirect Lighting. Benoît "Patapom" Mayaux.
        url:    https://github.com/Patapom/GodComplex/blob/master/Tests/TestHBIL/2018%20Mayaux%20-%20Horizon-Based%20Indirect%20Lighting%20(HBIL).pdf
        credit: interleaved sampling
                calculation of horizon based indirect light
    Legit Engine. Alexander "Raikiri" Sannikov.
        url:    https://github.com/Raikiri/LegitEngine
                multiple of their youtube videos and comments
        credit: motivation
                computeHorizonContribution function <src>
        license:
            Copyright 2020 Alexander Sannikov

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
    Screen Space Indirect Lighting with Visibility Bitmask. Olivier Therrien, Yannick Levesque, and Guillaume Gilet.
        url:    https://arxiv.org/abs/2301.11376
        credit: bitmask ao/il
    ReBLUR: A Hierarchical Recurrent Denoiser, Ray Tracing Gems II. Dmitry Zhdan, NVIDIA.
        url:    https://link.springer.com/book/10.1007/978-1-4842-7185-8
        credit: disocclusion by geometry
    Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination. 
     Christoph Schied, Anton Kaplanyan, Chris Wyman, Anjul Patney, Chakravarty R. Alla Chaitanya, John Burgess,
     Shiqiu Liu, Carsten Dachsbacher, Aaron Lefohn, Marco Salvi
        url:    https://research.nvidia.com/sites/default/files/pubs/2017-07_Spatiotemporal-Variance-Guided-Filtering%3A//svgf_preprint.pdf
        credit: spatial filter weight calculation
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
    pos_w: world
    angle: radian
*/
/*  TODO

    - pending  o finished  x canceled  * shelved  ? idk

    o il
    * bent normal
    o thickness heuristic
    o alternative bitmask impl (?)
    - hi-z buffer w/ cone tracing (?)
    o remove subtle grid like pattern
    * ibl
    x adaptive light src thres (?)
    - simple geometric light src
    - dx9 compatibility
    o edge firefly suppression
*/

#include "ReShade.fxh"
#include "ShaderFastMathLib.h"

namespace YASSGI
{

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Constants
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#if __RENDERER__ == 0x9000
#   define DX9 1
#else
#   define DX9 0
#endif

#define PI      3.14159265358979323846264
#define HALF_PI 1.5707963267948966
#define RCP_PI  0.3183098861837067

#define MAX_UINT_F 4294967295.0

#define EPS 1e-6

#define BUFFER_SIZE uint2(BUFFER_WIDTH, BUFFER_HEIGHT)
#define PIXEL_UV_SIZE float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT)

// bits of int
#define BITMASK_SIZE 32

#define NOISE_SIZE 256

#define INTERLEAVED_SIZE_PX 4
#define MAX_MIP 8


#ifndef YASSGI_PREBLUR_SCALE
#   define YASSGI_PREBLUR_SCALE 0.25
#endif

#ifndef YASSGI_DISABLE_IL
#   define YASSGI_DISABLE_IL 0
#endif

#if DX9 == 0
#   ifndef YASSGI_USE_BITMASK
#       define YASSGI_USE_BITMASK 1
#   endif
#else
#   define YASSGI_USE_BITMASK 0
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

uniform int   iFrameCount < source = "framecount"; >;

// <---- UI ---->

// uniform float fDebug <
//     ui_type = "slider";
//     ui_min = 0; ui_max = 1;
// > = 1;

uniform int iViewMode <
	ui_type = "combo";
    ui_label = "View Mode";
    ui_items = "YASSGI\0Depth\0Normal\0AO\0IL\0Accumulated Frames\0";
> = 0;

uniform int iConfigGuide <
	ui_text = "The default setting is where the author finds the equilibrium between visual and performance, "
              "that is, where a higher setting does not significantly affect final presentation.\n\n"
              "Here is what to tweak if you find your card burning (ranking from least to most visual impact):\n"
              "> [Spread Exponent] to max;\n"
              "> {YASSGI_PREBLUR_SCALE} to 0.125 (1/8, idk how low you can go without ruining everything);\n"
              "> [Slices] to 1, noisier. You can compensate by increasing Max Accumulated Frames;\n"
              "> Turn up [Base Stride] to skip minute details;\n"
              "> Turn down [Sample Distance] if you don't mind losing large scale GI;\n"
              "> {YASSGI_USE_BITMASK} to 1 is a bit faster if you want IL.\n"
              "> {YASSGI_DISABLE_IL} to 1 if you don't care about indirect lights.";
	ui_category = "Performance Guide";
	ui_category_closed = true;
	ui_label = " ";
	ui_type = "radio";
>;

uniform float fFov <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "FOV";
    ui_min = 60; ui_max = 130.0;
    ui_step = 0.1;
> = 90.0;

uniform float2 fZRange <
    ui_type = "slider";
    ui_category = "Input";
    ui_label = "Weapon/Sky Z Range";
    ui_min = 0.0; ui_max = RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;
    ui_step = 0.1;
> = float2(0.1, RESHADE_DEPTH_LINEARIZATION_FAR_PLANE - 0.1);

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

uniform uint iDirCount <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Slices";
    ui_tooltip = "The more slices, the less noisy of the raw ouput.";
    ui_min = 1; ui_max = 4;
> = 2;  

uniform float fBaseStridePx <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Base Stride (px)";
    ui_tooltip = "The greater the stride, the more it skips over small objects.";
    ui_min = 1; ui_max = 64;
    ui_step = 1;
> = 16;

uniform float fSpreadExp <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Spread Exponent";
    ui_tooltip = "Makes further samples less accurate but saves performance.";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.01;
> = 2.5;

uniform float fStrideJitter <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Stride Jitter";
    ui_tooltip = "A reasonable amount of jitter helps remove the unnatural shape/pattern of the AO.";
    ui_min = 0; ui_max = 1;
    ui_step = 0.01;
> = 0.66;

uniform float fMaxSampleDistPx <
    ui_type = "slider";
    ui_category = "Sampling";
    ui_label = "Sample Distance (px)";
    ui_tooltip = "How far in screen should an object contribute to its surroundings.";
    ui_min = 2; ui_max = BUFFER_WIDTH;
    ui_step = 1;
> = BUFFER_WIDTH * 0.2;

// uniform float fLodRangePx <
//     ui_type = "slider";
//     ui_category = "Sampling";
//     ui_label = "LOD Range (px)";
//     ui_min = 2; ui_max = 64;
//     ui_step = 1;
// > = 48;
static const float fLodRangePx = 32;

#if YASSGI_USE_BITMASK == 0
uniform float fFxRange <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Effect Reach";
    ui_min = 0; ui_max = 1.0;
    ui_step = 0.001;
> = 0.5;

uniform float fFxFalloff <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Effect Falloff";
    ui_min = 0; ui_max = 1.0;
    ui_step = 0.001;
> = 0.6;

uniform float fThinOccluderCompensation <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Thin Obj Compensation";
    ui_min = 0; ui_max = 0.7;
    ui_step = 0.01;
> = 0.7;
#else
uniform float fThickness <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Thickness";
    ui_min = 0.0; ui_max = 10.0;
    ui_step = 0.1;
> = 5;

uniform float fThicknessZScale <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Thickness Z Scaling";
    ui_tooltip = "Makes further object thicker.";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.02;

uniform float fBackfaceLighting <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Backface Lighting";
    ui_tooltip = "Allow an object's backside to illuminate others. A touch of this simulates subsurface scattering of grass and foilage.";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.1;
#endif  // YASSGI_USE_BITMASK == 0

#if YASSGI_DISABLE_IL == 0
uniform float fLightSrcThres <
    ui_type = "slider";
    ui_label = "Light Source Threshold";
    ui_tooltip = "Only pixels brighter than this are considered light-emitting.";
    ui_category = "Visual";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.0;

uniform float fAlbedoSatPower <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Albedo Saturation Power";
    ui_tooltip = "Since ReShade has no way of knowing the true albedo of a surface separate fq2rom lighting,\n"
        "any shader has to guess. A value of 0.0 tells the shader that everything is monochrome, and its\n"
        "hue is the result of lighting. Greater value yields more saturated output on colored surfaces.\n";
    ui_min = 0.0; ui_max = 8.0;
    ui_step = 0.01;
> = 1.0;

uniform float fAlbedoNorm <
    ui_type = "slider";
    ui_category = "Visual";
    ui_label = "Albedo Normalization";
    ui_tooltip = "Since ReShade has no way of knowing the true albedo of a surface separate from lighting,\n"
        "any shader has to guess. A value of 0.0 tells the shader that there is no lighting in the scene,\n"
        "so dark surfaces are actually black. 1.0 says that all surfaces are in fact colored brightly, and\n"
        "variation in brightness is the result of illumination, rather than the texture pattern itself.";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.8;
#endif  // YASSGI_DISABLE_IL == 0

#if YASSGI_DISABLE_FILTER == 0
uniform int iMaxAccumFrames <
    ui_type = "slider";
    ui_category = "Filter";
    ui_label = "Max Accumulated Frames";
    ui_min = 1; ui_max = 64;
    ui_step = 1;
> = 8;

uniform float fDisocclThres <
    ui_type = "slider";
    ui_category = "Filter";
    ui_label = "Disocclusion Threshold";
    ui_tooltip = "Lower this to suppress ghosting.";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
> = 0.3;

uniform float fEdgeThres <
    ui_type = "slider";
    ui_category = "Filter";
    ui_label = "Edge Detection Threshold";
    ui_tooltip = "Lower this to suppress firefly on edges.";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
> = 0.1;

// uniform float fBlurRadius <
//     ui_type = "slider";
//     ui_category = "Filter";
//     ui_label = "Blur Radius";
//     ui_min = 0.0; ui_max = 5.0;
//     ui_step = 0.01;
// > = 2.0;
static const float fBlurRadius = 2.0;

#endif  // YASSGI_DISABLE_FILTER == 0

uniform float fAoStrength <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "AO";
    ui_tooltip = "Negative value for non-physical-accurate exponential mixing.";
    ui_min = -2.0; ui_max = 1.0;
    ui_step = 0.01;
> = 1.0;

#if YASSGI_DISABLE_IL == 0
uniform float fIlStrength <
    ui_type = "slider";
    ui_category = "Mixing";
    ui_label = "IL";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.01;
> = 2;

uniform int iMixOrder <
	ui_type = "combo";
    ui_label = "Mixing Order";
    ui_items = "IL First\0AO First\0";
> = 1;
#endif

uniform int iMacroGuide <
	ui_text = "YASSGI_DISABLE_IL: set to 1 disables indirect light calculation for better performance.\n\n"
              "YASSGI_PREBLUR_SCALE: determines how detailed lighting calculation is.\n\n"
              "YASSGI_USE_BITMASK: switches between two gi algorithm of different flavor.\n\n"
              "\t0 uses GTAO/HBIL which is industry standard, but does not handle thickness well and generates more halos.\n\n"
              "\t1 uses Bitmask IL which has intrinsic thickness support. This is the author's pick. (Unavailable to DX9 games)";
	ui_category = "Preprocessor Guide";
	ui_category_closed = true;
	ui_label = " ";
	ui_type = "radio";
>;

}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Buffers
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// motion vectors via other fx
texture texMotionVectors { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
sampler sMotionVectorTex { Texture = texMotionVectors; };

namespace YASSGI
{
texture tex_blue <source = "YASSGI_bleu.png";> {Width = NOISE_SIZE; Height = NOISE_SIZE; Format = RGBA8;};
sampler samp_blue                              {Texture = tex_blue;};

texture tex_color  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_color {Texture = tex_color;};

texture tex_geo  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA32F;};
sampler samp_geo {Texture = tex_geo;};

// downscaled normal (RGB) raw_z (A)
// tbf orig paper uses interleaved buffers (not blurred) but no sampler spamming for dx9.
//                                          (or perhaps a separate shader(s) for those?)
// Edit: I do think we need those for maximal performance.
texture tex_blur_geo  {Width = BUFFER_WIDTH * YASSGI_PREBLUR_SCALE; Height = BUFFER_HEIGHT * YASSGI_PREBLUR_SCALE; Format = RGBA32F; MipLevels = MAX_MIP;};
sampler samp_blur_geo {Texture = tex_blur_geo;};

texture tex_blur_radiance  {Width = BUFFER_WIDTH * YASSGI_PREBLUR_SCALE; Height = BUFFER_HEIGHT * YASSGI_PREBLUR_SCALE; Format = RGBA16F; MipLevels = MAX_MIP;};
sampler samp_blur_radiance {Texture = tex_blur_radiance;};

// texture tex_bent_normal  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
// sampler samp_bent_normal {Texture = tex_bent_normal;};

texture tex_il_ao  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = MAX_MIP;};
sampler samp_il_ao {Texture = tex_il_ao;};

#if YASSGI_DISABLE_FILTER == 0
texture tex_il_ao_ac  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F;};
sampler samp_il_ao_ac {Texture = tex_il_ao_ac;};

texture tex_il_ao_ac_prev  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; MipLevels = 1;};
sampler samp_il_ao_ac_prev {Texture = tex_il_ao_ac_prev;};

texture tex_temporal  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = R16F;};
sampler samp_temporal {Texture = tex_temporal;};

// history len (R), raw z (G)
texture tex_temporal_z_prev  {Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG32F;};
sampler samp_temporal_z_prev {Texture = tex_temporal_z_prev;};
#endif

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Functions
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// <---- Input ---->

float zToLinearDepth(float z) {return z * rcp(RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);}
float linearDepthToZ(float depth) {return depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;}

float getLinearDepth(float2 uv) {return ReShade::GetLinearizedDepth(uv);}
float getZ(float2 uv) {return linearDepthToZ(getLinearDepth(uv));}

float getHalfFrustumSize(float z) {return z * tan(radians(fFov) * 0.5);}
float3 uvzToView(float2 uv, float z){ return float3((uv * 2 - 1) * getHalfFrustumSize(z) * float2(BUFFER_WIDTH * BUFFER_RCP_HEIGHT, 1), z);}
float2 viewToUv(float3 pos){return (pos.xy / getHalfFrustumSize(pos.z) * float2(BUFFER_HEIGHT * BUFFER_RCP_WIDTH, 1) + 1) * 0.5;}

bool isNear(float z){return z < 0;}
bool isFar(float z){return z > RESHADE_DEPTH_LINEARIZATION_FAR_PLANE;}
bool isWeapon(float z){return z < fZRange.x;}
bool isSky(float z){return z > fZRange.y;}

float3 getViewNormalAccurate(float2 uv)
{
    float3 view_pos = uvzToView(uv, getZ(uv));

    float2 off1 = float2(BUFFER_RCP_WIDTH, 0);
    float2 off2 = float2(0, BUFFER_RCP_HEIGHT);

    float3 view_pos_l = uvzToView(uv - off1, getZ(uv - off1));
    float3 view_pos_r = uvzToView(uv + off1, getZ(uv + off1));
    float3 view_pos_d = uvzToView(uv - off2, getZ(uv - off2));
    float3 view_pos_u = uvzToView(uv + off2, getZ(uv + off2));

    float3 l = view_pos - view_pos_l;
    float3 r = view_pos_r - view_pos;
    float3 d = view_pos - view_pos_d;
    float3 u = view_pos_u - view_pos;

    // get depth values at 1 & 2 px offset along both axis
    float4 H = float4(
        view_pos_l.z,
        view_pos_r.z,
        getZ(uv - 2 * off1),
        getZ(uv + 2 * off1)
    );
    float4 V = float4(
        view_pos_d.z,
        view_pos_u.z,
        getZ(uv - 2 * off2),
        getZ(uv + 2 * off2)
    );
    
    // current pixel's depth difference
    float2 he = abs((2 * H.xy - H.zw) - view_pos.z);
    float2 ve = abs((2 * V.xy - V.zw) - view_pos.z);

    // pick horizontal and vertical diff with smallest depth difference from slopes
    float3 h_deriv = he.x < he.y ? l : r;
    float3 v_deriv = ve.x < ve.y ? d : u;

    return -normalize(cross(h_deriv, v_deriv));
}

#if YASSGI_DISABLE_IL == 0
float3 fakeAlbedo(float3 color)
{
    float3 albedo = pow(max(0, color), fAlbedoSatPower * length(color));  // length(color) suppress saturation of darker colors
    albedo = saturate(lerp(albedo, normalize(albedo), fAlbedoNorm));
    return albedo;
}
#endif

// <---- Sampling ---->

float random3dTo1d(float3 value, float a, float3 b)
{			
	float3 small_value = sin(value);
	float  random = dot(small_value, b);
	random = frac(sin(random) * a);
	return random;
}
float2 random3dTo2d(float3 value){
	return float2(
		random3dTo1d(value, 14375.5964, float3(15.637, 76.243, 37.168)),
		random3dTo1d(value, 14684.6034, float3(45.366, 23.168, 65.918))
	);
}

#if DX9 == 0
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
#endif

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
float ilIntegral(float nx, float ny, float sin_prev, float sin_new)
{
    float delta_angle = asinFast4(sin_prev) - asinFast4(sin_new);
    float cos_prev = sqrt(1 - sin_prev * sin_prev);
    float cos_new = sqrt(1 - sin_new * sin_new);
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

void PS_Setup(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target0, out float4 geo : SV_Target1
)
{
    const uint2 px_coord = uv * BUFFER_SIZE;

    color = mul(g_colorInputMat, tex2Dfetch(ReShade::BackBuffer, px_coord).rgb);
    color.a = 1;

    float3 normal = getViewNormalAccurate(uv);
    
    float z = getZ(uv);

    if(isWeapon(z))
    {
        z *= fWeapDepthMult;
        normal = normalize(float3(normal.xy, normal.z / fWeapDepthMult));
    }

    geo = float4(normal, z);
}

void PS_PreBlur(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 blur_geo : SV_Target0, out float4 blur_radiance : SV_Target1
)
{  
    const float4 geo = tex2D(samp_geo, uv);
    float3 sum_normal = geo.xyz;
    float sum_z = geo.w;
    float3 sum_color = tex2D(samp_color, uv).rgb;
    float sum_w = 1;
    [unroll]
    for(uint i = 0; i < 8; ++i)
    {
        float2 offset_px; sincos(i * 0.25 * PI, offset_px.y, offset_px.x);  // <-sincos here!
        const float2 offset_uv = offset_px / YASSGI_PREBLUR_SCALE * 0.5 * PIXEL_UV_SIZE;
        const float2 uv_sample = uv + offset_uv;

        const float4 geo_sample = tex2Dlod(samp_geo, float4(uv_sample, 0, 0));

        const float w = exp(-0.66 * length(offset_px)) * isInScreen(uv_sample);
        sum_normal += geo_sample.xyz * w;
        sum_z += geo_sample.w * w;
        sum_color += tex2Dlod(samp_color, float4(uv_sample, 0, 0)).rgb * w;
        sum_w += w;
    }
    blur_geo = float4(sum_normal, sum_z) / sum_w;
    blur_radiance = float4(sum_color / sum_w, 1);
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

    const float3 color = tex2Dfetch(samp_color, px_coord).rgb;
#if YASSGI_DISABLE_IL == 0
    const float3 albedo = fakeAlbedo(color);
#endif
    const float4 geo = tex2Dfetch(samp_geo, px_coord);
    const float3 normal_v = geo.xyz;
    const float z = geo.w;
    const float3 pos_v = uvzToView(uv, z) * 0.99995;  // closer to the screen bc we're using blurred geometry
    const float3 dir_v_view = -normalize(pos_v);

    [branch]
    if(isWeapon(pos_v.z) || isSky(pos_v.z))  // leave sky alone
        return;

#if DX9
    const float2 rand = random3dTo2d(float3(uv, iFrameCount * RCP_PI));
    const float4 blue = tex2Dfetch(samp_blue, uint2((uv + rand.xy) * BUFFER_SIZE) % NOISE_SIZE);
#else
    const uint3 rand = pcg3d(uint3(px_coord, iFrameCount));
    const float4 blue = tex2Dfetch(samp_blue, (px_coord + rand.xy) % NOISE_SIZE);
#endif
    // // interleaved sampling (enough for ao)
    // uint2 px_coord_shifted = px_coord + blue.xy * 8;  // de-grid
    // const uint px_idx = (dot(px_coord_shifted % INTERLEAVED_SIZE_PX, uint2(1, INTERLEAVED_SIZE_PX)) + (iFrameCount % 16)) % 16;
    // const float2 distrib = hammersley(px_idx, INTERLEAVED_SIZE_PX * INTERLEAVED_SIZE_PX);
    // // ^^^ x for angle, y for stride

    // some consts
    const float rcp_dir_count = 1.0 / iDirCount;
    const float angle_pizza = PI * rcp_dir_count;
    const float stride_px = max(1, fBaseStridePx * (0.5 + blue.x * fStrideJitter));
    const float log2_stride_px = log2(stride_px);
#if YASSGI_USE_BITMASK == 0
    const float falloff_start_px = fFxRange * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * 0.01 * (1 - fFxFalloff);
    const float falloff_mul = -rcp(fFxRange * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * 0.01);
    const float falloff_add = falloff_start_px / fFxFalloff + 1;
#endif

    // per slice
    float4 sum = 0;  // visibility
    [loop]  // unroll?
    for(uint idx_dir = 0; idx_dir < iDirCount; ++idx_dir)
    {
        // slice directions
        const float angle_slice = (idx_dir + blue.y) * angle_pizza;
        float2 dir_px_slice; sincos(angle_slice, dir_px_slice.y, dir_px_slice.x);  // <-sincos here!
        const float2 dir_uv_slice = normalize(dir_px_slice * float2(BUFFER_WIDTH * BUFFER_RCP_HEIGHT, 1));

        const float3 dir_v_slice_screen = float3(dir_px_slice.x, dir_px_slice.y, 0);
        const float3 dir_v_slice_local = projectToPlane(dir_v_slice_screen, dir_v_view);
        const float3 normal_slice = normalize(cross(dir_v_slice_local, dir_v_view));
        const float3 normal_proj = projectToPlane(normal_v, normal_slice);  // not unit vector
        const float3 normal_proj_normalized = normalize(normal_proj);

#if YASSGI_USE_BITMASK == 0      
        const float sign_n = sign(dot(dir_v_slice_local, normal_proj));
        const float cos_n = saturate(dot(normal_proj_normalized, dir_v_view));
        const float n = sign_n * acosFast4(cos_n);
        const float sin_n = sin(n);
        
        float h0, h1;
#else
        uint bitmask = 0;
#endif

        // Algorithm 1 in the GTAO paper
        // 0 for -dir_px_slice, 1 for +dir_px_slice
        const float2 dists = intersectBox(px_coord, dir_px_slice, float4(0, 0, BUFFER_SIZE));
        [loop]
        for(uint side = 0; side <= 1; ++side)
        {
            const int side_sign = side * 2 - 1;
            const float max_dist = min(fMaxSampleDistPx, side ? dists.y : abs(dists.x));
            
#if YASSGI_USE_BITMASK == 0
            const float min_hor_cos = cos(n + side_sign * HALF_PI);
            float hor_cos = min_hor_cos;
            float3 radiance_sample = 0;
#endif

            // marching
            uint step = 0;
            float dist_px = stride_px;
            [loop]
            while(dist_px < max_dist && step < 64)  // preventing infinite loop when you tweak params in ReShade
            {
                const float2 offset_px = dir_px_slice * dist_px;
                const float2 px_coord_sample = px_coord + side_sign * offset_px;
                const float2 uv_sample = (px_coord_sample + 0.5) * PIXEL_UV_SIZE;

                const uint mip_level = clamp(log2(dist_px) - log2_stride_px, 0, MAX_MIP);
                const float4 geo_sample = tex2Dlod(samp_blur_geo, float4(uv_sample, mip_level, 0));
                const float3 pos_v_sample = uvzToView(uv_sample, geo_sample.w);
                
#if YASSGI_USE_BITMASK == 0
                [branch]
                if(!isWeapon(pos_v_sample.z))
                {
                    const float3 offset_v = pos_v_sample - pos_v;

                    // thin obj heuristics
                    const float falloff = length(offset_v * float3(1, 1, 1 + fThinOccluderCompensation));
                    const float weight = saturate(falloff * falloff_mul + falloff_add);

                    const float3 dir_v_hor = normalize(offset_v);
                    float hor_cos_sample = dot(dir_v_hor, dir_v_view);
                    hor_cos_sample = lerp(min_hor_cos, hor_cos_sample, weight);

#   if YASSGI_DISABLE_IL == 1
                    hor_cos = max(hor_cos, hor_cos_sample);
#else
                    [branch]
                    if(hor_cos_sample > hor_cos)
                    {
                        float3 radiance_sample_new = tex2Dlod(samp_blur_radiance, float4(uv_sample, mip_level, 0)).rgb;
                        radiance_sample_new = luminance(radiance_sample_new) > fLightSrcThres ? radiance_sample_new : 0;
                        // radiance_sample_new *= ilIntegral(cos_n, sin_n, hor_cos, hor_cos_sample);
                        radiance_sample_new *= computeHorizonContribution(dir_v_view, dir_v_slice_local, normal_v, acosFast4(hor_cos_sample), acosFast4(hor_cos)); 

                        // depth filtering. HBIL pp.38
                        float t = smoothstep(0, 1, dot(geo_sample.xyz, offset_v));
                        // float t = dot(geo_sample.xyz, offset_v) > -EPS;
                        // float t = 1;
                        radiance_sample = lerp(radiance_sample, radiance_sample_new, t);

                        sum.rgb += max(0, radiance_sample * albedo);

                        hor_cos = hor_cos_sample;
                    }
#endif  // YASSGI_DISABLE_IL == 1

                }

#else  // bitmask enabled

                const float3 pos_v_back = pos_v_sample + normalize(pos_v_sample) * fThickness 
                    * lerp(1, 1 + fThicknessZScale * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE, pos_v_sample.z / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);  // bc far plane changes between world space
                const float3 dir_v_front = normalize(pos_v_sample - pos_v);
                const float3 dir_v_back = normalize(pos_v_back - pos_v);
                const float2 angles = float2(acosFast4(dot(dir_v_front, normal_v)), acosFast4(dot(dir_v_back, normal_v)));
                float2 angles_minmax = float2(min(angles.x, angles.y), max(angles.x, angles.y));
                angles_minmax = clamp(angles_minmax, -HALF_PI, HALF_PI);
                const uint a = floor((angles_minmax.x + HALF_PI) * RCP_PI * BITMASK_SIZE);
                const uint b = ceil((angles_minmax.y - angles_minmax.x) * RCP_PI * BITMASK_SIZE);  // in orig paper...an additional half pi, wtf?
                const uint covered_bits = ((1 << b) - 1) << a;
#   if YASSGI_DISABLE_IL == 0
                const uint valid_bits = covered_bits & ~bitmask;
                [branch]
                if(valid_bits)
                {
                    float3 radiance_sample = tex2Dlod(samp_blur_radiance, float4(uv_sample, mip_level, 0)).rgb;
                    radiance_sample *= countbits(valid_bits) / (float)BITMASK_SIZE;
                    radiance_sample *= dot(normal_v, dir_v_front);
                    radiance_sample *= dot(geo_sample.xyz, dir_v_front) < -EPS ? 1.0 : fBackfaceLighting;
                    sum.rgb += max(0, radiance_sample * albedo);
                }
#   endif  // YASSGI_DISABLE_IL == 0
                bitmask = bitmask | covered_bits;
#endif  // YASSGI_USE_BITMASK == 0

                dist_px += stride_px * exp2(step * fSpreadExp);
                ++step;  // 2 same stride at the start. more precise perhaps (?)
            }

#if YASSGI_USE_BITMASK == 0
            const float h = acosFast4(hor_cos);
            const float angle_hor = n + clamp(side_sign * h - n, -HALF_PI, HALF_PI);  // XeGTAO suggested skipping clamping. Hmmm...
            sum.w += saturate(length(normal_proj) * 0.25 * (cos_n + 2 * angle_hor * sin_n - cos(2 * angle_hor - n)));
            
            side ? h1 : h0 = h;
#endif
        }
#if YASSGI_USE_BITMASK == 1
        sum.w += countbits(bitmask) / (float)BITMASK_SIZE;
#endif
    }

    sum /= iDirCount;
#if YASSGI_USE_BITMASK == 0
    sum.w = 1 - sum.w;
#else
    sum *= 2;  // I must have done something wrong but I dare not think of it.
#endif
    il_ao.w = clamp(sum.w, 0.05, 1.0);
    il_ao.rgb = sum.rgb;
    // il_ao = temp;
}

void PS_Accum(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 il_ao_accum : SV_Target0, out float temporal : SV_Target1)
{
    const uint2 px_coord = uv * BUFFER_SIZE;
    
    const float2 uv_prev = uv + tex2D(sMotionVectorTex, uv).xy;
    [branch]
    if(!isInScreen(uv_prev.xy))
    {
        il_ao_accum = tex2Dlod(samp_il_ao, float4(uv, 1, 0));
        temporal = 1;
        return;
    }

    const float4 temporal_prev = tex2D(samp_temporal_z_prev, uv_prev.xy);
    const float hist_len_prev = temporal_prev.x;

    const float4 z_gather = tex2DgatherA(samp_geo, uv);
    const float z_min = min(min(z_gather.x, z_gather.y), min(z_gather.z, z_gather.w));
    const float z_max = max(max(z_gather.x, z_gather.y), max(z_gather.z, z_gather.w));
    bool on_edge = z_max - z_min > fEdgeThres * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * 0.001;

    const float4 geo_curr = tex2Dfetch(samp_geo, px_coord);
    const float z_curr = geo_curr.w;
    const float z_prev = temporal_prev.y;

    [branch]
    if(isSky(z_curr))
    {
        il_ao_accum = 0;
        temporal = 1;
        return;
    }

    
    const float3 normal_curr = geo_curr.xyz;

    // disocclusion
    const float delta = abs(z_curr - z_prev) * abs(dot(normal_curr, normalize(uvzToView(uv, z_curr))));
    const bool occluded = (delta > fDisocclThres * (1 + z_curr) * 0.1) && !on_edge;

    temporal = min(hist_len_prev * !occluded + 1, iMaxAccumFrames);

    const float4 il_ao_curr = tex2Dlod(samp_il_ao, float4(uv, temporal <= 4 ? MAX_MIP : 0, 0));
    const float4 il_ao_prev = tex2D(samp_il_ao_ac_prev, uv_prev.xy);
    
    il_ao_accum = lerp(il_ao_curr, il_ao_prev, (1 - rcp(temporal)) * !occluded);
}

void PS_Filter(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 il_ao_blur : SV_Target0, out float2 temporal : SV_Target1)
{
    il_ao_blur = 0;

    const uint2 px_coord = uv * BUFFER_SIZE;

    temporal.x = tex2Dfetch(samp_temporal, px_coord).x;\

    const float4 geo = tex2Dfetch(samp_geo, px_coord);
    temporal.y = geo.w;

    const float z = temporal.y;
    [branch]
    if(isSky(z) || isWeapon(z))
        return;

    const float2 zgrad = float2(ddx(z), ddy(z));
    const float3 normal = geo.xyz;
    
    float4 sum = tex2Dlod(samp_il_ao_ac, float4(uv, 1, 0));
    // const float lum = luminance(sum.rgb);
    float weightsum = 1;
    for(int i = -1; i <= 1; i += 2)
        for(int j = -1; j <= 1; j += 2)
        {
            if(i == 0 && j == 0)
                continue;

            const int2 offset_px = int2(i, j);
            const float2 uv_sample = uv + offset_px.xy * fBlurRadius * PIXEL_UV_SIZE;

            [branch]
            if(!isInScreen(uv_sample))
                continue;

            const float4 geo_sample = tex2Dlod(samp_geo, float4(uv_sample, 0, 0));
            const float z_sample = geo_sample.w;
            const float3 normal_sample = geo_sample.xyz;
            
            [branch]
            if(isSky(z_sample) || isWeapon(z_sample))  // nan?
                continue;

            float4 il_ao_sample = tex2Dlod(samp_il_ao_ac, float4(uv_sample, 1, 0));
            // float lum_sample = luminance(il_ao_sample.rgb);

            float w = pow(max(EPS, dot(normal, normal_sample)), 64);                                // normal
            w *= exp(-abs(z - z_sample) / (1 * abs(dot(zgrad, offset_px.xy * fBlurRadius)) + EPS)); // depth
            // w *= exp(-abs(lum - lum_sample) / (fDebug + EPS));               // luminance
            w = saturate(w) * exp(-0.66 * length(offset_px));                                       // gaussian kernel

            weightsum += w;
            sum += il_ao_sample * w;
        }
            
    il_ao_blur = sum / weightsum;
}

void PS_Display(
    in float4 vpos : SV_Position, in float2 uv : TEXCOORD,
    out float4 color : SV_Target)
{
    color = tex2D(ReShade::BackBuffer, uv);

    float4 geo = tex2D(samp_geo, uv);
#if YASSGI_DISABLE_FILTER == 0
    float4 il_ao = tex2Dlod(samp_il_ao_ac_prev, float4(uv, 1, 0));
#else
    float4 il_ao = tex2Dlod(samp_il_ao, float4(uv, 1, 0));  // 2 slices and it's good enough for ao w/ vanilla TAA
#endif
    float ao_mult = fAoStrength > 0 ?
        lerp(1, 1 - il_ao.a, fAoStrength) :  // normal mixing
        exp2(il_ao.a * fAoStrength);  // exponential mixing

    [branch]
    if(iViewMode == 0)  // None
    {
        color.rgb = tex2Dlod(samp_color, float4(uv, 0, 0)).rgb;
#if YASSGI_DISABLE_IL == 0
        if(iMixOrder == 0)
            color.rgb += il_ao.rgb * fIlStrength;
#endif
        color.rgb = color.rgb * ao_mult;
#if YASSGI_DISABLE_IL == 0
        if(iMixOrder == 1)
            color.rgb += il_ao.rgb * fIlStrength;
#endif
        color.rgb = mul(g_colorOutputMat, color.rgb);
    }
    else if(iViewMode == 1)  // Depth
    {
        float z = geo.w;
        if(isWeapon(z))
            color = float3(z / fZRange.x, 0, 0);
        else if (isSky(z))
            color = float3(0.1, 0.5, 1.0);
        else
            color = z / fZRange.y;
    }
    else if(iViewMode == 2)  // Normal
    {
        float3 normal = geo.xyz;
        color = normal * 0.5 + 0.5;
    }
    else if(iViewMode == 3)  // AO
    {
        color.rgb = ao_mult;
    }
#if YASSGI_DISABLE_IL == 0
    else if(iViewMode == 4)  // IL
    {
        color.rgb = mul(g_colorOutputMat, il_ao.rgb * fIlStrength);
    }
#endif
    else if(iViewMode == 5)  // Accum
    {
        color.rgb = tex2Dlod(samp_temporal, float4(uv, 0, 0)).x / iMaxAccumFrames;
    }
}

technique YASSGI <ui_tooltip = "NOTE: This shader requires a motion vector/optical flow shader (google it!)";>
{
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Setup;
        RenderTarget0 = tex_color;
        RenderTarget1 = tex_geo;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_PreBlur;
        RenderTarget0 = tex_blur_geo;
        RenderTarget1 = tex_blur_radiance;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_GI;
        RenderTarget0 = tex_il_ao;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Accum;
        RenderTarget0 = tex_il_ao_ac;
        RenderTarget1 = tex_temporal;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Filter;
        RenderTarget0 = tex_il_ao_ac_prev;
        RenderTarget1 = tex_temporal_z_prev;
    }
    pass {
        VertexShader = PostProcessVS;
        PixelShader = PS_Display;
    }
}

}

