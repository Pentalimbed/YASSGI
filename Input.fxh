namespace Input
{

// COLOR

float linearToSrgb(float x)
{
    return x < 0.0031308 ? 12.92 * x : 1.055 * pow(x, rcp(2.4)) - 0.055;
}
float3 linearToSrgb(float3 c)
{
    return float3(linearToSrgb(c.x), linearToSrgb(c.y), linearToSrgb(c.z));
}
float srgbToLinear(float x)
{
    return x < 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}
float3 srgbToLinear(float3 c)
{
    return float3(srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z));
}

// DEPTH

float zToLinearDepth(float z) {return (z - 1) * rcp(RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);}
float linearDepthToZ(float depth) {return depth * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE + 1;}

float getLinearDepth(float2 uv) {return ReShade::GetLinearizedDepth(uv);}
float getZ(float2 uv) {return linearDepthToZ(getLinearDepth(uv));}

float3 uvToViewSpace(float2 uv, float z){return float3(uv * 2 - 1, 1) * z;}
float3 uvToViewSpace(float2 uv){return uvToViewSpace(uv, getZ(uv));}
float2 viewSpaceToUv(float3 pos){return (pos.xy / pos.z + 1) / 2;}

// src: https://gist.github.com/bgolus/a07ed65602c009d5e2f753826e8078a0
// src fr: https://atyuwen.github.io/posts/normal-reconstruction
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

    return normalize(cross(h_deriv, v_deriv));
}

// src: AstrayFX
float2 packNormals(float3 n)
{
    float f = rsqrt(8*n.z+8);
    return n.xy * f + 0.5;
}
float3 unpackNormals(float2 enc)
{
    float2 fenc = enc*4-2;
    float f = dot(fenc,fenc), g = sqrt(1-f/4);
    float3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}

}