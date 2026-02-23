#include <metal_stdlib>
using namespace metal;

// ============ PARTICLE STRUCT ============
struct Particle {
    float x;
    float y;
    float vx;
    float vy;
    float color; // 0 = orange, 1 = blue
};

struct PhysicsUniforms {
    float amplitude;
    float forceScale;
    float friction;
    float noise;
    float maxSpeed;
    float dt;
    uint particleCount;
    uint gridSize;
    uint isMusic;
    uint _pad;
};

// ============ COMPUTE SHADER: Particle Physics ============
// Simple hash for per-particle random
float hash(uint seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return float(seed) / float(0xFFFFFFFF);
}

kernel void updateParticles(
    device Particle* particles [[buffer(0)]],
    device const float* lut [[buffer(1)]],
    device const PhysicsUniforms& u [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= u.particleCount) return;

    Particle p = particles[gid];
    uint g = u.gridSize;

    // Sample gradient LUT (bilinear interpolation)
    if (u.forceScale > 0.001) {
        float fx = (p.x + 1.0) / 2.0 * float(g - 1);
        float fy = (p.y + 1.0) / 2.0 * float(g - 1);
        int ix = clamp(int(fx), 0, int(g) - 2);
        int iy = clamp(int(fy), 0, int(g) - 2);
        float tx = fx - float(ix);
        float ty = fy - float(iy);

        int a = (iy * int(g) + ix) * 2;
        int b = a + 2;
        int c = ((iy + 1) * int(g) + ix) * 2;
        int d = c + 2;

        float gx = (1-ty) * ((1-tx)*lut[a] + tx*lut[b]) + ty * ((1-tx)*lut[c] + tx*lut[d]);
        float gy = (1-ty) * ((1-tx)*lut[a+1] + tx*lut[b+1]) + ty * ((1-tx)*lut[c+1] + tx*lut[d+1]);

        p.vx -= gx * u.forceScale;
        p.vy -= gy * u.forceScale;
    }

    // Per-particle deterministic noise (frame-varying via gid + amplitude)
    uint seed = gid * 1237 + uint(u.amplitude * 10000);
    float rx = hash(seed) - 0.5;
    float ry = hash(seed + 7919) - 0.5;
    p.vx += rx * u.noise;
    p.vy += ry * u.noise;

    // Speed clamp
    float sp = sqrt(p.vx*p.vx + p.vy*p.vy);
    if (sp > u.maxSpeed) {
        p.vx = p.vx / sp * u.maxSpeed;
        p.vy = p.vy / sp * u.maxSpeed;
    }

    // Friction + integrate
    p.vx *= u.friction;
    p.vy *= u.friction;
    p.x += p.vx * u.dt * 60.0;
    p.y += p.vy * u.dt * 60.0;

    // Boundary clamp
    float r = sqrt(p.x*p.x + p.y*p.y);
    if (r > 0.99) {
        p.x = p.x / r * 0.98;
        p.y = p.y / r * 0.98;
        p.vx *= -0.3;
        p.vy *= -0.3;
    }

    particles[gid] = p;
}

// ============ RENDER SHADERS ============

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 particleColor;
    float pointSize [[point_size]];
};

struct RenderUniforms {
    float2 viewportSize;
    float particleSize;
    float plateRadius;
};

vertex VertexOut particleVertex(
    uint vid [[vertex_id]],
    device const Particle* particles [[buffer(0)]],
    device const RenderUniforms& ru [[buffer(1)]]
) {
    Particle p = particles[vid];
    VertexOut out;

    // Map particle coords [-1,1] to clip space, accounting for plate position
    float2 pos = float2(p.x, -p.y) * ru.plateRadius;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = float2(0.0);

    // Color: orange or blue
    if (p.color < 0.5) {
        out.particleColor = float4(0.91, 0.53, 0.23, 0.9); // #e8863a
    } else {
        out.particleColor = float4(0.29, 0.62, 0.84, 0.9); // #4a9ed6
    }

    out.pointSize = ru.particleSize;
    return out;
}

fragment float4 particleFragment(
    VertexOut in [[stage_in]],
    float2 pointCoord [[point_coord]]
) {
    // Circular particle with slight glow
    float dist = length(pointCoord - float2(0.5));
    if (dist > 0.5) discard_fragment();

    float alpha = in.particleColor.a * smoothstep(0.5, 0.3, dist);
    return float4(in.particleColor.rgb, alpha);
}

// Background plate
struct PlateVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex PlateVertexOut plateVertex(uint vid [[vertex_id]]) {
    // Full-screen quad
    float2 positions[4] = {float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1)};
    PlateVertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = positions[vid];
    return out;
}

fragment float4 plateFragment(PlateVertexOut in [[stage_in]]) {
    float r = length(in.uv);
    if (r > 1.02) discard_fragment();

    // Dark plate with subtle edge
    float3 plateColor = float3(0.1, 0.1, 0.14);
    float3 edgeColor = float3(0.2, 0.2, 0.2);
    float edge = smoothstep(0.97, 1.0, r);
    float3 color = mix(plateColor, edgeColor, edge);
    float alpha = smoothstep(1.02, 1.0, r);
    return float4(color, alpha);
}
