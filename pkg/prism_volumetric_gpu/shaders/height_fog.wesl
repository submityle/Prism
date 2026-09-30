// Exponential height-fog density twin: evaluates on-device the extinction
// density of an exponential height-fog layer, mirroring the CPU golden
// `height_fog_density` (in `prism_render_architecture::volumetric::fog`).
//
// The ground fog layer (design section 9f) is densest at sea level and thins
// with altitude: `density_at_sea_level * e^{-falloff * altitude}`, culled to
// exactly zero at or above `max_height` so the fog volume has a hard,
// artist-controlled ceiling. The sea-level density, falloff and ceiling are
// each floored at zero, and negative altitudes are treated as sea level so the
// value never exceeds the sea-level density. This is an extinction coefficient,
// not a probability, so it is only floored at zero, not clamped to one. The CPU
// golden `height_fog_density` owns that math; this kernel is the on-device twin
// that runs one thread per query and reproduces the same value.
//
// The exponential uses the SAME hand-rolled `exp_approx` the reference uses
// (base-two range reduction: a fractional seven-term polynomial times an
// integer power assembled from the float exponent field), NOT the device-native
// `exp`, so the twin stays bit-close to the reference and the parity test can
// assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops and
// multiply/add - so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard exponential height-fog extinction; no Unreal Engine
// source or derived code.

const LOG2_E: f32 = 1.4426950408889634;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One height-fog query: the sample altitude plus the layer parameters. Matches
// `GpuQuery` in `src/height_fog.rs`.
struct Query {
    altitude: f32,
    density_at_sea_level: f32,
    falloff: f32,
    max_height: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn exp2_int(k: i32) -> f32 {
    if (k < -126) {
        return 0.0;
    }
    if (k > 127) {
        return 3.4028235e38 * 3.4028235e38;
    }
    return bitcast<f32>(u32(k + 127) << 23u);
}

fn exp2_frac(f: f32) -> f32 {
    let c1 = 0.69314718;
    let c2 = 0.2402265;
    let c3 = 0.05550411;
    let c4 = 0.00961813;
    let c5 = 0.00133336;
    let c6 = 0.00015403;
    return 1.0 + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * (c5 + f * c6)))));
}

fn exp_approx(x: f32) -> f32 {
    let t = x * LOG2_E;
    let k = floor(t);
    let frac = t - k;
    return exp2_frac(frac) * exp2_int(i32(k));
}

// Exponential height-fog extinction density, mirroring `height_fog_density`.
fn height_fog_density(altitude_in: f32, density_at_sea_level: f32, falloff_in: f32, max_height: f32) -> f32 {
    let base = max(density_at_sea_level, 0.0);
    let falloff = max(falloff_in, 0.0);
    let ceiling = max(max_height, 0.0);
    let altitude = max(altitude_in, 0.0);
    if (altitude >= ceiling) {
        return 0.0;
    }
    return base * exp_approx(-falloff * altitude);
}

@compute @workgroup_size(64)
fn height_fog_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = height_fog_density(q.altitude, q.density_at_sea_level, q.falloff, q.max_height);
}
