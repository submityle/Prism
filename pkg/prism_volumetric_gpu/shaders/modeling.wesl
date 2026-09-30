// Procedural cloud-modelling twin: composes on-device the final cloud density
// from the four authored/weather modulators, mirroring the CPU golden
// `compose_from_modeling` (in
// `prism_render_architecture::volumetric::modeling`).
//
// The shape half of the density field (design section 4, Nubis-style pipeline)
// folds together: the cloud-type blend of the stratus/cumulus base shapes, the
// weather-driven coverage remap, the per-CloudKind vertical height gradient and
// the energy-preserving detail-erosion remap. Every stage stays in [0, 1] and
// every combination is either a remap or a product of unit-range factors, so
// the composed density is bounded to [0, 1] with no additive brightening or
// darkening drift.
//
// The kernel mirrors the reference formula-for-formula: `saturate`, `lerp`,
// `remap` (with the same EPS span guard) and Hermite `smoothstep` (with the
// same collapsed-edge hard step) are reproduced exactly, and the per-kind
// height profile edges are the same constants the CPU `HeightProfile::for_kind`
// uses. It needs only the portable core-WGSL subset (mul + add + compare) - no
// exp, no pow, no optional feature - so this twin runs unmodified on Metal,
// Vulkan and DX12. The composition is transcendental-free, so CPU and GPU
// evaluate the identical arithmetic and agree to a tight tolerance modulo the
// backend's freedom to contract multiply-adds.
//
// Provenance: standard Nubis-style procedural cloud modelling; no Unreal Engine
// source or derived code.

// Shared span-collapse epsilon, mirroring `math::EPS`.
const EPS: f32 = 1e-6;

struct Params {
    // Number of valid entries in `queries` / outputs in `results`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One modelling query. `40`-byte stride matching the `bytemuck` upload struct.
struct Query {
    cloud_type: f32,
    coverage: f32,
    detail_erosion: f32,
    base_low: f32,
    base_high: f32,
    height_fraction: f32,
    weather_coverage: f32,
    detail_noise: f32,
    kind: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Clamps `x` into `[lo, hi]`, mirroring `math::clamp`.
fn clamp_range(x: f32, lo: f32, hi: f32) -> f32 {
    if (x < lo) {
        return lo;
    } else if (x > hi) {
        return hi;
    }
    return x;
}

// Clamps `x` into `[0, 1]`, mirroring `math::saturate`.
fn saturate_v(x: f32) -> f32 {
    return clamp_range(x, 0.0, 1.0);
}

// Linear interpolation `a + (b - a) * t`, mirroring `math::lerp` (left
// associative, no `mix`).
fn lerp_v(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Remaps `x` from `[in_lo, in_hi]` to `[out_lo, out_hi]`, mirroring
// `math::remap` including the collapsed-span guard.
fn remap_v(x: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    let span = in_hi - in_lo;
    var t = 0.0;
    if (abs(span) >= EPS) {
        t = (x - in_lo) / span;
    }
    return out_lo + (out_hi - out_lo) * t;
}

// Hermite smoothstep clamped to [0, 1], mirroring `math::smoothstep` including
// the collapsed-edge hard step.
fn smoothstep_v(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (abs(span) < EPS) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = saturate_v((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// The four vertical-profile edges (rise_lo, rise_hi, fall_lo, fall_hi) of one
// CloudKind, mirroring `HeightProfile::for_kind`. Kind indices match the CPU
// enum declaration order: 0 Cumulus, 1 Stratus, 2 Cirrus, 3 Cumulonimbus.
fn height_profile(kind: u32) -> vec4<f32> {
    if (kind == 1u) {
        // Stratus.
        return vec4<f32>(0.00, 0.10, 0.55, 0.75);
    } else if (kind == 2u) {
        // Cirrus.
        return vec4<f32>(0.05, 0.25, 0.55, 0.85);
    } else if (kind == 3u) {
        // Cumulonimbus.
        return vec4<f32>(0.03, 0.12, 0.85, 1.00);
    }
    // Cumulus (default, kind == 0).
    return vec4<f32>(0.08, 0.30, 0.65, 0.95);
}

// Per-CloudKind vertical density weight, mirroring `HeightProfile::weight`
// (rising smoothstep times falling smoothstep, saturated).
fn height_gradient(height_fraction: f32, kind: u32) -> f32 {
    let edges = height_profile(kind);
    let h = saturate_v(height_fraction);
    let rising = smoothstep_v(edges.x, edges.y, h);
    let falling = 1.0 - smoothstep_v(edges.z, edges.w, h);
    return saturate_v(rising * falling);
}

// Blends the flat (stratus) and puffy (cumulus) base shapes by cloud type,
// mirroring `cloud_type_shape`.
fn cloud_type_shape(base_low: f32, base_high: f32, cloud_type: f32) -> f32 {
    let low = saturate_v(base_low);
    let high = saturate_v(base_high);
    let t = saturate_v(cloud_type);
    return saturate_v(lerp_v(low, high, t));
}

// Folds authored and weather coverage into one effective fill, mirroring
// `combine_coverage`.
fn combine_coverage(base_coverage: f32, weather_coverage: f32) -> f32 {
    return saturate_v(saturate_v(base_coverage) * saturate_v(weather_coverage));
}

// Applies the coverage gradient to a base shape via the Nubis remap, mirroring
// `coverage_remap`.
fn coverage_remap(base_shape: f32, coverage: f32) -> f32 {
    let base = saturate_v(base_shape);
    let cov = saturate_v(coverage);
    return saturate_v(remap_v(base, 1.0 - cov, 1.0, 0.0, 1.0));
}

// Erodes fluffy detail off the cloud edges via an energy-preserving remap,
// mirroring `detail_erosion`.
fn detail_erosion(base_density: f32, detail_noise: f32, erosion_strength: f32) -> f32 {
    let base = saturate_v(base_density);
    let detail = saturate_v(detail_noise);
    let strength = saturate_v(erosion_strength);
    let eaten = detail * strength;
    return saturate_v(remap_v(base, eaten, 1.0, 0.0, 1.0));
}

@compute @workgroup_size(64)
fn compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let base_shape = cloud_type_shape(q.base_low, q.base_high, q.cloud_type);
    let coverage = combine_coverage(q.coverage, q.weather_coverage);
    let height_weight = height_gradient(q.height_fraction, q.kind);

    let covered = coverage_remap(base_shape, coverage);
    let shaped = saturate_v(covered * saturate_v(height_weight));
    results[idx] = detail_erosion(shaped, q.detail_noise, q.detail_erosion);
}
