//! Glossy-specular ReSTIR *spatial* reuse kernel (GPU twin of the spatial
//! neighbour resampling layer in
//! `prism_render_shading::gi::spec_gi::glossy_reservoir`).
//!
//! This is the second stage of the screen-space glossy ReSTIR pipeline. The
//! `spec_gi_reuse` pass streams the current-frame screen-space GGX candidate and
//! temporally merges the reprojected same-pixel prior reservoir, writing this
//! frame's post-temporal reservoir table. This pass reads that *completed*
//! table read-only (a render-graph barrier separates the two dispatches, so the
//! neighbour reads see a stable snapshot — folding spatial reuse into the single
//! reuse dispatch would race, since neighbours would be mid-write) and, for each
//! pixel:
//!
//!   1. Reconstructs its view-space glossy shading point from the SSR prepass
//!      reverse-Z depth plus the packed view normal/roughness G-buffer (the same
//!      inputs `spec_gi_reuse.wesl` consumes).
//!   2. Seeds a working reservoir from its own post-temporal reservoir.
//!   3. Streams `sample_count` spatial neighbours sampled on a frame-jittered
//!      Fibonacci-spiral disc of pixel radius `radius`, re-targeting each
//!      neighbour's selected sample onto *this* pixel's GGX lobe through the
//!      golden `specr_merge_glossy` (roughness/normal reuse gate included).
//!   4. Clamps the pooled confidence with the roughness-tightened M-cap and
//!      finalises the unbiased contribution weight.
//!   5. Overwrites the resolved specular estimate (+ confidence) target the
//!      `spec_denoise` passes and the energy-conserving composite consume.
//!
//! The spatially pooled estimate is written to the *resolved* target only; the
//! reservoir table itself is left as the pure post-temporal history so spatial
//! correlation never feeds back into next frame's temporal reuse (the standard
//! screen-space ReSTIR split — spatial pooling at resolve, temporal history kept
//! clean — which avoids the progressive over-blur a spatial write-back causes).
//!
//! The reservoir maths is NOT duplicated: it is imported from
//! `spec_gi_reservoir.wesl`, whose numeric parity to the CPU golden is proven by
//! `spec_gi/reservoir_tests.rs`. Only the neighbour-sampling kernel and the
//! storage/dispatch glue live here; the storage ABI mirrors
//! `shading/spec_gi/abi.rs` byte-for-byte (`spatial_tests.rs`).

import prism_render_scene::shaders::spec_gi_reservoir::{
    SpecrGlossyPoint,
    SpecrReservoir,
    SpecrGiSample,
    specr_merge_glossy,
    specr_finalize_glossy,
    specr_glossy_contribution,
    specr_roughness_confidence_cap,
    specr_cap_confidence,
};

// Dielectric Fresnel-0 fallback (matches `spec_gi_reuse.wesl::SGR_DEFAULT_F0`):
// 0.04 is the standard dielectric reflectance until a per-pixel `f0` G-buffer
// channel arrives with the material-sample repack path.
const SGS_DEFAULT_F0: f32 = 0.04;

// Golden angle (radians) for the Fibonacci-spiral neighbour disc: successive
// taps are rotated by this so a handful of samples cover the disc near-uniformly
// without clustering. 2*pi * (1 - 1/phi) = 2.399963...
const SGS_GOLDEN_ANGLE: f32 = 2.399963229728653;

// Storage twin of the working reservoir — identical 64-byte layout to
// `spec_gi_reuse.wesl::GpuSpecrReservoir` and `shading/spec_gi/abi.rs`.
struct GpuSpecrReservoir {
    visible_point: vec3<f32>,
    w_sum: f32,
    sample_point: vec3<f32>,
    m: f32,
    radiance: vec3<f32>,
    w: f32,
    has_sample: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Immediate (push-constant) block driving one spatial dispatch. Mirrors
// `GpuSpecGiSpatialParams` in `shading/spec_gi/abi.rs` (96 bytes; the `mat4x4`
// forces 16-byte alignment so the eight trailing scalars pack with no padding).
struct SpatialParams {
    view_from_clip: mat4x4<f32>,
    width: u32,
    height: u32,
    radius: f32,
    sample_count: u32,
    frame: u32,
    m_cap: f32,
    roughness_cap_base: f32,
    sigma_roughness: f32,
}

// This frame's post-temporal reservoir table (read-only snapshot).
@group(0) @binding(0) var<storage, read> reservoirs: array<GpuSpecrReservoir>;
// Reverse-Z device depth used to reconstruct the view-space shading point.
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
// Packed view normal (xyz, [0,1] encoded) + perceptual roughness (w).
@group(0) @binding(2) var normal_roughness: texture_2d<f32>;
// Resolved specular estimate (rgb) + reservoir confidence (a) for the denoiser.
@group(0) @binding(3) var resolved_out: texture_storage_2d<rgba16float, write>;

var<immediate> params: SpatialParams;

// Reconstruct the view-space position from framebuffer UV + reverse-Z depth
// (identical to `spec_gi_reuse.wesl::sgr_reconstruct_view_position`).
fn sgs_reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = params.view_from_clip * clip;
    return view.xyz / view.w;
}

// PCG-style hash → uniform float in [0, 1) (identical to the reuse kernel's RNG
// so both passes draw deterministic, reproducible selectors).
fn sgs_hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

fn sgs_rng(pixel: vec2<u32>, salt: u32) -> f32 {
    let h = sgs_hash_u32(pixel.x * 1973u + pixel.y * 9277u + salt * 26699u + 1u);
    return f32(h) * (1.0 / 4294967296.0);
}

// Inverse of the reuse kernel's `sgr_pack` (same flattened storage layout).
fn sgs_unpack(g: GpuSpecrReservoir) -> SpecrReservoir {
    var s: SpecrGiSample;
    s.visible_point = g.visible_point;
    s.sample_point = g.sample_point;
    s.radiance = g.radiance;
    var r: SpecrReservoir;
    r.sample = s;
    r.w_sum = g.w_sum;
    r.m = g.m;
    r.w = g.w;
    r.has_sample = g.has_sample;
    return r;
}

// Frame-jittered Fibonacci-spiral disc offset for neighbour tap `k`. The radius
// grows as sqrt((k+0.5)/count) so taps are area-uniform across the disc; the
// per-pixel/per-frame angular jitter decorrelates the pattern spatially and
// temporally (the denoiser + temporal reuse resolve the residual).
fn sgs_neighbour_offset(pixel: vec2<u32>, k: u32) -> vec2<i32> {
    let count = max(params.sample_count, 1u);
    let jitter = sgs_rng(pixel, params.frame * 977u + 7u);
    let frac = (f32(k) + 0.5) / f32(count);
    let r = params.radius * sqrt(frac);
    let angle = (f32(k) + jitter) * SGS_GOLDEN_ANGLE;
    let offset = r * vec2<f32>(cos(angle), sin(angle));
    return vec2<i32>(i32(round(offset.x)), i32(round(offset.y)));
}

@compute @workgroup_size(8, 8, 1)
fn spec_gi_spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let pixel = vec2<i32>(i32(id.x), i32(id.y));
    let idx = id.y * params.width + id.x;

    let device_depth = textureLoad(scene_depth, pixel, 0).x;
    // Reverse-Z: depth 0 is the far plane / cleared background → no surface.
    if device_depth <= 0.0 {
        textureStore(resolved_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(f32(params.width), f32(params.height));
    let nr = textureLoad(normal_roughness, pixel, 0);
    let normal_view = normalize(nr.xyz * 2.0 - 1.0);
    let roughness = clamp(nr.w, 0.0, 1.0);

    let position_view = sgs_reconstruct_view_position(uv, device_depth);
    let incident = normalize(position_view);

    var point: SpecrGlossyPoint;
    point.position = position_view;
    point.normal = normal_view;
    point.view = -incident;
    point.roughness = roughness;
    point.anisotropy = 0.0;
    point.f0 = vec3<f32>(SGS_DEFAULT_F0);

    // Seed the working reservoir with this pixel's own post-temporal reservoir,
    // then pool spatial neighbours into it (GRIS cross-pixel resampling).
    var r = sgs_unpack(reservoirs[idx]);

    let w = i32(params.width);
    let h = i32(params.height);
    let count = max(params.sample_count, 1u);
    for (var k = 0u; k < count; k = k + 1u) {
        let offset = sgs_neighbour_offset(id.xy, k);
        if offset.x == 0 && offset.y == 0 {
            continue;
        }
        let np = pixel + offset;
        if np.x < 0 || np.y < 0 || np.x >= w || np.y >= h {
            continue;
        }
        let nd = textureLoad(scene_depth, np, 0).x;
        if nd <= 0.0 {
            continue;
        }
        let n_nr = textureLoad(normal_roughness, np, 0);
        let n_normal = normalize(n_nr.xyz * 2.0 - 1.0);
        let n_roughness = clamp(n_nr.w, 0.0, 1.0);
        let n_idx = u32(np.y) * params.width + u32(np.x);
        let neighbour = sgs_unpack(reservoirs[n_idx]);
        let _merged = specr_merge_glossy(
            &r,
            neighbour,
            point,
            n_roughness,
            n_normal,
            params.sigma_roughness,
            sgs_rng(id.xy, k + 1u),
        );
    }

    // Roughness-tightened confidence clamp, capped by the global M-cap: the same
    // ReSTIR history clamp the temporal pass applies, now bounding the pooled
    // spatial confidence so a dense neighbourhood cannot over-trust one estimate.
    let rough_cap = specr_roughness_confidence_cap(roughness, params.roughness_cap_base);
    specr_cap_confidence(&r, min(rough_cap, params.m_cap));

    // Finalise the unbiased contribution weight `W` in-place; scalar return
    // unused (folded into the resolved contribution below).
    _ = specr_finalize_glossy(&r, point);
    let contrib = specr_glossy_contribution(r, point);

    let confidence = clamp(r.m / max(params.m_cap, 1.0e-4), 0.0, 1.0);
    textureStore(resolved_out, pixel, vec4<f32>(contrib, confidence));
}
