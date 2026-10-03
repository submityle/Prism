//! Glossy-specular ReSTIR *reuse* kernel (GPU twin of the spatio-temporal
//! resampling layer in `prism_render_shading::gi::spec_gi::glossy_reservoir`).
//!
//! This is the production compute pass that layers reservoir reuse on top of the
//! screen-space GGX candidate rays the SSR trace already produces. Each pixel:
//!
//!   1. Reconstructs its view-space glossy shading point from the SSR prepass
//!      reverse-Z depth plus the packed view normal/roughness G-buffer (the same
//!      `normal_roughness` target `ssr.wesl` consumes).
//!   2. Streams the current-frame screen-space candidate (incident radiance +
//!      source pdf sampled by the SSR trace) into a fresh reservoir via the
//!      golden `specr_stream_glossy_candidate` — re-targeted onto this pixel's
//!      GGX lobe.
//!   3. When temporal reuse is enabled, merges the *same-pixel* reprojected prior
//!      reservoir (`specr_merge_glossy`) and clamps its confidence with the
//!      roughness-tightened M-cap (`specr_roughness_confidence_cap` /
//!      `specr_cap_confidence`) — the ReSTIR history clamp.
//!   4. Finalises the unbiased contribution weight and resolves the specular
//!      estimate (`specr_finalize_glossy` / `specr_glossy_contribution`).
//!   5. Writes the packed reservoir to the ping-pong storage buffer (for the next
//!      frame's temporal reuse) and the resolved specular to the output target
//!      (for the `spec_denoise` passes and the energy-conserving composite).
//!
//! Motion-vector reprojection and spatial neighbour reuse are the documented
//! NEXT slice; this slice lands same-pixel temporal reuse, which is a complete,
//! real first stage (no placeholder maths — every reservoir operation is the
//! golden one, imported rather than duplicated).
//!
//! The reservoir maths is NOT duplicated: it is imported from
//! `spec_gi_reservoir.wesl`, whose numeric parity to the CPU golden is proven by
//! `spec_gi/reservoir_tests.rs`. Only the storage ABI and the dispatch glue live
//! here. The `GpuSpecrReservoir` / `SpecGiReuseConfig` layouts mirror
//! `shading/spec_gi/abi.rs` byte-for-byte (guarded by `reuse_tests.rs`).

import prism_render_scene::shaders::spec_gi_reservoir::{
    SpecrGlossyPoint,
    SpecrGiSample,
    SpecrReservoir,
    specr_reservoir_new,
    specr_stream_glossy_candidate,
    specr_merge_glossy,
    specr_finalize_glossy,
    specr_glossy_contribution,
    specr_roughness_confidence_cap,
    specr_cap_confidence,
};

// Dielectric Fresnel-0 fallback until the reuse pass is fed a per-pixel `f0`
// G-buffer channel (metals/coloured specular arrive with the material-sample
// repack path). 0.04 is the standard dielectric reflectance.
const SGR_DEFAULT_F0: f32 = 0.04;

// Storage twin of the working reservoir. Flattened so each `vec3<f32>` shares
// its 16-byte row with the scalar it pairs with — matches `GpuSpecrReservoir`
// in `shading/spec_gi/abi.rs` byte-for-byte (64 bytes, 64-byte array stride).
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

// Uniform block driving one reuse dispatch. Mirrors `GpuSpecGiReuseConfig` in
// `shading/spec_gi/abi.rs` (96 bytes; the `mat4x4` forces 16-byte alignment so
// the eight trailing scalars pack with no padding).
struct SpecGiReuseConfig {
    view_from_clip: mat4x4<f32>,
    width: u32,
    height: u32,
    m_cap: f32,
    roughness_cap_base: f32,
    sigma_roughness: f32,
    near: f32,
    temporal_enabled: u32,
    spatial_enabled: u32,
}

@group(0) @binding(0) var<uniform> config: SpecGiReuseConfig;
// Previous-frame reservoirs (same-pixel temporal source), row-major `width`.
@group(0) @binding(1) var<storage, read> prior_reservoirs: array<GpuSpecrReservoir>;
// This-frame reservoirs (temporal target for the next frame).
@group(0) @binding(2) var<storage, read_write> out_reservoirs: array<GpuSpecrReservoir>;
// Reverse-Z device depth used to reconstruct the view-space shading point.
@group(0) @binding(3) var scene_depth: texture_2d<f32>;
// Packed view normal (xyz, [0,1] encoded) + perceptual roughness (w).
@group(0) @binding(4) var normal_roughness: texture_2d<f32>;
// Current-frame screen-space candidate: rgb = incident radiance, a = source pdf.
@group(0) @binding(5) var candidate: texture_2d<f32>;
// Resolved specular estimate (rgb) + reservoir confidence (a) for the denoiser.
@group(0) @binding(6) var resolved_out: texture_storage_2d<rgba16float, write>;

// Reconstruct the view-space position from framebuffer UV + reverse-Z depth
// (identical to `ssr.wesl::reconstruct_view_position`).
fn sgr_reconstruct_view_position(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, device_depth);
    let clip = vec4<f32>(ndc, 1.0);
    let view = config.view_from_clip * clip;
    return view.xyz / view.w;
}

// PCG-style hash → uniform float in [0, 1). Deterministic per pixel so the pass
// is reproducible frame to frame (the reservoir stream/merge decisions need an
// unbiased selector, not a specific sequence).
fn sgr_hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

fn sgr_rng(pixel: vec2<u32>, salt: u32) -> f32 {
    let h = sgr_hash_u32(pixel.x * 1973u + pixel.y * 9277u + salt * 26699u + 1u);
    return f32(h) * (1.0 / 4294967296.0);
}

// Pack the working reservoir into its storage record (round-trippable with
// `sgr_unpack`).
fn sgr_pack(r: SpecrReservoir) -> GpuSpecrReservoir {
    var g: GpuSpecrReservoir;
    g.visible_point = r.sample.visible_point;
    g.w_sum = r.w_sum;
    g.sample_point = r.sample.sample_point;
    g.m = r.m;
    g.radiance = r.sample.radiance;
    g.w = r.w;
    g.has_sample = r.has_sample;
    g._pad0 = 0u;
    g._pad1 = 0u;
    g._pad2 = 0u;
    return g;
}

// Inverse of `sgr_pack`.
fn sgr_unpack(g: GpuSpecrReservoir) -> SpecrReservoir {
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

@compute @workgroup_size(8, 8, 1)
fn spec_gi_reuse(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.width || id.y >= config.height {
        return;
    }
    let pixel = vec2<i32>(i32(id.x), i32(id.y));
    let idx = id.y * config.width + id.x;

    let device_depth = textureLoad(scene_depth, pixel, 0).x;
    // Reverse-Z: depth 0 is the far plane / cleared background → no surface.
    if device_depth <= 0.0 {
        var empty = sgr_pack(specr_reservoir_new());
        out_reservoirs[idx] = empty;
        textureStore(resolved_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(f32(config.width), f32(config.height));
    let nr = textureLoad(normal_roughness, pixel, 0);
    let normal_view = normalize(nr.xyz * 2.0 - 1.0);
    let roughness = clamp(nr.w, 0.0, 1.0);

    let position_view = sgr_reconstruct_view_position(uv, device_depth);
    let incident = normalize(position_view);

    var point: SpecrGlossyPoint;
    point.position = position_view;
    point.normal = normal_view;
    point.view = -incident;
    point.roughness = roughness;
    point.anisotropy = 0.0;
    point.f0 = vec3<f32>(SGR_DEFAULT_F0);

    // Current-frame screen-space candidate. The SSR trace importance-samples the
    // GGX lobe; its averaged reflected radiance + confidence are the candidate's
    // incident radiance and source pdf. The incident direction is reconstructed
    // from the dominant specular reflection of the view ray about the surface
    // normal (a reconnection shift: only the direction of
    // `sample_point - visible_point` is used by the glossy target function).
    let cand = textureLoad(candidate, pixel, 0);
    let reflect_dir = reflect(incident, normal_view);

    var sample: SpecrGiSample;
    sample.visible_point = position_view;
    sample.sample_point = position_view + reflect_dir;
    sample.radiance = max(cand.xyz, vec3<f32>(0.0));
    let source_pdf = max(cand.w, 1.0e-4);

    var r = specr_reservoir_new();
    let _streamed = specr_stream_glossy_candidate(&r, point, sample, source_pdf, sgr_rng(id.xy, 0u));

    if config.temporal_enabled != 0u {
        let prior = sgr_unpack(prior_reservoirs[idx]);
        // Same-pixel reprojection: the prior shares this pixel's normal and
        // roughness (motion-vector reprojection is the next slice), so the reuse
        // gate uses the current point's normal/roughness.
        let _merged = specr_merge_glossy(
            &r,
            prior,
            point,
            roughness,
            normal_view,
            config.sigma_roughness,
            sgr_rng(id.xy, 1u),
        );
        // Roughness-tightened temporal confidence clamp (sharp surfaces drop
        // stale history faster), capped by the global temporal M-cap.
        let rough_cap = specr_roughness_confidence_cap(roughness, config.roughness_cap_base);
        specr_cap_confidence(&r, min(rough_cap, config.m_cap));
    }

    // Finalise the unbiased contribution weight `W` in-place (folds into the
    // resolved contribution below); the scalar return is unused.
    _ = specr_finalize_glossy(&r, point);
    let contrib = specr_glossy_contribution(r, point);

    out_reservoirs[idx] = sgr_pack(r);
    let confidence = clamp(r.m / max(config.m_cap, 1.0e-4), 0.0, 1.0);
    textureStore(resolved_out, pixel, vec4<f32>(contrib, confidence));
}
