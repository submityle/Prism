//! The water-surface raster pass: the `@vertex`/`@fragment` draw that turns the
//! solved water fields into visible pixels in the `HDR` target.
//!
//! Every other water shader in this crate is a `@compute` solver that writes
//! device buffers (`water_ocean`, `water_flip`, `water_pbf`, `water_surface`,
//! `water_render_fx`); none of them draw. This module is the missing raster
//! half contracted by
//! `prism_render_architecture::water::gpu::surface_pass`: a single shared
//! vertex stage displaces the surface grid, and a per-frontend fragment stage
//! shades it through the §5 lighting-response fork. The entry-point names are
//! byte-identical to that `CPU` contract
//! (`SurfaceDrawDescriptor::{vertex_entry, fragment_entry}`):
//! `water_surface_vs` plus `water_surface_fs_{pbr,npr,custom,hybrid}`.
//!
//! The geometry stage is frontend-agnostic (the displaced mesh is identical for
//! every frontend); only the fragment stage forks. The physically based path
//! reuses the shared single-layer-water `BSDF` lobe `water_direct` from
//! `water.wesl` (the device twin of
//! `prism_render_shading::water::evaluate_water_direct`), so the `PBR` highlight
//! and `Beer-Lambert` body match the opaque material exactly. `NPR` quantizes
//! that same physically lit result onto a ramp (`Zelda`/`Genshin`-style),
//! custom adds an author emissive tint, and hybrid blends `PBR` and `NPR` by a
//! shore/foam factor. Refraction samples the already-rendered scene color
//! behind the surface, offset by the surface normal, matching the `UE5` Single
//! Layer Water and `Crest` transparent composite.

import prism_render_scene::shaders::brdf::{
    SurfaceSample,
    ShadingFrame,
    DirectLightSample,
    brdf_normalize_or,
};
import prism_render_scene::shaders::water::{
    water_direct,
};

// Per-view constants the host uploads once per frame.
struct WaterSurfaceView {
    // Clip-from-world transform for the surface vertices.
    clip_from_world: mat4x4<f32>,
    // World-space camera position (xyz); w is the refraction screen offset.
    world_camera_position: vec4<f32>,
    // Direction *towards* the key light (xyz, normalized); w unused.
    sun_direction: vec4<f32>,
    // Key-light illuminance (rgb); w unused.
    sun_illuminance: vec4<f32>,
    // base perceptual roughness (x), reflectance (y), optical thickness (z),
    // foam whiten strength (w).
    surface_params: vec4<f32>,
    // Base water albedo (rgb) and the minimum surface alpha (w).
    water_color: vec4<f32>,
    // NPR ramp step count (x), toon foam threshold (y), author tint strength
    // (z), hybrid shore blend bias (w).
    style_params: vec4<f32>,
    // Framebuffer size in pixels (xy) for the refraction screen lookup; zw
    // unused.
    viewport: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: WaterSurfaceView;
// Base (undisplaced) grid positions, one vec4 per surface vertex (xyz; w = 1).
@group(0) @binding(1) var<storage, read> base_positions: array<vec4<f32>>;
// Per-vertex texture coordinates (xy); zw unused.
@group(0) @binding(2) var<storage, read> surface_uvs: array<vec4<f32>>;
// Solver displacement per vertex (xyz world-space offset); w unused.
@group(0) @binding(3) var<storage, read> displacement: array<vec4<f32>>;
// Surface normal (xyz, unit) and foam coverage (w in 0..=1) per vertex.
@group(0) @binding(4) var<storage, read> normal_foam: array<vec4<f32>>;
// Already-rendered opaque scene color, sampled for refraction.
@group(0) @binding(5) var scene_color: texture_2d<f32>;
@group(0) @binding(6) var scene_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) foam: f32,
}

// Shared geometry stage: displace the base grid vertex by the solved field and
// project it. Identical for every frontend (§5).
@vertex
fn water_surface_vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let i = vertex_index;
    let world_position = base_positions[i].xyz + displacement[i].xyz;

    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world_position, 1.0);
    out.world_position = world_position;
    out.normal = normal_foam[i].xyz;
    out.uv = surface_uvs[i].xy;
    out.foam = clamp(normal_foam[i].w, 0.0, 1.0);
    return out;
}

// Build the frontend-agnostic surface description from the authored water
// parameters and the per-vertex foam coverage (foam raises roughness and
// whitens the body, matching the Jacobian foam mask).
fn surface_from(foam: f32) -> SurfaceSample {
    var s: SurfaceSample;
    let foam_color = mix(view.water_color.rgb, vec3<f32>(1.0), foam);
    s.base_color = foam_color;
    s.metallic = 0.0;
    s.perceptual_roughness = clamp(view.surface_params.x + foam * 0.5, 0.0, 1.0);
    s.reflectance = clamp(view.surface_params.y, 0.0, 1.0);
    s.ambient_occlusion = 1.0;
    s.emissive = vec3<f32>(0.0);
    s.clearcoat = 0.0;
    s.clearcoat_roughness = 0.0;
    s.sheen = 0.0;
    s.subsurface = 0.0;
    s.thickness = clamp(view.surface_params.z, 0.0, 1.0);
    s.anisotropy = 0.0;
    s.anisotropy_rotation = 0.0;
    return s;
}

// Build the view/normal shading frame for a world-space surface point.
fn frame_at(world_position: vec3<f32>, normal: vec3<f32>) -> ShadingFrame {
    let n = brdf_normalize_or(normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(view.world_camera_position.xyz - world_position, n);
    // Any tangent orthogonal to the normal; water uses an isotropic lobe so the
    // exact axis is irrelevant to the highlight.
    let ref_axis = select(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        abs(n.y) < 0.99,
    );
    let tangent = brdf_normalize_or(cross(ref_axis, n), vec3<f32>(1.0, 0.0, 0.0));
    let bitangent = cross(n, tangent);

    var f: ShadingFrame;
    f.normal = n;
    f.view = v;
    f.tangent = tangent;
    f.bitangent = bitangent;
    return f;
}

fn key_light() -> DirectLightSample {
    var l: DirectLightSample;
    l.direction = brdf_normalize_or(view.sun_direction.xyz, vec3<f32>(0.0, 1.0, 0.0));
    l.illuminance = view.sun_illuminance.rgb;
    l.visibility = 1.0;
    return l;
}

// Sample the scene color behind the surface, offset by the surface normal, for
// a cheap screen-space refraction. The offset is scaled by the authored amount
// in `world_camera_position.w`.
fn refraction_color(clip_position: vec4<f32>, normal: vec3<f32>) -> vec3<f32> {
    let screen_uv = clip_position.xy / max(view.viewport.xy, vec2<f32>(1.0));
    let offset = normal.xz * view.world_camera_position.w;
    let uv = clamp(screen_uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleLevel(scene_color, scene_sampler, uv, 0.0).rgb;
}

// Perceptual luminance for ramp quantization.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Quantize a scalar onto `steps` discrete bands.
fn ramp_quantize(x: f32, steps: f32) -> f32 {
    let n = max(steps, 1.0);
    return floor(x * n + 0.5) / n;
}

// Shared physically based response, composited with refraction. Returns the
// pre-multiplied color and the surface alpha.
fn shade_pbr_core(in: VertexOutput) -> vec4<f32> {
    let surface = surface_from(in.foam);
    let frame = frame_at(in.world_position, in.normal);
    let light = key_light();

    let lit = water_direct(surface, frame, light);
    let refracted = refraction_color(in.clip_position, frame.normal);

    // Grazing angles reflect more and refract less (Fresnel-weighted alpha).
    let n_dot_v = clamp(dot(frame.normal, frame.view), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - n_dot_v, 5.0);
    let alpha = clamp(max(view.water_color.a, fresnel) + in.foam, 0.0, 1.0);

    let body = mix(refracted, lit + refracted * (1.0 - fresnel), fresnel);
    let color = mix(body, vec3<f32>(1.0), in.foam * view.surface_params.w);
    return vec4<f32>(color * alpha, alpha);
}

@fragment
fn water_surface_fs_pbr(in: VertexOutput) -> @location(0) vec4<f32> {
    return shade_pbr_core(in);
}

@fragment
fn water_surface_fs_npr(in: VertexOutput) -> @location(0) vec4<f32> {
    let pbr = shade_pbr_core(in);
    let alpha = pbr.a;
    let base = select(pbr.rgb / alpha, vec3<f32>(0.0), alpha <= 0.0);

    // Ramp-quantize the physically lit luminance and recolor; add a hand-drawn
    // toon foam edge where coverage crosses the authored threshold.
    let lum = luminance(base);
    let stepped = ramp_quantize(lum, view.style_params.x);
    let toon = base * select(1.0, stepped / max(lum, 1.0e-4), lum > 1.0e-4);
    let foam_line = step(view.style_params.y, in.foam);
    let color = mix(toon, vec3<f32>(1.0), foam_line);
    return vec4<f32>(color * alpha, alpha);
}

@fragment
fn water_surface_fs_custom(in: VertexOutput) -> @location(0) vec4<f32> {
    // Author-hook default: the physical base plus a tunable emissive tint so a
    // custom closure has a non-trivial, non-identity starting point.
    let pbr = shade_pbr_core(in);
    let tint = view.water_color.rgb * view.style_params.z;
    return vec4<f32>(pbr.rgb + tint * pbr.a, pbr.a);
}

@fragment
fn water_surface_fs_hybrid(in: VertexOutput) -> @location(0) vec4<f32> {
    let pbr = shade_pbr_core(in);
    let npr = water_surface_fs_npr(in);
    // Blend physical body into stylized shallows by a shore factor driven by
    // foam coverage and the authored bias.
    let shore = clamp(in.foam + view.style_params.w, 0.0, 1.0);
    return mix(pbr, npr, shore);
}
