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
import prism_render_scene::shaders::lighting::{
    DirectionalLight,
    PunctualLight,
    LightEnvironment,
    sample_directional_light,
    sample_punctual_light,
    sh_irradiance,
    sh_radiance,
    env_brdf_approx,
    environment_has_image_based,
    INV_PI,
};

// Per-view constants the host uploads once per frame.
struct WaterSurfaceView {
    // Clip-from-world transform for the surface vertices.
    clip_from_world: mat4x4<f32>,
    // World-space camera position (xyz); w is the refraction screen offset.
    world_camera_position: vec4<f32>,
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

// Shared engine lighting base (`@group(1)`): byte-identical bindings to the
// opaque `shading_resolve` path. All analytic directional + punctual lights
// plus the image-based `SH` environment, so every §5 water frontend consumes
// the exact same light table as the rest of the scene instead of a single
// hard-coded key light.
@group(1) @binding(0) var<storage, read> directional_lights: array<DirectionalLight>;
@group(1) @binding(1) var<storage, read> punctual_lights: array<PunctualLight>;
@group(1) @binding(2) var<storage, read> light_env: array<LightEnvironment>;

// Virtual-shadow-map base (`@group(2)`): a byte-for-byte twin of the opaque
// `shading_resolve` pass's VSM group (there it is `@group(6)`), re-numbered to
// `@group(2)` here. The water surface's primary directional light samples the
// identical demand-paged page table + physical atlas the rest of the scene
// reads, so a shadow cast by an opaque caster also darkens the water it falls
// on. The one intentional divergence from the standalone `vsm_sample` shader is
// the miss return: this twin returns the sentinel `VSM_MISS = -1.0` on an
// out-of-window page or an unmapped slot so the caller keeps the analytic
// visibility (the water pass has no cascaded fallback of its own).
@group(2) @binding(0) var<storage, read> vsm_page_table: array<u32>;
@group(2) @binding(1) var vsm_physical_atlas: texture_2d<f32>;
@group(2) @binding(2) var vsm_physical_sampler: sampler;

// Mirrors `GpuVsmResolveParams` in `shading/resolve/abi.rs` (96 bytes). The
// three leading 16-byte scalar rows are followed by three `vec3<f32>` axes, each
// on its 16-byte boundary with the trailing `f32` filling the fourth column, so
// the std140 uniform layout is byte-identical to the `#[repr(C)]` record.
struct VsmResolveParams {
    // --- clipmap layout (mirrors ClipmapConfig / GpuVsmSampleParams) ---
    levels: u32,
    page_size: u32,
    pages_per_level_edge: u32,
    page_coord_bias: i32,
    level0_texel_world_size: f32,
    level0_max_distance: f32,
    // --- physical atlas geometry ---
    physical_pages: u32,
    physical_pages_per_edge: u32,
    pcf_radius: i32,
    // 1 when the resolve should sample the VSM for the primary directional
    // light, 0 to skip the branch entirely and always use the cascaded path.
    // (`enable` is a WESL reserved word, so the field is `sample_enable`.)
    sample_enable: u32,
    pad0: u32,
    pad1: u32,
    // Orthonormal clipmap basis of the light (see ReceiverProjection): the
    // world receiver is projected onto (right, up) for page addressing and onto
    // forward for the along-light reference depth.
    light_right: vec3<f32>,
    depth_scale: f32,
    light_up: vec3<f32>,
    depth_bias: f32,
    light_forward: vec3<f32>,
    pad2: f32,
}

@group(2) @binding(3) var<uniform> vsm_params: VsmResolveParams;

// Sentinel physical index for a virtual page that is not resident this frame
// (the twin of `VirtualPageTable::get` returning `None`).
const VSM_PAGE_UNMAPPED: u32 = 0xffffffffu;
// Returned by the twin when the receiver has no resident page, signalling the
// caller to keep the analytic directional visibility. Distinct from any lit
// fraction in [0, 1].
const VSM_MISS: f32 = -1.0;

// Number of clipmap levels, always >= 1 (mirrors level_count).
fn vsm_level_count() -> u32 {
    return max(vsm_params.levels, 1u);
}

// Pages along one edge of a level's resident window, always >= 1.
fn vsm_window_edge() -> i32 {
    return i32(max(vsm_params.pages_per_level_edge, 1u));
}

// World-space edge length of one texel at level (doubles each level).
fn vsm_texel_world_size(level: u32) -> f32 {
    let clamped = min(level, vsm_level_count() - 1u);
    return max(vsm_params.level0_texel_world_size, 1.0e-6) * exp2(f32(clamped));
}

// World-space edge length of one page at level (texel * page_size).
fn vsm_page_world_size(level: u32) -> f32 {
    return vsm_texel_world_size(level) * f32(max(vsm_params.page_size, 1u));
}

// Finest clip level whose texels are still dense enough for a receiver at
// view_distance. Twin of clipmap.rs's select_level.
fn vsm_select_level(view_distance: f32) -> u32 {
    let distance = max(view_distance, 0.0);
    let reference = max(vsm_params.level0_max_distance, 1.0e-6);
    let ratio = distance / reference;
    var level_i: i32;
    if (ratio <= 1.0) {
        level_i = 0;
    } else {
        level_i = i32(ceil(log2(ratio)));
    }
    let max_level = i32(vsm_level_count()) - 1;
    return u32(clamp(level_i, 0, max_level));
}

// Absolute world page a point lands on at level (camera independent).
fn vsm_world_page_coords(level: u32, world_xy: vec2<f32>) -> vec2<i32> {
    let pws = vsm_page_world_size(level);
    return vec2<i32>(
        i32(floor(world_xy.x / pws)),
        i32(floor(world_xy.y / pws)),
    );
}

// Bottom-left page of level's resident window, snapped to whole pages around
// the camera. Twin of ClipmapConfig::build_level's origin_page.
fn vsm_window_origin(level: u32, camera_xy: vec2<f32>) -> vec2<i32> {
    let pws = vsm_page_world_size(level);
    let center = vec2<i32>(
        i32(floor(camera_xy.x / pws)),
        i32(floor(camera_xy.y / pws)),
    );
    let half = vsm_window_edge() / 2;
    return center - vec2<i32>(half, half);
}

// Flat slot index of a window-local page at level, or -1 when the page lies
// outside the level's resident window. Identical addressing to
// vsm_page_mark.wesl so the physical index read corresponds to the flag marked.
fn vsm_window_slot(level: u32, world_page: vec2<i32>, camera_xy: vec2<f32>) -> i32 {
    let edge = vsm_window_edge();
    let local = world_page - vsm_window_origin(level, camera_xy);
    if (local.x < 0 || local.y < 0 || local.x >= edge || local.y >= edge) {
        return -1;
    }
    return i32(level) * edge * edge + local.y * edge + local.x;
}

// Physical pages along one edge of the square atlas grid, always >= 1.
fn vsm_physical_edge() -> u32 {
    return max(vsm_params.physical_pages_per_edge, 1u);
}

// Filtered VSM visibility for a world receiver under the primary directional
// light. Byte-for-byte addressing twin of the opaque resolve pass; the only
// change is sourcing the camera from this pass's own view uniform (the water
// pass has no `params` block). Returns a lit fraction in [0, 1], or `VSM_MISS`
// (< 0) on a page miss so the caller keeps the analytic visibility.
fn vsm_sample_directional(world_position: vec3<f32>) -> f32 {
    let camera_world = view.world_camera_position.xyz;
    let right = vsm_params.light_right;
    let up = vsm_params.light_up;
    let forward = vsm_params.light_forward;

    let world_xy = vec2<f32>(dot(world_position, right), dot(world_position, up));
    let camera_xy = vec2<f32>(dot(camera_world, right), dot(camera_world, up));
    let view_distance = length(world_position - camera_world);
    // Reproduce the caster-depth projection's NDC encoding exactly (abi.rs /
    // atlas::projection): clip.z = (world . forward) * depth_scale + depth_bias,
    // measured from the world-origin plane (depth_bias = 0.5), so the compare
    // against the stored depth is meaningful. `reference_depth <= stored => lit`.
    let reference_depth = clamp(
        dot(world_position, forward) * vsm_params.depth_scale + vsm_params.depth_bias,
        0.0,
        1.0,
    );

    let level = vsm_select_level(view_distance);
    let pws = vsm_page_world_size(level);
    let world_page = vsm_world_page_coords(level, world_xy);

    let slot = vsm_window_slot(level, world_page, camera_xy);
    if (slot < 0) {
        return VSM_MISS;
    }
    let physical = vsm_page_table[u32(slot)];
    if (physical == VSM_PAGE_UNMAPPED) {
        return VSM_MISS;
    }

    // Fraction of the sampled point within its virtual page, in [0, 1).
    let page_origin_world = vec2<f32>(f32(world_page.x), f32(world_page.y)) * pws;
    let in_page_frac = clamp((world_xy - page_origin_world) / pws, vec2<f32>(0.0), vec2<f32>(1.0));

    // The physical page's tile in the square atlas grid.
    let edge_phys = vsm_physical_edge();
    let tile = vec2<f32>(
        f32(physical % edge_phys),
        f32(physical / edge_phys),
    );
    let inv_edge_phys = 1.0 / f32(edge_phys);
    let page_size = f32(max(vsm_params.page_size, 1u));
    // Half a physical texel, in page-local fraction units; the PCF step and the
    // clamp that keeps taps inside this page so the filter cannot bleed across a
    // seam into the unrelated pages packed beside it in the atlas.
    let texel_frac = 1.0 / page_size;
    let half_texel = 0.5 * texel_frac;

    let r = max(vsm_params.pcf_radius, 0);
    var lit = 0.0;
    var taps = 0.0;
    for (var y: i32 = -r; y <= r; y = y + 1) {
        for (var x: i32 = -r; x <= r; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_frac;
            let tap_frac = clamp(
                in_page_frac + offset,
                vec2<f32>(half_texel),
                vec2<f32>(1.0 - half_texel),
            );
            let atlas_uv = (tile + tap_frac) * inv_edge_phys;
            let stored = textureSampleLevel(vsm_physical_atlas, vsm_physical_sampler, atlas_uv, 0.0).r;
            if (reference_depth <= stored) {
                lit = lit + 1.0;
            }
            taps = taps + 1.0;
        }
    }
    return lit / taps;
}

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

// Image-based ambient term, mirroring the opaque `shading_resolve` indirect
// path: a split-sum specular reflection plus `SH` irradiance diffuse when the
// environment carries image-based data, falling back to the flat ambient
// radiance otherwise. Shared verbatim by every §5 frontend.
fn water_ibl(surface: SurfaceSample, frame: ShadingFrame) -> vec3<f32> {
    let env = light_env[0];
    let occlusion = clamp(surface.ambient_occlusion, 0.0, 1.0);
    let n = brdf_normalize_or(frame.normal, vec3<f32>(0.0, 1.0, 0.0));
    let v = brdf_normalize_or(frame.view, n);
    let n_dot_v = max(dot(n, v), 1.0e-4);
    let roughness = clamp(surface.perceptual_roughness, 0.0, 1.0);
    let metallic = clamp(surface.metallic, 0.0, 1.0);
    let reflectance = clamp(surface.reflectance, 0.0, 1.0);
    let f0 = mix(vec3<f32>(0.16 * reflectance * reflectance), surface.base_color, metallic);
    let dfg = env_brdf_approx(n_dot_v, roughness);
    let spec_weight = (f0 * dfg.x + vec3<f32>(dfg.y)) * occlusion;
    if environment_has_image_based(env) {
        let irradiance = max(sh_irradiance(env, n), vec3<f32>(0.0));
        let diffuse = surface.base_color * (1.0 - metallic) * irradiance * INV_PI;
        let reflection = reflect(-v, n);
        let prefiltered = max(sh_radiance(env, reflection), vec3<f32>(0.0));
        let env_specular = prefiltered * spec_weight;
        return diffuse * occlusion + env_specular;
    }
    return surface.base_color * env.ambient * (occlusion * (1.0 - metallic));
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

    // Accumulate the engine's full analytic light table (all directional +
    // punctual lights) through the shared single-layer-water lobe, then add the
    // image-based ambient term. Identical to the opaque path's direct+indirect
    // split; `water_direct` already folds in each light's analytic visibility.
    let env = light_env[0];
    var lit = vec3<f32>(0.0);
    for (var i = 0u; i < env.directional_count; i = i + 1u) {
        let s = sample_directional_light(directional_lights[i]);
        var dl: DirectLightSample;
        dl.direction = s.direction;
        dl.illuminance = s.illuminance;
        dl.visibility = s.visibility;
        // Primary directional light (index 0) is the VSM caster. A resident
        // page replaces the analytic visibility with the filtered shadow; a
        // miss (`VSM_MISS` < 0) leaves the analytic value untouched, so the
        // water pass needs no cascaded fallback of its own.
        if i == 0u && vsm_params.sample_enable != 0u {
            let v = vsm_sample_directional(in.world_position);
            if v >= 0.0 {
                dl.visibility = s.visibility * v;
            }
        }
        lit = lit + water_direct(surface, frame, dl);
    }
    for (var i = 0u; i < env.punctual_count; i = i + 1u) {
        let s = sample_punctual_light(punctual_lights[i], in.world_position);
        if s.valid > 0.5 {
            var dl: DirectLightSample;
            dl.direction = s.direction;
            dl.illuminance = s.illuminance;
            dl.visibility = s.visibility;
            lit = lit + water_direct(surface, frame, dl);
        }
    }
    lit = lit + water_ibl(surface, frame);
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
