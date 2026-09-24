//! GPU shadow-visibility core: the WESL twin of the CPU golden reference in
//! `prism_render_shading::shadow`.
//!
//! Every function below mirrors, line for line, one function of the Rust
//! shadow core so the analytic `visibility` term the resolve pass multiplies
//! into a light's `DirectLightSample::visibility` is computed identically on
//! the CPU reference and on the GPU.  The pieces are:
//!
//! * column-major `transform_point` for the world -> light-clip projection,
//! * the PSSM (`cascaded shadow map`) split scheme, cascade selection and the
//!   cross-cascade blend for directional lights,
//! * normal-offset and slope-scaled depth bias (shadow acne / peter-panning),
//! * a manual PCF box filter and PCSS contact-hardening filter over the atlas,
//! * the directional (cascaded) and point (cube distance map) orchestrators.
//!
//! Conventions shared with the Rust core:
//! * wgpu NDC has `z` in `[0, 1]` (near = 0, far = 1); shadow UV flips V.
//! * A texel is **lit** when `reference_depth <= stored_depth`; the caller
//!   folds any depth bias into `reference_depth` before filtering.
//! * Samples outside the `[0, 1]` UV square read the far value (`1.0`) so
//!   unmapped regions stay lit.
//! * The point-light atlas stores a **linear** distance normalized by the
//!   light range, not a projected NDC depth.
//!
//! The module is intentionally self-contained (no `import`) so it compiles
//! standalone in a WESL unit test.  The shadow atlas is a single
//! `texture_2d_array<f32>` whose `.r` channel holds the stored depth/distance;
//! directional cascades and cube faces are addressed as array layers, matching
//! the CPU `ShadowDepthSampler`'s `layer` parameter.

const MAX_CASCADE_COUNT: u32 = 4u;
const SHADOW_LIT: f32 = 1.0;
const SHADOW_FAR: f32 = 1.0;

// Standard cube-map face layer indices (OpenGL convention: +X, -X, +Y, -Y,
// +Z, -Z), matching `shadow/point.rs`.
const FACE_POS_X: u32 = 0u;
const FACE_NEG_X: u32 = 1u;
const FACE_POS_Y: u32 = 2u;
const FACE_NEG_Y: u32 = 3u;
const FACE_POS_Z: u32 = 4u;
const FACE_NEG_Z: u32 = 5u;

// Which soft-shadow filter the directional evaluator applies per cascade.
const SHADOW_FILTER_PCF: u32 = 0u;
const SHADOW_FILTER_PCSS: u32 = 1u;

// ---------------------------------------------------------------------------
// Resource bindings
// ---------------------------------------------------------------------------
// The shadow atlas: one array texture whose `.r` channel stores either the
// wgpu NDC depth (directional / spot cascades) or the light-range-normalized
// linear distance (point-light cube faces).  Filtered with a plain linear
// sampler; the depth comparison is done manually in-shader to stay a
// byte-for-byte twin of the CPU reference.
@group(0) @binding(0) var shadow_atlas: texture_2d_array<f32>;
@group(0) @binding(1) var shadow_atlas_sampler: sampler;

// ---------------------------------------------------------------------------
// math.rs
// ---------------------------------------------------------------------------

// Multiplies the column-major `matrix` by the homogeneous point
// `(point, 1.0)`, returning the full `vec4` (perspective divide not applied).
// `mat4x4<f32>` is column-major, so this is exactly `matrix * vec4(point, 1)`.
fn transform_point(matrix: mat4x4<f32>, point: vec3<f32>) -> vec4<f32> {
    return matrix * vec4<f32>(point, 1.0);
}

// Multiplies the column-major `matrix` by the homogeneous direction
// `(direction, 0.0)`; the translation column is ignored.
fn transform_direction(matrix: mat4x4<f32>, direction: vec3<f32>) -> vec3<f32> {
    return (matrix * vec4<f32>(direction, 0.0)).xyz;
}

// ---------------------------------------------------------------------------
// cascade.rs
// ---------------------------------------------------------------------------

// Ordered far-plane view-space distances for each cascade plus the shared near
// plane, mirroring `CascadeSplits`.
struct CascadeSplits {
    near: f32,
    distances: array<f32, 4>,
    count: u32,
}

// Near boundary of cascade `index`: its predecessor's far plane, or the shared
// near plane for cascade 0.  Out-of-range indices clamp to the last cascade.
fn cascade_near(splits: CascadeSplits, index: u32) -> f32 {
    let last = max(splits.count, 1u) - 1u;
    let i = min(index, last);
    if (i == 0u) {
        return splits.near;
    }
    return splits.distances[i - 1u];
}

// Far boundary of cascade `index`, clamped to the last active cascade.
fn cascade_far(splits: CascadeSplits, index: u32) -> f32 {
    let last = max(splits.count, 1u) - 1u;
    let i = min(index, last);
    return splits.distances[i];
}

// PSSM logarithmic/uniform blend split scheme (Zhang et al.).  `lambda`
// blends the purely logarithmic (1.0) and purely uniform (0.0) distributions.
// The last cascade is pinned to `far` so out-of-range reads clamp cleanly.
fn compute_cascade_splits(near_in: f32, far_in: f32, count_in: u32, lambda_in: f32) -> CascadeSplits {
    let count = clamp(count_in, 1u, MAX_CASCADE_COUNT);
    let lambda = clamp(lambda_in, 0.0, 1.0);
    let near = max(near_in, 1.0e-4);
    let far = max(far_in, near + 1.0e-4);

    let ratio = far / near;
    let range = far - near;
    let inv_count = 1.0 / f32(count);

    var distances = array<f32, 4>(far, far, far, far);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let fraction = f32(i + 1u) * inv_count;
        let logarithmic = near * pow(ratio, fraction);
        let uniform = near + range * fraction;
        distances[i] = lambda * logarithmic + (1.0 - lambda) * uniform;
    }
    distances[count - 1u] = far;

    var splits: CascadeSplits;
    splits.near = near;
    splits.distances = distances;
    splits.count = count;
    return splits;
}

// Selects the first cascade whose far plane covers `view_depth`, clamping to
// the last cascade for anything beyond the final split.
fn select_cascade(view_depth: f32, splits: CascadeSplits) -> u32 {
    for (var i: u32 = 0u; i < splits.count; i = i + 1u) {
        if (view_depth <= splits.distances[i]) {
            return i;
        }
    }
    return splits.count - 1u;
}

// Cross-cascade blend weight in `[0, 1]`: 0 outside the blend band near the
// cascade's far plane, ramping to 1 at the boundary so the seam to the next
// cascade can be cross-faded.  Returns 0 for the last cascade.
fn cascade_blend_weight(view_depth: f32, splits: CascadeSplits, index: u32, band_fraction_in: f32) -> f32 {
    if (index + 1u >= splits.count) {
        return 0.0;
    }
    let band_fraction = clamp(band_fraction_in, 0.0, 1.0);
    if (band_fraction <= 0.0) {
        return 0.0;
    }

    let near = cascade_near(splits, index);
    let far = cascade_far(splits, index);
    let cascade_range = max(far - near, 1.0e-4);
    let band = cascade_range * band_fraction;
    let band_start = far - band;

    if (view_depth <= band_start) {
        return 0.0;
    }
    return clamp((view_depth - band_start) / band, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// bias.rs
// ---------------------------------------------------------------------------

// Offsets `position` along the surface `normal` by `scale * texel_world_size`,
// boosted toward grazing angles by `sin/cos` (clamped), so nearly tangent
// surfaces are lifted the most.  Applied *before* the light-space projection.
fn apply_normal_offset(position: vec3<f32>, normal: vec3<f32>, texel_world_size: f32, scale: f32, n_dot_l: f32) -> vec3<f32> {
    let cos_theta = clamp(n_dot_l, 1.0e-2, 1.0);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let grazing = clamp(sin_theta / cos_theta, 0.0, 8.0);
    let magnitude = texel_world_size * scale * (1.0 + grazing);
    return position + normal * magnitude;
}

// Slope-scaled constant depth bias in normalized shadow-depth units:
// `const_bias + slope_bias * tan(theta)`, clamped to `max_bias`.  `tan(theta)`
// is `sqrt(1 - c^2) / c` from the clamped cosine `n_dot_l`.
fn slope_scaled_depth_bias(n_dot_l: f32, const_bias: f32, slope_bias: f32, max_bias: f32) -> f32 {
    let cos_theta = clamp(n_dot_l, 1.0e-3, 1.0);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let tan_theta = sin_theta / cos_theta;
    let bias = const_bias + slope_bias * tan_theta;
    return clamp(bias, min(const_bias, max_bias), max_bias);
}

// ---------------------------------------------------------------------------
// filter.rs
// ---------------------------------------------------------------------------

// Reads the stored depth/distance at `uv` on atlas `layer`, returning the far
// value (`1.0`) for `uv` outside `[0, 1]^2` so off-map regions stay lit.
// Mirrors the `ShadowDepthSampler` contract of the CPU core.
fn sample_shadow_depth(layer: u32, uv: vec2<f32>) -> f32 {
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return SHADOW_FAR;
    }
    return textureSampleLevel(shadow_atlas, shadow_atlas_sampler, uv, i32(layer), 0.0).r;
}

// Fraction of `(2 * radius + 1)^2` box-grid taps that pass the depth test, i.e.
// the lit fraction in `[0, 1]`.  `radius == 0` degenerates to a single hard tap.
fn pcf_visibility(layer: u32, center_uv: vec2<f32>, reference_depth: f32, texel_size: vec2<f32>, radius: i32) -> f32 {
    let r = max(radius, 0);
    var lit = 0.0;
    var taps = 0.0;
    for (var y: i32 = -r; y <= r; y = y + 1) {
        for (var x: i32 = -r; x <= r; x = x + 1) {
            let uv = vec2<f32>(
                center_uv.x + f32(x) * texel_size.x,
                center_uv.y + f32(y) * texel_size.y,
            );
            let stored = sample_shadow_depth(layer, uv);
            if (reference_depth <= stored) {
                lit = lit + 1.0;
            }
            taps = taps + 1.0;
        }
    }
    return lit / taps;
}

// Result of the PCSS blocker search.
struct BlockerSearch {
    average_depth: f32,
    blocker_fraction: f32,
}

// Averages the depth of blockers (taps strictly closer to the light than
// `reference_depth`) in a `(2 * radius + 1)^2` box.  With no blocker, returns
// `average_depth = 1.0` (far) and `blocker_fraction = 0.0`.
fn blocker_search(layer: u32, center_uv: vec2<f32>, reference_depth: f32, texel_size: vec2<f32>, radius: i32) -> BlockerSearch {
    let r = max(radius, 0);
    var sum = 0.0;
    var blockers = 0.0;
    var taps = 0.0;
    for (var y: i32 = -r; y <= r; y = y + 1) {
        for (var x: i32 = -r; x <= r; x = x + 1) {
            let uv = vec2<f32>(
                center_uv.x + f32(x) * texel_size.x,
                center_uv.y + f32(y) * texel_size.y,
            );
            let stored = sample_shadow_depth(layer, uv);
            if (stored < reference_depth) {
                sum = sum + stored;
                blockers = blockers + 1.0;
            }
            taps = taps + 1.0;
        }
    }
    var result: BlockerSearch;
    if (blockers <= 0.0) {
        result.average_depth = 1.0;
        result.blocker_fraction = 0.0;
    } else {
        result.average_depth = sum / blockers;
        result.blocker_fraction = blockers / taps;
    }
    return result;
}

// Tunables for the three PCSS stages.
struct PcssConfig {
    search_radius: i32,
    light_size_uv: f32,
    min_filter_radius: i32,
    max_filter_radius: i32,
}

// Percentage-closer soft shadows: estimates a penumbra width from the average
// blocker depth via the similar-triangles ratio, converts it to an integer
// texel radius bounded by the config, then runs a variable-radius PCF.
fn pcss_visibility(layer: u32, center_uv: vec2<f32>, reference_depth: f32, texel_size: vec2<f32>, config: PcssConfig) -> f32 {
    let search = blocker_search(layer, center_uv, reference_depth, texel_size, config.search_radius);
    if (search.blocker_fraction <= 0.0) {
        return SHADOW_LIT;
    }

    let blocker = max(search.average_depth, 1.0e-4);
    let penumbra_ratio = max((reference_depth - blocker) / blocker, 0.0);
    let penumbra_uv = penumbra_ratio * config.light_size_uv;
    let texel = max(min(texel_size.x, texel_size.y), 1.0e-6);
    let radius_texels = i32(round(penumbra_uv / texel));
    let radius = clamp(radius_texels, max(config.min_filter_radius, 0), config.max_filter_radius);

    return pcf_visibility(layer, center_uv, reference_depth, texel_size, radius);
}

// ---------------------------------------------------------------------------
// directional.rs
// ---------------------------------------------------------------------------

// Tunables for the directional shadow bias/blend, shared across cascades.
struct DirectionalShadowConfig {
    normal_offset_scale: f32,
    const_depth_bias: f32,
    slope_depth_bias: f32,
    max_depth_bias: f32,
    cascade_blend_fraction: f32,
    // SHADOW_FILTER_PCF or SHADOW_FILTER_PCSS.
    filter_kind: u32,
    // PCF box half-extent (used when filter_kind == PCF).
    pcf_radius: i32,
    // PCSS parameters (used when filter_kind == PCSS).
    pcss: PcssConfig,
}

// All per-fragment and per-light inputs needed to evaluate a directional
// shadow.  Matrices and texel sizes are indexed per cascade.
struct DirectionalShadowInput {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    n_dot_l: f32,
    view_depth: f32,
    splits: CascadeSplits,
    light_view_projections: array<mat4x4<f32>, 4>,
    texel_world_sizes: array<f32, 4>,
    texel_uv_size: vec2<f32>,
}

// Projects the (normal-offset) surface point into `cascade`'s light clip
// space, applies the slope-scaled depth bias and filters.  Returns 1.0 (lit)
// when the point projects behind the light or outside the cascade's `[0, 1]`
// UV / depth range.
fn evaluate_single_cascade(input: DirectionalShadowInput, config: DirectionalShadowConfig, cascade_in: u32) -> f32 {
    let last = max(input.splits.count, 1u) - 1u;
    let cascade = min(cascade_in, last);

    let offset_position = apply_normal_offset(
        input.world_position,
        input.world_normal,
        input.texel_world_sizes[cascade],
        config.normal_offset_scale,
        input.n_dot_l,
    );

    let clip = transform_point(input.light_view_projections[cascade], offset_position);
    // Orthographic directional projection has w == 1, but guard the divide.
    if (clip.w <= 0.0) {
        return SHADOW_LIT;
    }
    let inv_w = 1.0 / clip.w;
    let ndc = clip.xyz * inv_w;

    // wgpu NDC: x,y in [-1, 1], z in [0, 1]; map to shadow UV with flipped V.
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return SHADOW_LIT;
    }
    if (ndc.z < 0.0 || ndc.z > 1.0) {
        return SHADOW_LIT;
    }

    let bias = slope_scaled_depth_bias(
        input.n_dot_l,
        config.const_depth_bias,
        config.slope_depth_bias,
        config.max_depth_bias,
    );
    let reference_depth = ndc.z - bias;

    if (config.filter_kind == SHADOW_FILTER_PCSS) {
        return pcss_visibility(cascade, uv, reference_depth, input.texel_uv_size, config.pcss);
    }
    return pcf_visibility(cascade, uv, reference_depth, input.texel_uv_size, config.pcf_radius);
}

// Evaluates the directional-light visibility in `[0, 1]`.  1.0 means fully lit.
// Surfaces the light cannot reach in shadow space fall back to lit so they use
// ordinary `n·l` shading rather than turning black.
fn evaluate_directional_shadow(input: DirectionalShadowInput, config: DirectionalShadowConfig) -> f32 {
    let cascade = select_cascade(input.view_depth, input.splits);
    let primary = evaluate_single_cascade(input, config, cascade);

    let blend = cascade_blend_weight(input.view_depth, input.splits, cascade, config.cascade_blend_fraction);
    if (blend <= 0.0) {
        return primary;
    }

    let next = evaluate_single_cascade(input, config, cascade + 1u);
    return primary * (1.0 - blend) + next * blend;
}

// ---------------------------------------------------------------------------
// point.rs
// ---------------------------------------------------------------------------

// Cube-map face selection result: face layer (0..6) and in-face UV.
struct CubeFace {
    face: u32,
    uv: vec2<f32>,
}

// Bias/filter tunables for point-light shadows.
struct PointShadowConfig {
    const_bias: f32,
    slope_bias: f32,
    max_bias: f32,
    pcf_radius: i32,
    texel_uv_size: vec2<f32>,
}

// Per-fragment inputs for a point-light shadow lookup.
struct PointShadowInput {
    world_position: vec3<f32>,
    light_position: vec3<f32>,
    range: f32,
    n_dot_l: f32,
}

// Selects the cube-map face (layer 0..6) and in-face UV in `[0, 1]^2` for a
// world-space direction from the light to the fragment, using the OpenGL
// cube-map convention.  Mirrors `cube_face_and_uv` in `point.rs`.
fn cube_face_and_uv(direction: vec3<f32>) -> CubeFace {
    let ax = abs(direction.x);
    let ay = abs(direction.y);
    let az = abs(direction.z);

    var face: u32;
    var sc: f32;
    var tc: f32;
    var ma: f32;
    if (ax >= ay && ax >= az) {
        if (direction.x >= 0.0) {
            face = FACE_POS_X;
            sc = -direction.z;
            tc = -direction.y;
        } else {
            face = FACE_NEG_X;
            sc = direction.z;
            tc = -direction.y;
        }
        ma = ax;
    } else if (ay >= az) {
        if (direction.y >= 0.0) {
            face = FACE_POS_Y;
            sc = direction.x;
            tc = direction.z;
        } else {
            face = FACE_NEG_Y;
            sc = direction.x;
            tc = -direction.z;
        }
        ma = ay;
    } else {
        if (direction.z >= 0.0) {
            face = FACE_POS_Z;
            sc = direction.x;
            tc = -direction.y;
        } else {
            face = FACE_NEG_Z;
            sc = -direction.x;
            tc = -direction.y;
        }
        ma = az;
    }

    let inv_ma = 1.0 / max(ma, 1.0e-8);
    let u = 0.5 * (sc * inv_ma + 1.0);
    let v = 0.5 * (tc * inv_ma + 1.0);
    var result: CubeFace;
    result.face = face;
    result.uv = vec2<f32>(clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
    return result;
}

// Evaluates point-light visibility in `[0, 1]`, sampling the distance cube map
// through the atlas (layer = cube face).  A fragment beyond the light's range
// reads off-map as far and stays lit.  Mirrors `evaluate_point_shadow`.
fn evaluate_point_shadow(input: PointShadowInput, config: PointShadowConfig) -> f32 {
    let light_to_frag = input.world_position - input.light_position;
    let distance = length(light_to_frag);
    let range = max(input.range, 1.0e-4);
    let reference = clamp(distance / range, 0.0, 1.0);

    let cube = cube_face_and_uv(light_to_frag);
    let bias = slope_scaled_depth_bias(input.n_dot_l, config.const_bias, config.slope_bias, config.max_bias);

    return pcf_visibility(cube.face, cube.uv, reference - bias, config.texel_uv_size, config.pcf_radius);
}

// ---------------------------------------------------------------------------
// Standalone compilation self-test entry point
// ---------------------------------------------------------------------------
// This module is a shared library with no draw entry point of its own; the
// resolve pass calls the functions above.  A trivial compute entry keeps the
// whole call graph reachable so the standalone WESL unit test validates every
// function (and the atlas bindings) end to end rather than compiling dead code.
@group(0) @binding(2) var<storage, read_write> shadow_selftest_out: array<f32>;

@compute @workgroup_size(1)
fn shadow_selftest() {
    var dir_input: DirectionalShadowInput;
    dir_input.world_position = vec3<f32>(0.0, 0.0, 0.0);
    dir_input.world_normal = vec3<f32>(0.0, 1.0, 0.0);
    dir_input.n_dot_l = 1.0;
    dir_input.view_depth = 1.0;
    dir_input.splits = compute_cascade_splits(1.0, 100.0, 4u, 0.5);
    let identity = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    dir_input.light_view_projections = array<mat4x4<f32>, 4>(identity, identity, identity, identity);
    dir_input.texel_world_sizes = array<f32, 4>(0.01, 0.02, 0.04, 0.08);
    dir_input.texel_uv_size = vec2<f32>(1.0 / 2048.0, 1.0 / 2048.0);

    var dir_config: DirectionalShadowConfig;
    dir_config.normal_offset_scale = 2.0;
    dir_config.const_depth_bias = 0.0005;
    dir_config.slope_depth_bias = 0.002;
    dir_config.max_depth_bias = 0.02;
    dir_config.cascade_blend_fraction = 0.1;
    dir_config.filter_kind = SHADOW_FILTER_PCF;
    dir_config.pcf_radius = 1;
    dir_config.pcss = PcssConfig(2, 0.01, 1, 8);

    let directional = evaluate_directional_shadow(dir_input, dir_config);

    var pt_input: PointShadowInput;
    pt_input.world_position = vec3<f32>(10.0, 0.0, 0.0);
    pt_input.light_position = vec3<f32>(0.0, 0.0, 0.0);
    pt_input.range = 100.0;
    pt_input.n_dot_l = 1.0;

    var pt_config: PointShadowConfig;
    pt_config.const_bias = 0.001;
    pt_config.slope_bias = 0.002;
    pt_config.max_bias = 0.02;
    pt_config.pcf_radius = 1;
    pt_config.texel_uv_size = vec2<f32>(1.0 / 512.0, 1.0 / 512.0);

    let point = evaluate_point_shadow(pt_input, pt_config);

    shadow_selftest_out[0] = directional;
    shadow_selftest_out[1] = point;
}
