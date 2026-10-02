// GPU twin of `prism_render_shading::tonemap` (AAA display tone mapping:
// Reinhard, extended Reinhard, ACES Narkowicz, ACES fitted and AgX).
// Self-contained (no intra-crate imports, matching exposure.wesl / ssgi.wesl)
// so the WESL compile test can validate it standalone. Every function mirrors
// the CPU golden arm-for-arm on vec3<f32>, using explicit row dot products for
// the 3x3 transforms so the maths matches the row-major CPU matrices exactly.

const LUMINANCE_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

// Rec. 709 relative luminance of a linear RGB radiance sample.
fn tonemap_luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, LUMINANCE_WEIGHTS);
}

// Row-major 3x3 times a column vector, expressed as three row dot products so
// it matches the CPU golden's `mat3_mul_vec3`.
fn mat3_mul_vec3(row0: vec3<f32>, row1: vec3<f32>, row2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(row0, v), dot(row1, v), dot(row2, v));
}

// --- Reinhard ---

// Reinhard global tone map x / (1 + x) per channel (negatives clamped away).
fn tonemap_reinhard(rgb: vec3<f32>) -> vec3<f32> {
    let x = max(rgb, vec3<f32>(0.0));
    return x / (vec3<f32>(1.0) + x);
}

// Extended Reinhard with a white point: x * (1 + x / white^2) / (1 + x).
// Non-positive white degrades to plain Reinhard.
fn tonemap_reinhard_extended(rgb: vec3<f32>, white: f32) -> vec3<f32> {
    if (white <= 0.0) {
        return tonemap_reinhard(rgb);
    }
    let x = max(rgb, vec3<f32>(0.0));
    let white_sq = white * white;
    return x * (vec3<f32>(1.0) + x / vec3<f32>(white_sq)) / (vec3<f32>(1.0) + x);
}

// --- ACES (Narkowicz) ---

// Krzysztof Narkowicz's cheap ACES fit per channel, clamped to [0, 1].
fn tonemap_aces_narkowicz(rgb: vec3<f32>) -> vec3<f32> {
    let x = max(rgb, vec3<f32>(0.0));
    let numerator = x * (2.51 * x + vec3<f32>(0.03));
    let denominator = x * (2.43 * x + vec3<f32>(0.59)) + vec3<f32>(0.14);
    return clamp(numerator / denominator, vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- ACES (Hill fitted) ---

// Hill's RRTAndODTFit rational curve, per channel in ACEScg space.
fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
    let b = v * (0.983729 * v + vec3<f32>(0.432951)) + vec3<f32>(0.238081);
    return a / b;
}

// Stephen Hill's ACES fit: input matrix, RRTAndODTFit, output matrix, clamp.
fn tonemap_aces_fitted(rgb: vec3<f32>) -> vec3<f32> {
    let clamped = max(rgb, vec3<f32>(0.0));
    let acescg = mat3_mul_vec3(
        vec3<f32>(0.59719, 0.35458, 0.04823),
        vec3<f32>(0.07600, 0.90834, 0.01566),
        vec3<f32>(0.02840, 0.13383, 0.83777),
        clamped,
    );
    let fitted = rrt_and_odt_fit(acescg);
    let srgb = mat3_mul_vec3(
        vec3<f32>(1.60475, -0.53108, -0.07367),
        vec3<f32>(-0.10208, 1.10813, -0.00605),
        vec3<f32>(-0.00327, -0.07276, 1.07602),
        fitted,
    );
    return clamp(srgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- AgX (Troy Sobotka minimal) ---

// AgX input transform: matrix into working space, log2 encode normalised into
// the [AGX_MIN_EV, AGX_MAX_EV] window mapped to [0, 1].
fn agx_encode(rgb: vec3<f32>) -> vec3<f32> {
    let clamped = max(rgb, vec3<f32>(0.0));
    let working = mat3_mul_vec3(
        vec3<f32>(0.84247905, 0.0784336, 0.079223745),
        vec3<f32>(0.042328242, 0.87846863, 0.07916613),
        vec3<f32>(0.042375654, 0.0784336, 0.879143),
        clamped,
    );
    let safe = max(working, vec3<f32>(1.0e-10));
    let span = AGX_MAX_EV - AGX_MIN_EV;
    let log_ev = clamp(log2(safe), vec3<f32>(AGX_MIN_EV), vec3<f32>(AGX_MAX_EV));
    return (log_ev - vec3<f32>(AGX_MIN_EV)) / vec3<f32>(span);
}

// 6th-order polynomial sigmoid (agxDefaultContrastApprox) per channel.
fn agx_contrast_sigmoid(v: vec3<f32>) -> vec3<f32> {
    let x2 = v * v;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v
        + 0.4298 * x2
        + 0.1191 * v
        - vec3<f32>(0.00232);
}

// AgX look grade pow(v * slope + offset, power) then a saturation blend about
// luminance. Neutral parameters (slope 1, offset 0, power 1, sat 1) return v.
fn agx_apply_look(
    v: vec3<f32>,
    slope: vec3<f32>,
    offset: vec3<f32>,
    power: vec3<f32>,
    saturation: f32,
) -> vec3<f32> {
    let base = max(v * slope + offset, vec3<f32>(0.0));
    let graded = pow(base, power);
    let luma = tonemap_luminance(graded);
    return vec3<f32>(luma) + saturation * (graded - vec3<f32>(luma));
}

// Full AgX pipeline with an explicit look grade, then a [0, 1] clamp.
fn tonemap_agx_with_look(
    rgb: vec3<f32>,
    slope: vec3<f32>,
    offset: vec3<f32>,
    power: vec3<f32>,
    saturation: f32,
) -> vec3<f32> {
    let encoded = agx_encode(rgb);
    let contrast = agx_contrast_sigmoid(encoded);
    let graded = agx_apply_look(contrast, slope, offset, power, saturation);
    let out = mat3_mul_vec3(
        vec3<f32>(1.196879, -0.09802088, -0.09902974),
        vec3<f32>(-0.052896854, 1.1519032, -0.098961174),
        vec3<f32>(-0.052971635, -0.09804345, 1.1510737),
        graded,
    );
    return clamp(out, vec3<f32>(0.0), vec3<f32>(1.0));
}

// AgX with the neutral (default) look.
fn tonemap_agx(rgb: vec3<f32>) -> vec3<f32> {
    return tonemap_agx_with_look(
        rgb,
        vec3<f32>(1.0),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
        1.0,
    );
}

// Unified dispatch mirroring the CPU `apply_tonemap`. Operator codes follow the
// CPU enum order: 0 Reinhard, 1 ReinhardExtended, 2 AcesNarkowicz,
// 3 AcesFitted, 4 AgX.
fn apply_tonemap(op: u32, rgb: vec3<f32>, reinhard_white_point: f32) -> vec3<f32> {
    if (op == 1u) {
        return tonemap_reinhard_extended(rgb, reinhard_white_point);
    } else if (op == 2u) {
        return tonemap_aces_narkowicz(rgb);
    } else if (op == 3u) {
        return tonemap_aces_fitted(rgb);
    } else if (op == 4u) {
        return tonemap_agx(rgb);
    }
    return tonemap_reinhard(rgb);
}

// ABI twin of the Rust `GpuTonemapParams` (16 bytes): operator selector, the
// extended-Reinhard white point and the framebuffer extent.
struct GpuTonemapParams {
    // Operator selector: 0 Reinhard, 1 ReinhardExtended, 2 AcesNarkowicz,
    // 3 AcesFitted, 4 AgX — matching the CPU enum order and `apply_tonemap`.
    // `operator` is a WGSL reserved word, so the shader field is `operator_code`;
    // the byte layout still mirrors the Rust `GpuTonemapParams::operator`.
    operator_code: u32,
    // White point for the extended-Reinhard operator; ignored otherwise.
    reinhard_white_point: f32,
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<u32>,
}

// The pre-exposed, resolved linear HDR scene colour this pass maps.
@group(0) @binding(0) var tonemap_scene_color: texture_2d<f32>;
// The tone-mapped output; copied back over `scene_color` after the dispatch.
@group(0) @binding(1) var tonemap_out: texture_storage_2d<rgba16float, write>;

var<immediate> tonemap_immediate: GpuTonemapParams;

// One invocation per pixel. Applies the golden operator to the pre-exposed
// linear HDR radiance and writes the display-referred texel (alpha preserved).
@compute @workgroup_size(8, 8, 1)
fn tonemap_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(tonemap_immediate.screen_size.x);
    let height = i32(tonemap_immediate.screen_size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }

    let sample = textureLoad(tonemap_scene_color, pc, 0);
    let mapped = apply_tonemap(
        tonemap_immediate.operator_code,
        sample.rgb,
        tonemap_immediate.reinhard_white_point,
    );
    textureStore(tonemap_out, pc, vec4<f32>(mapped, sample.a));
}
