// GPU twin of `prism_render_shading::cas` (AMD FidelityFX Contrast-Adaptive
// Sharpening, the "no-scaling" cross kernel over a 3x3 neighbourhood on the
// display-referred, tone-mapped [0, 1] buffer).
// Self-contained (no intra-crate imports, matching color_grade.wesl /
// bloom.wesl / exposure.wesl) so the WESL compile test can validate it
// standalone. Every function mirrors the CPU golden arm-for-arm: WGSL `min`,
// `max`, `sqrt` and `clamp` on vec3 are componentwise, matching the per-channel
// Rust, and lerps are written `a + (b - a) * t` rather than `mix` so the GPU
// twin stays bit-identical with the golden.

// Neighbourhood is row-major (a b c / d e f / g h i); index 4 is the centre e.

fn cas_min3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return min(min(a, b), c);
}

fn cas_max3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return max(max(a, b), c);
}

// FidelityFX doubled soft minimum: min of the five-tap cross (b, d, e, f, h)
// plus min of the four corners (a, c, g, i).
fn cas_soft_min(n: array<vec3<f32>, 9>) -> vec3<f32> {
    let cross = cas_min3(cas_min3(n[3], n[4], n[5]), n[1], n[7]);
    let corners = cas_min3(cas_min3(n[0], n[2], n[6]), n[8], n[8]);
    return cross + corners;
}

// FidelityFX doubled soft maximum (cross max plus corner max).
fn cas_soft_max(n: array<vec3<f32>, 9>) -> vec3<f32> {
    let cross = cas_max3(cas_max3(n[3], n[4], n[5]), n[1], n[7]);
    let corners = cas_max3(cas_max3(n[0], n[2], n[6]), n[8], n[8]);
    return cross + corners;
}

// Adaptive amplitude sqrt(saturate(min(soft_min, 2 - soft_max) / soft_max)),
// zero where soft_max is non-positive (pure black), mirroring the CPU guard.
fn cas_amplitude(soft_min: vec3<f32>, soft_max: vec3<f32>) -> vec3<f32> {
    let limited = min(soft_min, vec3<f32>(2.0) - soft_max);
    let amp = clamp(limited / soft_max, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(sqrt(amp), vec3<f32>(0.0), soft_max <= vec3<f32>(0.0));
}

// Shape the amplitude into the negative cross weight w. sharpness in [0, 1]
// maps the peak from -1/8 (gentle) to -1/5 (strong).
fn cas_weight(amplitude: vec3<f32>, sharpness: f32) -> vec3<f32> {
    let s = clamp(sharpness, 0.0, 1.0);
    let denom = 8.0 + (5.0 - 8.0) * s;
    let peak = -1.0 / denom;
    return amplitude * peak;
}

// Blend cross taps against the centre: saturate(((b+d+f+h)*w + e) / (1 + 4w)).
fn cas_blend(b: vec3<f32>, d: vec3<f32>, e: vec3<f32>, f: vec3<f32>, h: vec3<f32>, w: vec3<f32>) -> vec3<f32> {
    let rcp = vec3<f32>(1.0) / (vec3<f32>(1.0) + 4.0 * w);
    return clamp(((b + d + f + h) * w + e) * rcp, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Full CAS kernel over the 3x3 neighbourhood at the given sharpness.
fn cas_sharpen(n: array<vec3<f32>, 9>, sharpness: f32) -> vec3<f32> {
    let smn = cas_soft_min(n);
    let smx = cas_soft_max(n);
    let amp = cas_amplitude(smn, smx);
    let w = cas_weight(amp, sharpness);
    return cas_blend(n[1], n[3], n[4], n[5], n[7], w);
}

struct CasParams {
    strength: f32,
    sharpness: f32,
}

// Apply the kernel and blend it over the centre by params.strength.
fn cas_apply(n: array<vec3<f32>, 9>, params: CasParams) -> vec3<f32> {
    let e = n[4];
    let s = clamp(params.strength, 0.0, 1.0);
    let sharp = cas_sharpen(n, params.sharpness);
    return e + (sharp - e) * s;
}

// --- Real full-screen compute entry -------------------------------------------
// One invocation per output pixel. The 3x3 gather taps are `textureLoad`ed from
// `scene_color` with clamped coordinates (edge pixels reuse the border tap,
// matching a clamp-to-edge sampler), fed through the golden `cas_apply`, and the
// sharpened `rgb` is written to `cas_out` with the centre alpha preserved.
//
// `CAS` is display-referred (it belongs after the display tone-map curve), but
// the current pipeline defers tone-mapping to a post-`MainPass` node, so this
// pass runs on the pre-`MainPass` linear `HDR` `scene_color` to stay consistent
// with the rest of the post chain; it may be repositioned once tone-map
// ownership is decided.

// Immediate (push-constant) block; mirrors `abi::GpuCasParams` byte-for-byte:
// `vec2<u32>` extent at offset 0, then the two artist scalars. 16 bytes.
struct GpuCasParams {
    screen_size: vec2<u32>,
    strength: f32,
    sharpness: f32,
}

var<immediate> cas_params: GpuCasParams;

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var cas_out: texture_storage_2d<rgba16float, write>;

// Workgroup size (per axis) of `cas_main`; mirrors `abi::CAS_WORKGROUP_SIZE`.
const CAS_WG: u32 = 8u;

// Clamp-to-edge gather tap: reads `scene_color.rgb` at `coord` clamped into the
// valid pixel range `[0, max_xy]`, so border invocations reuse the edge tap.
fn cas_load_tap(coord: vec2<i32>, max_xy: vec2<i32>) -> vec3<f32> {
    let c = clamp(coord, vec2<i32>(0, 0), max_xy);
    return textureLoad(scene_color, c, 0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn cas_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= cas_params.screen_size.x || gid.y >= cas_params.screen_size.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let max_xy = vec2<i32>(
        i32(cas_params.screen_size.x) - 1,
        i32(cas_params.screen_size.y) - 1,
    );

    // Row-major 3x3 neighbourhood (a b c / d e f / g h i); index 4 is centre e.
    var n: array<vec3<f32>, 9>;
    n[0] = cas_load_tap(coord + vec2<i32>(-1, -1), max_xy);
    n[1] = cas_load_tap(coord + vec2<i32>( 0, -1), max_xy);
    n[2] = cas_load_tap(coord + vec2<i32>( 1, -1), max_xy);
    n[3] = cas_load_tap(coord + vec2<i32>(-1,  0), max_xy);
    n[4] = cas_load_tap(coord, max_xy);
    n[5] = cas_load_tap(coord + vec2<i32>( 1,  0), max_xy);
    n[6] = cas_load_tap(coord + vec2<i32>(-1,  1), max_xy);
    n[7] = cas_load_tap(coord + vec2<i32>( 0,  1), max_xy);
    n[8] = cas_load_tap(coord + vec2<i32>( 1,  1), max_xy);

    var params: CasParams;
    params.strength = cas_params.strength;
    params.sharpness = cas_params.sharpness;
    let rgb = cas_apply(n, params);

    // Preserve the centre alpha; the sharpen operates on colour only.
    let centre = textureLoad(scene_color, coord, 0);
    _ = CAS_WG;
    textureStore(cas_out, coord, vec4<f32>(rgb, centre.a));
}
