// GPU twin of `prism_render_shading::film_grain` (industry-standard film-grain
// post-processing: a deterministic hash-noise field, a luminance response that
// biases the grain toward the shadows, and an additive intensity mix over the
// resolved, pre-exposed linear HDR radiance).
// Self-contained (no intra-crate imports, matching color_grade.wesl /
// bloom.wesl / exposure.wesl) so the WESL compile test can validate it
// standalone. Every function mirrors the CPU golden arm-for-arm: same operation
// order, same constants, `fract` written `x - floor(x)` (WGSL `fract` floors,
// matching the CPU helper for negatives too), `smoothstep` written out as the
// Hermite `3t^2 - 2t^3` (not the intrinsic) and lerps written `a + (b - a) * t`
// rather than `mix`, so the GPU twin stays bit-identical with the golden. The
// hash is Dave Hoskins' *Hash without Sine* (constants `0.1031` / `33.33`), so
// it uses only multiply/add/`fract` with no transcendental `sin`.

// Rec. 709 luma weights (linear sRGB primaries).
const FILM_GRAIN_LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Fractional part `x - floor(x)`, matching the CPU golden's `grain_fract`
// (which floors, so `fract(-0.25) == 0.75`) rather than a truncating fract.
// Written out as a shared helper so the CPU and GPU twins agree for negatives.
fn film_grain_fract(x: f32) -> f32 {
    return x - floor(x);
}

// Rec. 709 relative luma of a linear RGB radiance sample.
fn film_grain_luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, FILM_GRAIN_LUMA_WEIGHTS);
}

// Dave Hoskins' *Hash without Sine* `hash12`: hashes a 2D coordinate to a
// uniform pseudo-random value in [0, 1). Pure multiply/add/`fract` (no `sin`)
// with the canonical constants 0.1031 and 33.33, mirroring the CPU golden
// scalar-for-scalar so it is deterministic and identical on device.
fn film_grain_hash12(p: vec2<f32>) -> f32 {
    // p3 = fract(vec3(p.x, p.y, p.x) * 0.1031)
    var px = film_grain_fract(p.x * 0.1031);
    var py = film_grain_fract(p.y * 0.1031);
    var pz = film_grain_fract(p.x * 0.1031);
    // p3 += dot(p3, p3.yzx + 33.33)
    let d = px * (py + 33.33) + py * (pz + 33.33) + pz * (px + 33.33);
    px = px + d;
    py = py + d;
    pz = pz + d;
    // return fract((p3.x + p3.y) * p3.z)
    return film_grain_fract((px + py) * pz);
}

// Signed grain value in [-1, 1) for a screen coordinate. The UV is scaled by
// `resolution` (so grain is per-pixel, not stretched with the image) and offset
// by `time_seed` (so it animates per frame), hashed, then remapped from [0, 1)
// to [-1, 1).
fn film_grain_grain(uv: vec2<f32>, resolution: vec2<f32>, time_seed: f32) -> f32 {
    let p = vec2<f32>(
        uv.x * resolution.x + time_seed,
        uv.y * resolution.y + time_seed,
    );
    return film_grain_hash12(p) * 2.0 - 1.0;
}

// Self-written `smoothstep`: 0 at/below `edge0`, 1 at/above `edge1`, and the
// Hermite `3t^2 - 2t^3` in between. Written out (not the intrinsic) so the CPU
// golden and GPU twin round identically. Degenerate `edge0 == edge1` returns a
// hard step at the edge rather than dividing by zero.
fn film_grain_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (span == 0.0) {
        return select(1.0, 0.0, x < edge0);
    }
    let t = clamp((x - edge0) / span, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Linear interpolation `a + (b - a) * t`, written out to match the CPU golden.
fn film_grain_lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Luminance-dependent grain weight. Real film shows grain most in the shadows,
// so the shadow-biased weight is `1 - smoothstep(0, 1, luma)` (weight 1 at
// black, 0 at white). `response` in [0, 1] blends from a flat weight of 1
// (response = 0) to the full shadow-biased falloff (response = 1).
fn film_grain_luminance_weight(luma: f32, response: f32) -> f32 {
    let shadow = 1.0 - film_grain_smoothstep(0.0, 1.0, luma);
    return film_grain_lerp(1.0, shadow, clamp(response, 0.0, 1.0));
}

// Adds a scalar grain value to a radiance sample: `rgb + grain * intensity *
// luma_weight` per channel (additive, so an `intensity` of 0 is the identity).
// The same value is added to every channel for monochrome grain.
fn film_grain_apply_grain(rgb: vec3<f32>, grain_value: f32, intensity: f32, luma_weight: f32) -> vec3<f32> {
    let g = grain_value * intensity * luma_weight;
    return vec3<f32>(rgb.x + g, rgb.y + g, rgb.z + g);
}

// Artist controls for the film-grain pass (mirrors the CPU `FilmGrainParams`;
// the neutral `intensity == 0` gives an identity pass). `colored` is a `u32`
// flag here (WGSL host-shareable structs cannot hold `bool`): non-zero hashes
// each channel with an offset seed, zero shares one value across channels.
struct FilmGrainParams {
    intensity: f32,
    response: f32,
    size: f32,
    colored: u32,
}

// Applies film grain to a radiance sample at screen coordinate `uv`, animated
// by `time_seed`. Uses the Rec. 709 luma for the response weight; a coloured
// grain hashes each channel with an offset seed. With the default
// (`intensity == 0`) this returns `rgb` unchanged. Mirrors the CPU
// `FilmGrainParams::apply` arm-for-arm.
fn film_grain_apply(rgb: vec3<f32>, uv: vec2<f32>, resolution: vec2<f32>, time_seed: f32, params: FilmGrainParams) -> vec3<f32> {
    let luma = film_grain_luminance(rgb);
    let weight = film_grain_luminance_weight(luma, params.response);
    // Effective sampling resolution after the grain `size` scale
    // (`resolution / size`, guarding a non-positive size).
    let inv_size = select(1.0, 1.0 / params.size, params.size > 0.0);
    let res = vec2<f32>(resolution.x * inv_size, resolution.y * inv_size);
    if (params.colored != 0u) {
        let gr = film_grain_grain(uv, res, time_seed);
        let gg = film_grain_grain(uv, res, time_seed + 1.0);
        let gb = film_grain_grain(uv, res, time_seed + 2.0);
        return vec3<f32>(
            rgb.x + gr * params.intensity * weight,
            rgb.y + gg * params.intensity * weight,
            rgb.z + gb * params.intensity * weight,
        );
    }
    let g = film_grain_grain(uv, res, time_seed);
    return film_grain_apply_grain(rgb, g, params.intensity, weight);
}

// ---------------------------------------------------------------------------
// Device entry point.
// ---------------------------------------------------------------------------

// Workgroup size (per axis) of the `film_grain_main` entry point; mirrored by
// `FILM_GRAIN_WORKGROUP_SIZE` on the Rust side. The dispatch rounds the target
// extent up to a multiple of this on both axes and the shader bounds-checks
// every invocation against `params.screen_size`.
const FILM_GRAIN_WG: u32 = 8u;

// Immediate (push-constant) block consumed by `film_grain_main`, mirroring the
// Rust `GpuFilmGrainParams`. `screen_size` (a `vec2<u32>`) leads for its 8-byte
// alignment; the trailing scalars (`intensity`, `response`, `size`,
// `time_seed`, `colored`) plus one `_padding` word round the block to 32 bytes,
// a multiple of the 8-byte alignment the `vec2` forces, with no implicit gap.
struct GpuFilmGrainParams {
    // Full-resolution framebuffer extent in texels (`vec2<u32>`, 8-byte aligned).
    screen_size: vec2<u32>,
    // Artist grain strength; `0` is the golden identity.
    intensity: f32,
    // Luminance response in [0, 1]: `0` grains uniformly, `1` biases the grain
    // fully toward the shadows.
    response: f32,
    // Grain cell size; larger values sample a lower frequency (coarser grain).
    size: f32,
    // Per-frame animation seed (a wrapped frame index cast to `f32`); offsets
    // the hash so the grain field animates frame to frame.
    time_seed: f32,
    // Coloured-grain flag: non-zero hashes each channel with an offset seed,
    // zero shares one value across channels (monochrome). WGSL host-shareable
    // structs cannot hold `bool`.
    colored: u32,
    // Padding to a multiple of the `vec2<u32>` 8-byte alignment.
    _padding: u32,
}

var<immediate> film_grain_params: GpuFilmGrainParams;

// group 0: the pre-exposed scene colour (non-filterable float, `textureLoad`ed
// per pixel) at binding 0, then the write-only `rgba16float` grained output at
// binding 1. Matches the Rust `sequential` {0, 1} layout.
@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var film_grain_out: texture_storage_2d<rgba16float, write>;

// `film_grain_main`: one invocation per output pixel. Bounds-checks against the
// framebuffer extent, loads the pre-exposed HDR radiance, applies the golden
// film-grain model (`film_grain_apply`) at this pixel's screen `uv`, and stores
// the grained radiance (alpha preserved) to the dedicated output the dispatch
// then copies back over `scene_color`.
@compute @workgroup_size(8, 8, 1)
fn film_grain_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let extent = film_grain_params.screen_size;
    if (gid.x >= extent.x || gid.y >= extent.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel = textureLoad(scene_color, coord, 0);
    let rgb = texel.rgb;

    let resolution = vec2<f32>(f32(extent.x), f32(extent.y));
    // Pixel-centre screen uv in [0, 1); `uv * resolution` reproduces the golden
    // `p = uv * resolution` sampling coordinate (i.e. pixel-space grain).
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / resolution;

    var params: FilmGrainParams;
    params.intensity = film_grain_params.intensity;
    params.response = film_grain_params.response;
    params.size = film_grain_params.size;
    params.colored = film_grain_params.colored;

    let grained = film_grain_apply(rgb, uv, resolution, film_grain_params.time_seed, params);

    // Reference the workgroup constant so it cannot drift unused; the compare is
    // always true (dispatch uses @workgroup_size(8, 8, 1)).
    let write_rgb = select(rgb, grained, FILM_GRAIN_WG == 8u);
    textureStore(film_grain_out, coord, vec4<f32>(write_rgb, texel.a));
}
