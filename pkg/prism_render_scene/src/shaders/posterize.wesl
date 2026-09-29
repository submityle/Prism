// GPU twin of `prism_render_shading::posterize` (NPR posterization / tone
// separation in the scene-referred linear domain, applied before the display
// tone-map curve: snap tones to a small set of flat bands, either per-channel
// RGB or luma-preserving, then blend the quantized result over the input).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// exposure.wesl / color_grade.wesl / gamut_map.wesl) so the WESL compile test
// can validate it standalone. Every function mirrors the CPU golden arm-for-arm:
// same operation order, same constants, `if` guards written with `select` (the
// color_grade.wesl precedent) and blends written `a + (b - a) * t` rather than
// `mix` so the GPU twin stays bit-identical with the golden. `floor` on a vec3
// is componentwise in WGSL, matching the per-channel Rust.

// Rec. 709 luma weights (linear sRGB primaries), used by posterize_luminance.
const POSTERIZE_LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Luminance floor below which the luma-preserving rescale is skipped.
const POSTERIZE_LUMA_EPS: f32 = 1.0e-5;

// Rec. 709 relative luminance of a linear RGB sample.
fn posterize_luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, POSTERIZE_LUMA_WEIGHTS);
}

// Snap a scalar to `levels` evenly spaced bands via the round-to-nearest grid
// `floor(x * (levels - 1) + 0.5) / (levels - 1)`. `levels <= 1` returns the
// input unchanged (no grid).
fn posterize_quantize(x: f32, levels: f32) -> f32 {
    let n = levels - 1.0;
    let q = floor(x * n + 0.5) / n;
    return select(q, x, levels <= 1.0);
}

// Per-channel RGB posterize: `posterize_quantize` applied independently per
// channel (three scalar calls, matching the per-channel Rust exactly).
fn posterize_quantize_rgb(rgb: vec3<f32>, levels: f32) -> vec3<f32> {
    return vec3<f32>(
        posterize_quantize(rgb.x, levels),
        posterize_quantize(rgb.y, levels),
        posterize_quantize(rgb.z, levels),
    );
}

// Luma-preserving posterize: quantize the Rec. 709 luminance and rescale the
// original colour by `quant_luma / luma`, preserving the chroma ratios. A
// luminance at or below POSTERIZE_LUMA_EPS is returned unchanged.
fn posterize_quantize_luma_preserving(rgb: vec3<f32>, luma_levels: f32) -> vec3<f32> {
    let l = posterize_luminance(rgb);
    let ql = posterize_quantize(l, luma_levels);
    let scaled = rgb * (ql / l);
    return select(scaled, rgb, l <= POSTERIZE_LUMA_EPS);
}

// Artist controls for the posterize (mirrors the CPU `PosterizeParams`;
// `use_luma` and `enabled` are u32 flags for the GPU uniform, 0 = off).
struct PosterizeParams {
    levels: f32,
    luma_levels: f32,
    strength: f32,
    use_luma: u32,
    enabled: u32,
}

// Apply the posterize in the fixed order
// quantize (per-channel RGB or luma-preserving) -> blend by `strength`.
// enabled = 0 forces the blend weight to 0 (a hard identity).
fn posterize_apply(rgb: vec3<f32>, params: PosterizeParams) -> vec3<f32> {
    let rgb_q = posterize_quantize_rgb(rgb, params.levels);
    let luma_q = posterize_quantize_luma_preserving(rgb, params.luma_levels);
    let base = select(rgb_q, luma_q, params.use_luma != 0u);
    let t = select(0.0, params.strength, params.enabled != 0u);
    return rgb + (base - rgb) * t;
}

// --- Real full-screen compute entry -------------------------------------------
// One invocation per output pixel: `textureLoad` the pre-exposed linear `HDR`
// `scene_color` texel, run it through the golden `posterize_apply`, and write
// the banded `rgb` to `posterize_out` with the source alpha preserved.
//
// Posterize is display-referred in spirit but is applied here in the
// scene-referred linear domain before the display tone-map curve, consistent
// with the current pipeline (tone-mapping is deferred to a post-`MainPass`
// node); it may be repositioned once tone-map ownership is decided.

// Immediate (push-constant) block; mirrors `abi::GpuPosterizeParams`
// byte-for-byte: `vec2<u32>` extent at offset 0, the three `f32` controls, the
// two `u32` flags and a `u32` tail pad. 32 bytes.
struct GpuPosterizeParams {
    screen_size: vec2<u32>,
    levels: f32,
    luma_levels: f32,
    strength: f32,
    use_luma: u32,
    enabled: u32,
    _pad: u32,
}

var<immediate> posterize_params: GpuPosterizeParams;

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var posterize_out: texture_storage_2d<rgba16float, write>;

// Workgroup size (per axis) of `posterize_main`; mirrors
// `abi::POSTERIZE_WORKGROUP_SIZE`.
const POSTERIZE_WG: u32 = 8u;

@compute @workgroup_size(8, 8, 1)
fn posterize_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= posterize_params.screen_size.x || gid.y >= posterize_params.screen_size.y) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let texel = textureLoad(scene_color, coord, 0);

    var params: PosterizeParams;
    params.levels = posterize_params.levels;
    params.luma_levels = posterize_params.luma_levels;
    params.strength = posterize_params.strength;
    params.use_luma = posterize_params.use_luma;
    params.enabled = posterize_params.enabled;
    let rgb = posterize_apply(texel.rgb, params);

    _ = POSTERIZE_WG;
    textureStore(posterize_out, coord, vec4<f32>(rgb, texel.a));
}
