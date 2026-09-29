// GPU twin of `prism_render_shading::vignette` (AAA vignette / lens shading:
// physical cos^4 optical fall-off and an artist Unity-PPv2-style smoothstep
// vignette).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// color_grade.wesl) so the WESL compile test can validate it standalone. Every
// function mirrors the CPU golden arm-for-arm: same function split, same
// constants, same operation order. The cos^4 power is `c2 = c*c; c2*c2`,
// smoothstep is written `3t*t - 2t*t*t` and the degenerate/soft guards use
// `select(false, true, cond)` so the GPU twin stays bit-identical with the
// golden. `sqrt`, `abs`, `max`, `clamp` are componentwise / scalar as on the CPU.

// The cos^4 optical fall-off given the cosine of the field angle.
fn vignette_natural(cos_theta: f32) -> f32 {
    let c2 = cos_theta * cos_theta;
    return c2 * c2;
}

// Natural cos^4 fall-off at a normalised uv in [0, 1]; optical axis at the
// frame centre (0.5, 0.5), `focal_ratio` the normalised focal length.
fn vignette_natural_falloff(uv: vec2<f32>, focal_ratio: f32) -> f32 {
    let dx = uv.x - 0.5;
    let dy = uv.y - 0.5;
    let r2 = dx * dx + dy * dy;
    let cos_theta = focal_ratio / sqrt(focal_ratio * focal_ratio + r2);
    return vignette_natural(cos_theta);
}

// Hermite smoothstep written 3t^2 - 2t^3 (t clamped to [0, 1]); degenerate
// edge0 == edge1 guarded so the division never goes non-finite.
fn vignette_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    let d = select(1.0e-6, denom, abs(denom) > 1.0e-6);
    let t = clamp((x - edge0) / d, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Artist vignette fall-off at `uv` (Unity PPv2 idiom). Coordinates map to
// [-1, 1] about `center`; `roundness` blends a square (0) and circular (1)
// metric; `intensity = 0` returns 1 (identity).
fn vignette_artistic_falloff(
    uv: vec2<f32>,
    center: vec2<f32>,
    intensity: f32,
    smoothness: f32,
    roundness: f32,
    aspect_ratio: f32,
) -> f32 {
    let dx = (uv.x - center.x) * 2.0 * aspect_ratio;
    let dy = (uv.y - center.y) * 2.0;

    let circular = sqrt(dx * dx + dy * dy);
    let square = max(abs(dx), abs(dy));
    let dist = square + (circular - square) * roundness;

    let s = select(1.0e-4, smoothness, smoothness > 1.0e-4);
    let edge0 = 1.0 - s;
    let edge1 = 1.0 + s;
    let t = vignette_smoothstep(edge0, edge1, dist);

    return 1.0 - clamp(t * intensity, 0.0, 1.0);
}

// Multiply linear RGB radiance by a scalar vignette factor in [0, 1].
fn vignette_apply(rgb: vec3<f32>, factor: f32) -> vec3<f32> {
    return rgb * factor;
}

// Artist controls for the vignette pass (mirrors the CPU `VignetteParams`;
// `mode` 0 = natural cos^4, 1 = artistic).
struct VignetteParams {
    center: vec2<f32>,
    intensity: f32,
    smoothness: f32,
    feather: f32,
    roundness: f32,
    aspect_ratio: f32,
    focal_ratio: f32,
    mode: u32,
}

// Evaluate the vignette factor for `uv`, dispatching on the mode; the artistic
// transition width is smoothness + feather.
fn vignette_factor(uv: vec2<f32>, params: VignetteParams) -> f32 {
    if (params.mode == 0u) {
        return vignette_natural_falloff(uv, params.focal_ratio);
    }
    return vignette_artistic_falloff(
        uv,
        params.center,
        params.intensity,
        params.smoothness + params.feather,
        params.roundness,
        params.aspect_ratio,
    );
}

// Apply the full vignette to `rgb` at `uv` (rgb * vignette_factor).
fn vignette_apply_params(rgb: vec3<f32>, uv: vec2<f32>, params: VignetteParams) -> vec3<f32> {
    return vignette_apply(rgb, vignette_factor(uv, params));
}

// ---------------------------------------------------------------------------
// Device wiring: full-screen compute pass over the pre-exposed HDR scene colour.
// ---------------------------------------------------------------------------

// Immediate (push-constant) block driving the vignette pass. Mirrors the Rust
// `GpuVignetteParams` byte-for-byte: the framebuffer extent and the artist
// controls (the golden `VignetteParams` fields) plus a global `scale` in [0, 1]
// that fades the whole effect (scale = 1 reproduces the golden exactly).
struct GpuVignetteParams {
    center: vec2<f32>,
    screen_size: vec2<u32>,
    intensity: f32,
    smoothness: f32,
    feather: f32,
    roundness: f32,
    aspect_ratio: f32,
    focal_ratio: f32,
    mode: u32,
    scale: f32,
}

// The pre-exposed, resolved linear HDR scene colour this pass darkens.
@group(0) @binding(0) var vignette_scene_color: texture_2d<f32>;
// The darkened HDR output; copied back over `scene_color` after the dispatch.
@group(0) @binding(1) var vignette_out: texture_storage_2d<rgba16float, write>;

var<immediate> vignette_params: GpuVignetteParams;

// Repack the immediate block into the golden `VignetteParams` the maths twin
// consumes, so the on-device evaluation is arm-for-arm identical to the CPU
// reference (`scale` is applied on top and is the only non-golden control).
fn vignette_params_from_immediate() -> VignetteParams {
    var params: VignetteParams;
    params.center = vignette_params.center;
    params.intensity = vignette_params.intensity;
    params.smoothness = vignette_params.smoothness;
    params.feather = vignette_params.feather;
    params.roundness = vignette_params.roundness;
    params.aspect_ratio = vignette_params.aspect_ratio;
    params.focal_ratio = vignette_params.focal_ratio;
    params.mode = vignette_params.mode;
    return params;
}

// One invocation per pixel. Reconstructs the pixel-centre uv, evaluates the
// golden vignette factor, fades it toward identity by `scale`, multiplies the
// scene radiance and writes the darkened HDR texel (alpha preserved).
@compute @workgroup_size(8, 8, 1)
fn vignette_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(vignette_params.screen_size.x);
    let height = i32(vignette_params.screen_size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }

    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(vignette_params.screen_size.x),
        (f32(gid.y) + 0.5) / f32(vignette_params.screen_size.y),
    );

    let sample = textureLoad(vignette_scene_color, pc, 0);

    let params = vignette_params_from_immediate();
    let factor = vignette_factor(uv, params);
    // Fade the darkening toward identity (1.0) by the global scale; scale = 1
    // is the exact golden factor, scale = 0 disables the effect per-pixel.
    let scaled = mix(1.0, factor, clamp(vignette_params.scale, 0.0, 1.0));

    let out_rgb = vignette_apply(sample.rgb, scaled);
    textureStore(vignette_out, pc, vec4<f32>(out_rgb, sample.a));
}
