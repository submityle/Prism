// GPU twin of `prism_render_shading::lens_flare` (screen-space lens-flare /
// glare: a soft bright-tail threshold, ghost discs mirrored through the optical
// centre, a radial halo ring, optional per-channel chromatic dispersion and the
// artist composite).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// color_grade.wesl / exposure.wesl) so the WESL compile test can validate it
// standalone. Every function mirrors the CPU golden arm-for-arm: same
// primitives, same constants, `smoothstep` / `mix` written out (`3t^2 - 2t^3`
// and `a + (b - a) * t`) rather than the WGSL builtins so the GPU twin stays
// bit-identical with the golden, and WGSL componentwise vector ops match the
// per-channel Rust.

// Rec. 709 luminance weights (linear sRGB primaries), used to isolate the
// bright tail.
const LENS_FLARE_LUMINANCE_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Optical centre of the frame in normalised uv space; ghosts and halo are
// mirrored/aligned through this point.
const LENS_FLARE_CENTER: vec2<f32> = vec2<f32>(0.5, 0.5);

// Rec. 709 relative luminance of a linear RGB radiance sample.
fn lens_flare_luminance(rgb: vec3<f32>) -> f32 {
    return dot(rgb, LENS_FLARE_LUMINANCE_WEIGHTS);
}

// Own smoothstep `t^2 * (3 - 2t)` on the normalised, clamped
// `t = (x - edge0) / (edge1 - edge0)`. Written out (not the WGSL builtin) so it
// stays bit-identical with the CPU golden; a degenerate `edge0 == edge1` is a
// hard step at the edge.
fn lens_flare_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (span == 0.0) {
        return select(0.0, 1.0, x >= edge0);
    }
    let t = clamp((x - edge0) / span, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Per-channel lerp `a + (b - a) * t`, written out (not the WGSL builtin `mix`)
// so it matches the golden exactly. `t` is not clamped.
fn lens_flare_mix(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// Soft, hue-preserving isolation of the bright tail:
// contribution = max(0, luminance - threshold) / max(luminance, 1e-6);
// out = rgb * contribution. Zero at or below `threshold`; keeps channel ratios.
fn lens_flare_threshold_prefilter(rgb: vec3<f32>, threshold: f32) -> vec3<f32> {
    let luma = lens_flare_luminance(rgb);
    let contribution = max(luma - threshold, 0.0) / max(luma, 1.0e-6);
    return rgb * contribution;
}

// Sampling uv for the `ghost_index`-th ghost: step from `uv` toward the optical
// centre by `dispersal * ghost_index` of the centre-facing vector. Index 0
// returns `uv`; the centre itself is stable (no direction).
fn lens_flare_ghost_uv(uv: vec2<f32>, ghost_index: f32, dispersal: f32) -> vec2<f32> {
    let dir = LENS_FLARE_CENTER - uv;
    let step = dispersal * ghost_index;
    return uv + dir * step;
}

// Sampling uv for the halo ring: step a fixed `halo_width` along the unit vector
// pointing from `uv` to the optical centre. A `uv` at the centre has no
// direction and is returned unchanged.
fn lens_flare_halo_uv(uv: vec2<f32>, halo_width: f32) -> vec2<f32> {
    let dir = LENS_FLARE_CENTER - uv;
    let len = sqrt(dir.x * dir.x + dir.y * dir.y);
    if (len < 1.0e-6) {
        return uv;
    }
    let n = dir / len;
    return uv + n * halo_width;
}

// Radial weight that peaks at the optical centre and smoothly falls to 0 by
// half a frame out: 1 - smoothstep(0, 0.5, distance(uv, centre)).
fn lens_flare_radial_weight(uv: vec2<f32>) -> f32 {
    let d = uv - LENS_FLARE_CENTER;
    let dist = sqrt(d.x * d.x + d.y * d.y);
    return 1.0 - lens_flare_smoothstep(0.0, 0.5, dist);
}

// Vignette multiplier in [1 - strength, 1]: full weight at the centre,
// darkening toward the frame edge by strength * smoothstep(0, 1, 2 * dist).
fn lens_flare_vignette_weight(uv: vec2<f32>, strength: f32) -> f32 {
    let d = uv - LENS_FLARE_CENTER;
    let dist = sqrt(d.x * d.x + d.y * d.y);
    let edge = lens_flare_smoothstep(0.0, 1.0, 2.0 * dist);
    return 1.0 - strength * edge;
}

// Per-channel radial micro-offset for a chromatic (dispersed) ghost. `channel`
// selects a signed lobe: red (0) shifts one way, green (1) is unshifted, blue
// (2) the other, by distortion * (channel - 1) along the centre-facing vector.
fn lens_flare_chromatic_ghost_offset(uv: vec2<f32>, distortion: f32, channel: u32) -> vec2<f32> {
    let dir = LENS_FLARE_CENTER - uv;
    let lobe = f32(channel) - 1.0;
    let step = distortion * lobe;
    return uv + dir * step;
}

// Accumulate four weighted ghost samples: sum(color_i * weight_i). The CPU
// golden takes an arbitrary-length slice; the GPU twin fixes the fan-out at the
// four-tap chain a compute pass repeats. Same weighted-sum arm.
fn lens_flare_accumulate_ghosts(colors: array<vec3<f32>, 4>, weights: array<f32, 4>) -> vec3<f32> {
    var out = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        out = out + colors[i] * weights[i];
    }
    return out;
}

// Composite the accumulated flare over the scene additively:
// scene + flare * intensity. intensity = 0 returns the untouched scene.
fn lens_flare_apply(scene_rgb: vec3<f32>, flare_rgb: vec3<f32>, intensity: f32) -> vec3<f32> {
    return scene_rgb + flare_rgb * intensity;
}

// Artist controls for the full pass (mirrors the CPU `LensFlareParams`; the
// neutral field values give a disabled flare: intensity 0 is the identity).
struct LensFlareParams {
    intensity: f32,
    threshold: f32,
    ghost_count: u32,
    dispersal: f32,
    halo_width: f32,
    distortion: f32,
}

// ---------------------------------------------------------------------------
// Device wiring: full-screen compute pass over the pre-exposed HDR scene colour.
// ---------------------------------------------------------------------------

// Immediate (push-constant) block driving the flare pass. Mirrors the Rust
// `GpuLensFlareParams` byte-for-byte: the framebuffer extent leads (8-byte
// aligned vec2<u32>), then the six golden `LensFlareParams` artist scalars.
struct GpuLensFlareParams {
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<u32>,
    // Number of ghost discs (golden `ghost_count`); the GPU twin caps the
    // fan-out at four taps and masks beyond this count.
    ghost_count: u32,
    // Composite blend weight over the scene (golden `intensity`).
    intensity: f32,
    // Bright-tail luminance threshold (golden `threshold`).
    threshold: f32,
    // Ghost chain spacing toward the centre (golden `dispersal`).
    dispersal: f32,
    // Halo ring radial offset (golden `halo_width`).
    halo_width: f32,
    // Per-channel chromatic dispersion (golden `distortion`).
    distortion: f32,
}

// The pre-exposed, resolved linear HDR scene colour this pass reads.
@group(0) @binding(0) var lens_flare_scene_color: texture_2d<f32>;
// The flared HDR output; copied back over `scene_color` after the dispatch.
@group(0) @binding(1) var lens_flare_out: texture_storage_2d<rgba16float, write>;

var<immediate> lens_flare_params: GpuLensFlareParams;

// Repack the immediate block into the golden `LensFlareParams` the maths twin
// consumes, so the on-device evaluation is arm-for-arm identical to the CPU
// reference.
fn lens_flare_params_from_immediate() -> LensFlareParams {
    var params: LensFlareParams;
    params.intensity = lens_flare_params.intensity;
    params.threshold = lens_flare_params.threshold;
    params.ghost_count = lens_flare_params.ghost_count;
    params.dispersal = lens_flare_params.dispersal;
    params.halo_width = lens_flare_params.halo_width;
    params.distortion = lens_flare_params.distortion;
    return params;
}

// Manual bilinear read of the scene colour at a normalised uv. The scene colour
// is bound non-filterable (the ghost/halo taps land at fractional uv, so the
// four-texel blend is done by hand with the golden `lens_flare_mix` lerp rather
// than a hardware sampler). Out-of-frame taps clamp to the edge texel.
fn lens_flare_sample(uv: vec2<f32>, size: vec2<u32>) -> vec3<f32> {
    let dims = vec2<f32>(f32(size.x), f32(size.y));
    let coord = uv * dims - vec2<f32>(0.5, 0.5);
    let base = floor(coord);
    let frac = coord - base;
    let max_x = i32(size.x) - 1;
    let max_y = i32(size.y) - 1;
    let x0 = clamp(i32(base.x), 0, max_x);
    let y0 = clamp(i32(base.y), 0, max_y);
    let x1 = clamp(i32(base.x) + 1, 0, max_x);
    let y1 = clamp(i32(base.y) + 1, 0, max_y);
    let c00 = textureLoad(lens_flare_scene_color, vec2<i32>(x0, y0), 0).rgb;
    let c10 = textureLoad(lens_flare_scene_color, vec2<i32>(x1, y0), 0).rgb;
    let c01 = textureLoad(lens_flare_scene_color, vec2<i32>(x0, y1), 0).rgb;
    let c11 = textureLoad(lens_flare_scene_color, vec2<i32>(x1, y1), 0).rgb;
    let top = lens_flare_mix(c00, c10, frac.x);
    let bottom = lens_flare_mix(c01, c11, frac.x);
    return lens_flare_mix(top, bottom, frac.y);
}

// Sample one ghost tap with per-channel chromatic dispersion: red/green/blue are
// each read at their own radial micro-offset (golden `chromatic_ghost_offset`),
// so the disc fringes red/blue like real lens dispersion. distortion == 0
// collapses all three offsets to `guv`.
fn lens_flare_sample_dispersed(guv: vec2<f32>, distortion: f32, size: vec2<u32>) -> vec3<f32> {
    let r_uv = lens_flare_chromatic_ghost_offset(guv, distortion, 0u);
    let g_uv = lens_flare_chromatic_ghost_offset(guv, distortion, 1u);
    let b_uv = lens_flare_chromatic_ghost_offset(guv, distortion, 2u);
    let r = lens_flare_sample(r_uv, size).r;
    let g = lens_flare_sample(g_uv, size).g;
    let b = lens_flare_sample(b_uv, size).b;
    return vec3<f32>(r, g, b);
}

// One invocation per pixel. Isolates the bright tail of the scene colour along
// the four-tap ghost chain and the halo ring (each threshold-prefiltered and
// radially weighted exactly like the golden), accumulates them with the golden
// weighted sum and composites the flare additively over the scene by the artist
// intensity. Alpha is preserved. Runs in the scene-referred linear HDR domain,
// before the display tone-map curve, exactly like the golden.
@compute @workgroup_size(8, 8, 1)
fn lens_flare_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = lens_flare_params.screen_size;
    let width = i32(size.x);
    let height = i32(size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }

    let scene = textureLoad(lens_flare_scene_color, pc, 0);
    let params = lens_flare_params_from_immediate();

    let dims = vec2<f32>(f32(size.x), f32(size.y));
    let uv = (vec2<f32>(f32(pc.x), f32(pc.y)) + vec2<f32>(0.5, 0.5)) / dims;

    // Four-tap ghost chain: each disc is the bright tail sampled a `dispersal`
    // step deeper toward the optical centre, weighted by the radial falloff and
    // masked off past `ghost_count`. Fixed fan-out matching the GPU twin form of
    // `lens_flare_accumulate_ghosts`.
    var colors = array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    var weights = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let guv = lens_flare_ghost_uv(uv, f32(i), params.dispersal);
        let sampled = lens_flare_sample_dispersed(guv, params.distortion, size);
        colors[i] = lens_flare_threshold_prefilter(sampled, params.threshold);
        let ghost_mask = select(0.0, 1.0, i < params.ghost_count);
        weights[i] = lens_flare_radial_weight(guv) * ghost_mask;
    }
    var flare = lens_flare_accumulate_ghosts(colors, weights);

    // Halo ring: one bright-tail tap a fixed `halo_width` along the centre-facing
    // unit vector, radially weighted like the ghosts.
    let huv = lens_flare_halo_uv(uv, params.halo_width);
    let halo_sample = lens_flare_sample(huv, size);
    let halo_bright = lens_flare_threshold_prefilter(halo_sample, params.threshold);
    flare = flare + halo_bright * lens_flare_radial_weight(huv);

    // Vignette-fade the accumulated flare toward the frame edge before the
    // additive composite so the artefacts never clip hard at the border (golden
    // `vignette_weight`, full-strength radial edge darkening).
    flare = flare * lens_flare_vignette_weight(uv, 1.0);

    let out_rgb = lens_flare_apply(scene.rgb, flare, params.intensity);
    textureStore(lens_flare_out, pc, vec4<f32>(out_rgb, scene.a));
}
