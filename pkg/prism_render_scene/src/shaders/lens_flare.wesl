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

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real flare pass wires the HDR target and the
// params uniform later. References every helper once so none drifts unused.
@compute @workgroup_size(64, 1, 1)
fn lens_flare_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.01;
    let rgb = vec3<f32>(0.15 + seed, 0.4, 0.85);
    let uv = vec2<f32>(0.2 + seed, 0.7);

    let lum = lens_flare_luminance(rgb);
    let ss = lens_flare_smoothstep(0.0, 1.0, 0.5 + seed);
    let mixed = lens_flare_mix(rgb, vec3<f32>(1.0, 0.5, 0.25), 0.5);
    let pf = lens_flare_threshold_prefilter(rgb, 0.2);
    let guv = lens_flare_ghost_uv(uv, 1.0, 0.6);
    let huv = lens_flare_halo_uv(uv, 0.2);
    let rw = lens_flare_radial_weight(uv);
    let vw = lens_flare_vignette_weight(uv, 0.5);
    let cro = lens_flare_chromatic_ghost_offset(uv, 0.3, 0u);

    let colors = array<vec3<f32>, 4>(pf, mixed, rgb, vec3<f32>(0.1, 0.2, 0.3));
    let weights = array<f32, 4>(rw, vw, 0.25, 0.125);
    let acc = lens_flare_accumulate_ghosts(colors, weights);
    let composited = lens_flare_apply(rgb, acc, 0.0);

    var params: LensFlareParams;
    params.intensity = 0.0;
    params.threshold = 0.0;
    params.ghost_count = 0u;
    params.dispersal = 0.0;
    params.halo_width = 0.0;
    params.distortion = 0.0;

    _ = lum + ss + mixed.x + pf.x + guv.x + huv.x + rw + vw + cro.x
        + acc.x + composited.x + f32(params.ghost_count) + params.intensity;
}
