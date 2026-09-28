// GPU twin of `prism_render_shading::hatching` (layered cross-hatching NPR: a
// four-tier luminance ramp overlays 1..3 sets of rotated periodic line strokes,
// densest in the shadows, then composites ink over paper).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// exposure.wesl / color_grade.wesl / kuwahara.wesl) so the WESL compile test
// can validate it standalone. Every helper is `hatching_`-prefixed to avoid
// symbol collisions with the other standalone kernels, and mirrors the CPU
// golden arm-for-arm: same Rec. 709 luma weights, same edge softness, same
// hand-written smoothstep (no builtin `smoothstep`/`mix`), same tier order
// (threshold0/angle0 first, threshold2/angle2 last) and the same `max` union of
// stroke sets. CPU `if` tier guards are rewritten with `select` so both sides
// stay in lockstep. WGSL `min`/`max`/`clamp`/`fract` are componentwise, matching
// the per-channel Rust.

// Rec. 709 luma weights (linear sRGB primaries) that map tone to line density.
const HATCHING_LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Half-width of the smoothstep anti-aliasing band on each stroke edge.
const HATCHING_EDGE_SOFTNESS: f32 = 0.02;

// Hermite smoothstep (t * t * (3 - 2t)) over [edge0, edge1]. Hand-written (not
// the WGSL builtin) so the GPU twin stays bit-identical with the CPU golden.
fn hatching_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Rec. 709 relative luminance of a linear RGB sample.
fn hatching_luminance(c: vec3<f32>) -> f32 {
    return dot(c, HATCHING_LUMA_WEIGHTS);
}

// Rotate a screen-space coordinate by `angle` radians into a stroke set's local
// frame. `sin`/`cos` are WGSL builtins on device (the CPU golden routes them
// through bevy_math::ops for libm determinism); rotation preserves length.
fn hatching_rotate2d(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

// Coverage of a single stroke set at a rotated coordinate: strokes lie at
// integer multiples of 1/frequency along the rotated y axis; `thickness` is the
// stroke half-width. 1 on a stroke, 0 between strokes, anti-aliased across edges.
fn hatching_line_coverage(rotated: vec2<f32>, frequency: f32, thickness: f32) -> f32 {
    let coord = rotated.y * frequency;
    let f = fract(coord);
    let dist = min(f, 1.0 - f);
    return 1.0 - hatching_smoothstep(thickness, thickness + HATCHING_EDGE_SOFTNESS, dist);
}

// Accumulated stroke coverage for a pixel of the given luminance. Four-tier
// ramp: no strokes above threshold0, then angle0 below it, angle1 below
// threshold1, angle2 below threshold2. Sets combine by `max`. The CPU `if` tier
// guards are rewritten as `select(0, coverage_i, lum < threshold_i)`.
fn hatching_hatch_coverage(
    luminance: f32,
    screen_pos: vec2<f32>,
    angle0: f32,
    angle1: f32,
    angle2: f32,
    threshold0: f32,
    threshold1: f32,
    threshold2: f32,
    frequency: f32,
    thickness: f32,
) -> f32 {
    var coverage = 0.0;

    let c0 = hatching_line_coverage(hatching_rotate2d(screen_pos, angle0), frequency, thickness);
    coverage = max(coverage, select(0.0, c0, luminance < threshold0));

    let c1 = hatching_line_coverage(hatching_rotate2d(screen_pos, angle1), frequency, thickness);
    coverage = max(coverage, select(0.0, c1, luminance < threshold1));

    let c2 = hatching_line_coverage(hatching_rotate2d(screen_pos, angle2), frequency, thickness);
    coverage = max(coverage, select(0.0, c2, luminance < threshold2));

    return coverage;
}

// Layered cross-hatching controls (mirrors the CPU `HatchingParams`; `enabled`
// stored as u32 for uniform layout). The default is disabled (identity).
struct HatchingParams {
    angle0: f32,
    angle1: f32,
    angle2: f32,
    threshold0: f32,
    threshold1: f32,
    threshold2: f32,
    frequency: f32,
    thickness: f32,
    enabled: u32,
}

// Composite the cross-hatch ink over paper for one pixel. While disabled the
// `scene` colour passes through (identity); otherwise the scene luminance drives
// the coverage and the result is the lerp `bg + (fg - bg) * coverage` (written
// out, not the builtin `mix`). The CPU `if !enabled` guard is a `select`.
fn hatching_apply(
    scene: vec3<f32>,
    screen_pos: vec2<f32>,
    enabled: bool,
    fg: vec3<f32>,
    bg: vec3<f32>,
    angle0: f32,
    angle1: f32,
    angle2: f32,
    threshold0: f32,
    threshold1: f32,
    threshold2: f32,
    frequency: f32,
    thickness: f32,
) -> vec3<f32> {
    let lum = hatching_luminance(scene);
    let coverage = hatching_hatch_coverage(
        lum, screen_pos, angle0, angle1, angle2,
        threshold0, threshold1, threshold2, frequency, thickness,
    );
    let hatched = bg + (fg - bg) * coverage;
    return select(hatched, scene, !enabled);
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real hatching pass wires the scene target and the
// params uniform later. References every helper so none drifts unused, using
// fixed literal inputs (angles PI/4, -PI/4, 0; thresholds 0.75/0.5/0.25).
@compute @workgroup_size(64, 1, 1)
fn hatching_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.01;
    let scene = vec3<f32>(0.15 + seed, 0.4, 0.85);
    let screen_pos = vec2<f32>(0.13 + seed, 0.27);

    let ss = hatching_smoothstep(0.15, 0.17, 0.16);
    let lum = hatching_luminance(scene);
    let rot = hatching_rotate2d(screen_pos, 0.7853981633974483);
    let line = hatching_line_coverage(rot, 40.0, 0.15);
    let cov = hatching_hatch_coverage(
        lum, screen_pos,
        0.7853981633974483, -0.7853981633974483, 0.0,
        0.75, 0.5, 0.25,
        40.0, 0.15,
    );

    var params: HatchingParams;
    params.angle0 = 0.7853981633974483;
    params.angle1 = -0.7853981633974483;
    params.angle2 = 0.0;
    params.threshold0 = 0.75;
    params.threshold1 = 0.5;
    params.threshold2 = 0.25;
    params.frequency = 40.0;
    params.thickness = 0.15;
    params.enabled = 1u;

    let fg = vec3<f32>(0.0, 0.0, 0.0);
    let bg = vec3<f32>(1.0, 1.0, 1.0);
    let applied = hatching_apply(
        scene, screen_pos, params.enabled != 0u, fg, bg,
        params.angle0, params.angle1, params.angle2,
        params.threshold0, params.threshold1, params.threshold2,
        params.frequency, params.thickness,
    );

    _ = ss + lum + rot.x + line + cov + applied.x;
}
