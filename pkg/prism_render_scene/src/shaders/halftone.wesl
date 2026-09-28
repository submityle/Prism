// GPU twin of `prism_render_shading::halftone` (NPR manga / screentone
// halftone: a rotated screen of tone-driven dots over the frame).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// color_grade.wesl / vignette.wesl) so the WESL compile test can validate it
// standalone. Every helper is `halftone_`-prefixed and mirrors the CPU golden
// arm-for-arm: same function split, same constants, same operation order.
// smoothstep is written `t*t*(3 - 2t)`, squares as `x*x`, and the disabled
// branch returns the scene colour, so the twin stays lock-step with the golden.
// `sin`/`cos`/`sqrt`/`floor`/`clamp`/`length`/`mix`/`select` are WGSL builtins.

// Artist controls for the halftone screen. `enabled == 0` is the identity pass.
struct HalftoneParams {
    cell_size: f32,
    angle: f32,
    enabled: u32,
}

const HALFTONE_LUMA_R: f32 = 0.2126;
const HALFTONE_LUMA_G: f32 = 0.7152;
const HALFTONE_LUMA_B: f32 = 0.0722;
// Darkest-tone dot radius as a fraction of cell_size (matches the CPU golden).
const HALFTONE_MAX_DOT_RADIUS: f32 = 0.5;

// Rec. 709 relative luminance of a linear colour.
fn halftone_luminance(c: vec3<f32>) -> f32 {
    return c.x * HALFTONE_LUMA_R + c.y * HALFTONE_LUMA_G + c.z * HALFTONE_LUMA_B;
}

// Rotate `p` by `angle` radians (counter-clockwise) into the dot-screen frame.
fn halftone_rotate2d(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let cs = cos(angle);
    return vec2<f32>(p.x * cs - p.y * s, p.x * s + p.y * cs);
}

// Position within a cell, relative to its centre; periodic in cell_size.
fn halftone_cell_coord(screen_pos: vec2<f32>, cell_size: f32) -> vec2<f32> {
    let cs = max(cell_size, 1.0e-6);
    let normalized = screen_pos / cs;
    let frac = normalized - floor(normalized);
    return (frac - vec2<f32>(0.5, 0.5)) * cs;
}

// Dot radius (fraction of cell_size) for a local tone: dark -> big, bright -> 0.
fn halftone_dot_radius_from_tone(luma: f32) -> f32 {
    let t = clamp(luma, 0.0, 1.0);
    return sqrt(1.0 - t) * HALFTONE_MAX_DOT_RADIUS;
}

// Hermite smoothstep 3t^2 - 2t^3 (t clamped to [0, 1]); degenerate guard so the
// division stays finite. Matches the CPU golden exactly.
fn halftone_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    let d = select(1.0e-6, denom, abs(denom) > 1.0e-6);
    let t = clamp((x - edge0) / d, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Anti-aliased ink coverage of a dot: 1 inside, 0 outside, 0.5 at the edge.
fn halftone_dot_coverage(dist: f32, radius: f32, aa: f32) -> f32 {
    let a = max(aa, 1.0e-6);
    return 1.0 - halftone_smoothstep(radius - a, radius + a, dist);
}

// Full pass. `enabled == 0` returns the scene colour; otherwise the tone grows
// a dot and the result is the foreground ink over the paper background.
fn halftone_apply(
    scene: vec3<f32>,
    screen_pos: vec2<f32>,
    params: HalftoneParams,
    fg: vec3<f32>,
    bg: vec3<f32>,
) -> vec3<f32> {
    if (params.enabled == 0u) {
        return scene;
    }
    let luma = halftone_luminance(scene);
    let rotated = halftone_rotate2d(screen_pos, params.angle);
    let local = halftone_cell_coord(rotated, params.cell_size);
    let dist = length(local);
    let radius = halftone_dot_radius_from_tone(luma) * params.cell_size;
    let coverage = halftone_dot_coverage(dist, radius, 1.0);
    return mix(bg, fg, coverage);
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real halftone pass wires the HDR target and the
// params uniform later. References every helper with fixed literal inputs.
@compute @workgroup_size(64, 1, 1)
fn halftone_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.001;
    let scene = vec3<f32>(0.2 + seed, 0.5, 0.9);
    let pos = vec2<f32>(8.0, 5.0);
    let fg = vec3<f32>(0.0, 0.0, 0.0);
    let bg = vec3<f32>(1.0, 1.0, 1.0);

    let luma = halftone_luminance(scene);
    let rot = halftone_rotate2d(pos, 0.7853981633974483);
    let local = halftone_cell_coord(rot, 10.0);
    let radius = halftone_dot_radius_from_tone(luma);
    let ss = halftone_smoothstep(0.0, 1.0, 0.5);
    let cov = halftone_dot_coverage(length(local), radius * 10.0, 1.0);

    var params: HalftoneParams;
    params.cell_size = 10.0;
    params.angle = 0.7853981633974483;
    params.enabled = 1u;
    let out = halftone_apply(scene, pos, params, fg, bg);

    _ = luma + rot.x + local.x + radius + ss + cov + out.x;
}
