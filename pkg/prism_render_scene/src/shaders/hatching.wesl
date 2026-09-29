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

// ---------------------------------------------------------------------------
// Device wiring: full-screen compute pass over the resolved HDR scene colour.
// ---------------------------------------------------------------------------
//
// display-referred, stylized NPR filter (ink cross-hatching). It conceptually
// belongs after the display tone-map curve, but the current Prism chain defers
// tone-mapping to a Bevy node after the MainPass, so every pre-MainPass consumer
// (this one included) runs on linear HDR radiance. Placed as a pre-MainPass
// linear consumer for chain consistency; may be repositioned once tone-map
// ownership is decided.

// Immediate (push-constant) block driving the hatching pass. Mirrors the Rust
// GpuHatchingParams byte-for-byte: the framebuffer extent leads (vec2<u32>,
// 8-byte aligned), then the eight golden scalars, the enable flag and an
// explicit tail pad so the block is a multiple of 8 bytes (48), matching the
// WGSL struct alignment the vec2<u32> forces.
struct GpuHatchingParams {
    // Full-resolution framebuffer extent in texels.
    screen_size: vec2<u32>,
    // Stripe frequency (periods across the normalized screen).
    frequency: f32,
    // Stroke half-width in fractional-period units.
    thickness: f32,
    // The three stroke directions (radians), added darkest-last.
    angle0: f32,
    angle1: f32,
    angle2: f32,
    // Descending luminance tiers at which each stroke set switches on.
    threshold0: f32,
    threshold1: f32,
    threshold2: f32,
    // Master enable; 0 passes the scene through (identity).
    enabled: u32,
    // Explicit tail pad so the block is a multiple of 8 bytes.
    _pad: u32,
}

// The resolved linear HDR scene colour this pass reads (textureLoad-ed per
// pixel).
@group(0) @binding(0) var hatching_scene_color: texture_2d<f32>;
// The stylized HDR output; copied back over scene_color after the dispatch.
@group(0) @binding(1) var hatching_out: texture_storage_2d<rgba16float, write>;

var<immediate> hatching_params: GpuHatchingParams;

// One invocation per pixel. Maps the pixel to a normalized [0, 1] screen
// coordinate (so the golden stripe frequency spans the frame), composites the
// ink (black) over paper (white) by the golden coverage ramp and writes the
// result. Disabled pixels pass the scene through. Alpha is preserved. Runs on
// the resolved HDR radiance, exactly like the golden.
@compute @workgroup_size(8, 8, 1)
fn hatching_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = i32(hatching_params.screen_size.x);
    let height = i32(hatching_params.screen_size.y);
    let pc = vec2<i32>(i32(gid.x), i32(gid.y));
    if (pc.x >= width || pc.y >= height) {
        return;
    }

    let scene = textureLoad(hatching_scene_color, pc, 0);

    // Normalized [0, 1] screen coordinate so the golden frequency spans the
    // frame (the golden screen_pos is a normalized UV, not a pixel coord).
    let screen_pos = vec2<f32>(
        f32(pc.x) / f32(max(width, 1)),
        f32(pc.y) / f32(max(height, 1)),
    );

    // Ink over paper: black strokes on a white ground, matching the golden fg/bg.
    let fg = vec3<f32>(0.0, 0.0, 0.0);
    let bg = vec3<f32>(1.0, 1.0, 1.0);

    let result = hatching_apply(
        scene.rgb, screen_pos, hatching_params.enabled != 0u, fg, bg,
        hatching_params.angle0, hatching_params.angle1, hatching_params.angle2,
        hatching_params.threshold0, hatching_params.threshold1, hatching_params.threshold2,
        hatching_params.frequency, hatching_params.thickness,
    );

    textureStore(hatching_out, pc, vec4<f32>(result, scene.a));
}
