// Per-pixel analytic sub-pixel line-coverage twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::line_coverage::pixel_coverage`
// (`coverage_map` for the batch form). One thread per pixel centre computes the
// closed-form fraction of that pixel a single width-carrying fibre segment
// covers, the deterministic analytic anti-aliasing `alpha` the hair rasteriser
// uses for sub-pixel strands.
//
// The pixel is a round coverage kernel of radius `0.5`; the fibre is a capsule
// of half-width `width * 0.5`; coverage is the monotone trapezoidal ramp
// `clamp(width*0.5 + 0.5 - d, 0, 1)`, where `d` is the point-to-segment distance
// from the pixel centre to the segment `[a, b]` (projection clamped to the
// segment, degenerate zero-length segment collapsing to the point distance).
// Closed form, no iteration, only `sqrt` (via `length`) for the distance, no
// transcendental step — exactly the golden's model.
//
// The segment endpoints and stroke width come from `Params` so the shader
// never duplicates host state; the per-thread variation is the pixel centre.
// The stroke width is sanitized bit-faithfully to the golden's
// `sanitized_width`: negative, NaN and +/-inf widths collapse to `0` so the
// coverage stays finite and in `[0, 1]` for every input.
//
// Provenance: analytic line-AA coverage (UE5 Groom / Weta / Pixar line-AA
// literature) plus `wgpu` compute dispatch; no Unreal Engine source or derived
// code.

struct Params {
    // xy = segment start `a`, zw = segment end `b`, in pixel coordinates.
    segment: vec4<f32>,
    width: f32,
    pixel_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per pixel: xy = pixel centre in pixel coordinates.
@group(0) @binding(1) var<storage, read> centers: array<vec2<f32>>;
// Per pixel: analytic coverage ratio in `[0, 1]`.
@group(0) @binding(2) var<storage, read_write> coverage: array<f32>;

// Degenerate-segment / distance epsilon, matching the golden's module `EPS`.
const EPS: f32 = 1e-6;
// Largest finite f32; `w <= MAX_FINITE_F32` rejects +inf (and `w == w` rejects
// NaN), so the guard reproduces `f32::is_finite() && w > 0.0` exactly.
const MAX_FINITE_F32: f32 = 3.40282347e+38;

// Mirrors the golden's `ScreenSegment::sanitized_width`: a finite, strictly
// positive width passes through unchanged; everything else (negative, zero,
// NaN, +/-inf) collapses to `0.0`, with no floating-point equality on a literal.
fn sanitized_width(w: f32) -> f32 {
    if w == w && w > 0.0 && w <= MAX_FINITE_F32 {
        return w;
    }
    return 0.0;
}

// Shortest Euclidean distance from `p` to the segment `[a, b]`, projection
// parameter clamped to `[0, 1]`; a degenerate (near-zero-length) segment
// collapses to the point distance `|p - a|`, mirroring the golden exactly.
fn point_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let len_sq = dot(ab, ab);
    let ap = p - a;
    if abs(len_sq) < EPS {
        return length(ap);
    }
    let t = clamp(dot(ap, ab) / len_sq, 0.0, 1.0);
    let c = a + t * ab;
    return length(p - c);
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.pixel_count {
        return;
    }
    let a = params.segment.xy;
    let b = params.segment.zw;
    let half_width = sanitized_width(params.width) * 0.5;
    let d = point_segment_distance(centers[idx], a, b);
    coverage[idx] = clamp(half_width + 0.5 - d, 0.0, 1.0);
}
