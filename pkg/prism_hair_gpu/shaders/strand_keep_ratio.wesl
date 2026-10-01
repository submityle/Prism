// Continuous strand keep-ratio twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::cluster::strand_keep_ratio`. One thread per
// cluster projected pixel footprint maps that footprint to a continuous strand
// keep ratio in `[min_ratio, 1]`: at or above `full_px` every strand is kept
// (ratio 1); at or below `cull_px` only `min_ratio` survive; between the two the
// ratio ramps linearly, so a receding groom thins out without a pop (anti-pop
// continuous decimation LOD).
//
// The decimation thresholds (`full_px >= cull_px >= 0`, both finite, `min_ratio`
// in `[0, 1]`) are a per-dispatch constant, so the host sanitises them once with
// the golden's `DecimationThresholds::sanitized` and uploads the finished
// `full_px`/`cull_px`/`min_ratio` as uniforms; the per-element work is
// sanitising the footprint and evaluating the ramp. A non-finite or negative
// footprint collapses to 0 (the most-decimated footprint, ratio `min_ratio`),
// bit-faithfully to the golden.
//
// The hard saturation returns (`1.0`, `min_ratio`) are exact; only the in-band
// `(px - cull) / span` ramp divides, which the GPU may round a few ULP
// differently, so the twin is matched against a tolerance (abs_diff < 1e-4 or
// rel_diff < 1e-3) rather than bit-for-bit.
//
// The kernel uses only compares, `max`, a divide and multiply/add in the
// portable core-WGSL subset - no `exp`, `pow` or optional device feature - so
// the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: continuous screen-footprint strand decimation ramp plus wgpu
// compute dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of cluster footprints in `footprints` / `out_ratios`.
    elem_count: u32,
    // Sanitised full-keep footprint (ratio 1 at or above this).
    full_px: f32,
    // Sanitised cull footprint (ratio min_ratio at or below this).
    cull_px: f32,
    // Sanitised floor keep ratio in [0, 1].
    min_ratio: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per element, one f32: the cluster projected pixel footprint.
@group(0) @binding(1) var<storage, read> footprints: array<f32>;
// Per element, one f32: the strand keep ratio in [min_ratio, 1].
@group(0) @binding(2) var<storage, read_write> out_ratios: array<f32>;

// Degenerate-band guard matching the golden's `EPS`.
const EPS: f32 = 1.0e-6;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Finite means not NaN (v == v) and not an infinity (|v| below +inf).
fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < inf();
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.elem_count) {
        return;
    }
    let full = params.full_px;
    let cull = params.cull_px;
    let min_ratio = params.min_ratio;

    // Non-finite footprints collapse to 0 (finest decimation); negatives clamp.
    let raw = footprints[i];
    let px = select(0.0, max(raw, 0.0), finite_scalar(raw));

    var ratio: f32;
    if (px >= full) {
        ratio = 1.0;
    } else if (px <= cull) {
        ratio = min_ratio;
    } else {
        let span = full - cull;
        if (span <= EPS) {
            // Defensive: a strictly in-band footprint only exists when span > 0.
            ratio = min_ratio;
        } else {
            let blend = (px - cull) / span;
            ratio = min_ratio + (1.0 - min_ratio) * blend;
        }
    }
    out_ratios[i] = ratio;
}
