// Near-/far-field scatter-blend twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::scatter_lod::scatter_blend`. One thread per
// projected fibre width maps that width to a continuous near->far blend factor
// (d'Eon 2011 energy-conserving near-/far-field split): `0` = pure near-field
// (wide / close-up fibre), `1` = pure far-field (sub-pixel fibre), linearly
// ramped across the transition band so the §4 LOD ladder crosses the boundary
// without a pop.
//
// The screen-space thresholds (`near_px >= far_px >= 0`, both finite) are a
// per-dispatch constant, so the host sanitises them once with the golden's
// `ScatterLodThresholds::sanitized` and uploads the finished `near_px`/`far_px`
// as uniforms; the per-element work is sanitising the width and evaluating the
// ramp. A non-finite width collapses to `0` (treated as the finest footprint,
// blend `1`), bit-faithfully to the golden.
//
// The hard `0.0`/`1.0` saturation returns are exact; only the in-band
// `(near - w) / span` ramp divides, which the GPU may round a few ULP
// differently, so the twin is matched against a tolerance (abs_diff < 1e-4 or
// rel_diff < 1e-3) rather than bit-for-bit.
//
// The kernel uses only compares, `max`, a divide and `clamp` in the portable
// core-WGSL subset - no `exp`, `pow` or optional device feature - so the twin
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: d'Eon 2011 near-/far-field hair scattering split plus wgpu compute
// dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of fibre widths in `widths` / `out_blends`.
    width_count: u32,
    // Sanitised near threshold (pure near-field at or above this width).
    near_px: f32,
    // Sanitised far threshold (pure far-field at or below this width).
    far_px: f32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per element, one f32: the projected fibre width in pixels.
@group(0) @binding(1) var<storage, read> widths: array<f32>;
// Per element, one f32: the near->far blend factor in [0, 1].
@group(0) @binding(2) var<storage, read_write> out_blends: array<f32>;

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
    if (i >= params.width_count) {
        return;
    }
    let near = params.near_px;
    let far = params.far_px;

    // Non-finite widths collapse to 0 (finest footprint); negatives clamp to 0.
    let raw = widths[i];
    let w = select(0.0, max(raw, 0.0), finite_scalar(raw));

    var blend: f32;
    if (w >= near) {
        blend = 0.0;
    } else if (w <= far) {
        blend = 1.0;
    } else {
        let span = near - far;
        if (span <= EPS) {
            // Defensive: a strictly in-band width only exists when span > 0.
            blend = 1.0;
        } else {
            blend = clamp((near - w) / span, 0.0, 1.0);
        }
    }
    out_blends[i] = blend;
}
