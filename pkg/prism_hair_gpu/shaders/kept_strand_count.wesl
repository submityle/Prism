// Kept-strand-count twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::cluster::kept_strand_count`. One thread per
// cluster maps a `(total, ratio)` pair to the integer number of strands kept:
// the keep ratio (from the continuous decimation ramp, see
// `strand_keep_ratio.wesl`) is clamped into `[0, 1]`, scaled by the member
// count, rounded to the nearest strand and clamped to never exceed `total`.
// A non-finite ratio collapses to `0` kept, bit-faithfully to the golden.
//
// Rounding is the subtle part: Rust's `f32::round` rounds halves *away from
// zero* (2.5 -> 3), whereas WGSL's built-in `round` rounds halves *to even*
// (2.5 -> 2). To stay bit-identical to the golden the kernel never calls the
// built-in `round`; it computes `floor(scaled + 0.5)`, which reproduces
// round-half-away-from-zero exactly for the non-negative `scaled` here. The
// counts are therefore matched with integer equality (not a tolerance) for
// member counts in the f32-exact integer range (strand counts, well under
// 2^24), which is the only regime a groom ever reaches.
//
// The kernel uses only clamp/compare/mul/floor/convert in the portable
// core-WGSL subset - no `exp`, `pow`, built-in `round` or optional device
// feature - so the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: screen-footprint strand decimation member count plus wgpu compute
// dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of (total, ratio) pairs in `totals` / `ratios` / `out_counts`.
    elem_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per element, one u32: the cluster member (strand) count.
@group(0) @binding(1) var<storage, read> totals: array<u32>;
// Per element, one f32: the continuous keep ratio to apply.
@group(0) @binding(2) var<storage, read> ratios: array<f32>;
// Per element, one u32: the kept strand count in `[0, total]`.
@group(0) @binding(3) var<storage, read_write> out_counts: array<u32>;

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
    let total = totals[i];
    let raw = ratios[i];

    // A non-finite ratio collapses to 0; a finite ratio clamps into [0, 1].
    let r = select(0.0, clamp(raw, 0.0, 1.0), finite_scalar(raw));
    let scaled = f32(total) * r;

    // Round half away from zero (scaled >= 0) to match Rust `f32::round`, not
    // WGSL's round-to-even built-in.
    let kept = floor(scaled + 0.5);
    let kept_u = u32(max(kept, 0.0));
    out_counts[i] = min(kept_u, total);
}
