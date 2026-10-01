// Far-field roughness-gain twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::scatter_lod::far_field_roughness_gain`. One
// thread per `(blend, max_gain)` pair widens the hair lobes by
// `1 + blend * max_gain` as the projected footprint shrinks: the far-field blend
// (0 near-field .. 1 far-field) drives a roughness gain that integrates
// sub-pixel fibres and cuts aliasing, always `>= 1`.
//
// Each input is sanitised in-shader bit-faithfully to the golden: the blend is
// clamped to `[0, 1]` (a non-finite blend collapses to `0`) and a negative or
// non-finite `max_gain` collapses to `0` (no widening). A stray NaN therefore
// never poisons the gain.
//
// The `1 + blend * max_gain` is a single multiply-add the GPU may fuse where the
// scalar reference leaves it separate, so the twin is matched against a
// tolerance (abs_diff < 1e-4 or rel_diff < 1e-3) rather than bit-for-bit.
//
// The kernel uses only compares, `clamp`, `max` and a multiply-add in the
// portable core-WGSL subset - no `exp`, `pow` or optional device feature - so
// the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: d'Eon 2011 near-/far-field hair scattering split plus wgpu compute
// dispatch; no Unreal Engine source or derived code.

struct Params {
    // Number of (blend, max_gain) pairs in `inputs`; one gain per pair out.
    pair_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per element, stride 2 f32: blend, max_gain.
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
// Per element, one f32: the roughness gain in [1, 1 + max_gain].
@group(0) @binding(2) var<storage, read_write> out_gains: array<f32>;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Finite means not NaN (v == v) and not an infinity (|v| below +inf).
fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < inf();
}

// Clamp a blend to [0, 1]; a non-finite blend collapses to 0 (the golden's
// `clamp01`).
fn clamp01(v: f32) -> f32 {
    return select(0.0, clamp(v, 0.0, 1.0), finite_scalar(v));
}

// Clamp a gain to a finite, non-negative value (the golden's `sanitize_nonneg`):
// negative or non-finite values collapse to 0.
fn sanitize_nonneg(v: f32) -> f32 {
    return select(0.0, max(v, 0.0), finite_scalar(v));
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.pair_count) {
        return;
    }
    let si = 2u * i;
    let b = clamp01(inputs[si]);
    let g = sanitize_nonneg(inputs[si + 1u]);
    out_gains[i] = 1.0 + b * g;
}
