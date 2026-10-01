// Marschner lobe importance-sampling twin.
//
// Device mirror of the CPU golden
// `prism_render_architecture::hair::scatter_lod::sample_lobe`. One thread per
// `(pdf, u)` tuple importance-samples one of the three Marschner lobes
// (`R` -> `TT` -> `TRT`) from a canonical uniform `u`, walking the cdf and
// returning the chosen lobe index, the sample remapped back into `[0, 1)` inside
// the chosen lobe's cdf slice (so the caller reuses it as a fresh stratified
// sample for that lobe's own term), and the lobe's probability.
//
// Each input is sanitised in-shader bit-faithfully to the golden: `u` is clamped
// to `[0, 1]` (a non-finite `u` collapses to `0`) and every pdf component is
// clamped to a finite, non-negative value. The cdf is walked `R -> TT -> TRT`
// and the final `TRT` lobe absorbs the upper end (including `u == 1`), so a valid
// lobe is always returned even for a degenerate all-zero pdf.
//
// The chosen lobe index and its probability are exact (a select and a pass of a
// sanitised component), so only the `(uu - lo) / p` remap divides, which the
// `GPU` may round a few `ULP` differently from the scalar reference; the twin is
// therefore matched against a tolerance (abs_diff < 1e-4 or rel_diff < 1e-3) on
// the floats, with the integer lobe index asserted exactly.
//
// The kernel uses only compares, `clamp`, `max`, add and one divide in the
// portable core-WGSL subset - no `exp`, `pow` or optional device feature - so
// the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: Marschner 2003 three-lobe hair BSDF importance sampling plus wgpu
// compute dispatch; no Unreal Engine source or derived code.

const EPS: f32 = 1e-6;

struct Params {
    // Number of (pdf, u) tuples in `inputs`; one 3-f32 sample per tuple out.
    elem_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per element, stride 4 f32: pdf.r, pdf.tt, pdf.trt, u.
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
// Per element, stride 3 f32: lobe index (as f32), remapped_u, pdf.
@group(0) @binding(2) var<storage, read_write> out_samples: array<f32>;

fn inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// Finite means not NaN (v == v) and not an infinity (|v| below +inf).
fn finite_scalar(v: f32) -> bool {
    return v == v && abs(v) < inf();
}

// Clamp to [0, 1]; a non-finite input collapses to 0 (the golden's `clamp01`,
// which also absorbs the golden's `if u.is_finite() { u } else { 0.0 }`).
fn clamp01(v: f32) -> f32 {
    return select(0.0, clamp(v, 0.0, 1.0), finite_scalar(v));
}

// Clamp to a finite, non-negative value (the golden's `sanitize_nonneg`):
// negative or non-finite values collapse to 0.
fn sanitize_nonneg(v: f32) -> f32 {
    return select(0.0, max(v, 0.0), finite_scalar(v));
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.elem_count) {
        return;
    }
    let si = 4u * i;
    let uu = clamp01(inputs[si + 3u]);
    let p_r = sanitize_nonneg(inputs[si]);
    let p_tt = sanitize_nonneg(inputs[si + 1u]);
    let p_trt = sanitize_nonneg(inputs[si + 2u]);

    let c_r = p_r;
    let c_tt = p_r + p_tt;

    // Walk the cdf R -> TT -> TRT; TRT absorbs the upper end (incl. uu == 1).
    var lobe: u32 = 2u;
    var lo: f32 = c_tt;
    var p: f32 = p_trt;
    if (uu < c_r) {
        lobe = 0u;
        lo = 0.0;
        p = p_r;
    } else if (uu < c_tt) {
        lobe = 1u;
        lo = c_r;
        p = p_tt;
    }

    var remapped_u: f32 = 0.0;
    if (p > EPS) {
        remapped_u = clamp((uu - lo) / p, 0.0, 1.0);
    }

    let oi = 3u * i;
    out_samples[oi] = f32(lobe);
    out_samples[oi + 1u] = remapped_u;
    out_samples[oi + 2u] = p;
}
