//! GPU groom-global density-LOD importance fold.
//!
//! Compute twin of the per-element body of the CPU golden reference
//! `prism_render_architecture::hair::decimation::compute_importance` — the
//! final density-LOD step that folds each render strand's blended `(length,
//! curvature, authored)` triple (produced by `hair_binding_metrics`) into a
//! single normalized importance in `[0, 1]`. `UE5` Groom biases which strands
//! survive as a groom recedes by exactly this kind of length/curvature/thickness
//! priority; here it decides the nested decimation ranking.
//!
//! One invocation owns one render-strand importance. It reads that strand's
//! blended metric triple, normalizes length and curvature by their groom-global
//! maxima, clamps the already-normalized authored priority, blends the three by
//! the artist weights and renormalizes by the weight sum, writing the clamped
//! `[0, 1]` scalar to `out_importance[i]`.
//!
//! Host / other-kernel work: the two groom-global maxima (`max_len`, `max_curv`
//! over every blended triple) are a reduction, supplied here as immediate
//! params — the same split the sibling `hair_binding_metrics` documents, where
//! the groom-global reduction stays host/other-kernel and this kernel owns only
//! the embarrassingly-parallel per-strand fold. Because `max` over finite `f32`
//! is exact and order-independent, a host reduction and any `GPU` reduction
//! yield bit-identical maxima, so feeding them as params does not perturb
//! faithfulness. The ranking sort that consumes this importance
//! (`build_decimation_order`) is a groom-global permutation and stays a CPU
//! pass, like `self_collision`'s Gauss-Seidel order.
//!
//! Bit-faithfulness: the guard order and arithmetic mirror the CPU loop
//! exactly — `weight_sum <= 0` short-circuits to `0`; each metric divides by its
//! max only when that max is `> 0` (else `0`); `authored` is `clamp(.., 0, 1)`;
//! the blend is `w_length*len_n + w_curvature*curv_n + w_authored*auth` summed
//! left-to-right; and the result is `clamp(blended / weight_sum, 0, 1)`. The
//! weight sum is recomputed in-shader as `w_length + w_curvature + w_authored`
//! in the same order as the CPU, so the divisor matches bit-for-bit. A `GPU` is
//! now available, so this shader is compile- and type-checked through the same
//! render-world `ShaderCache` / `wesl` pipeline the runtime uses (see
//! `shader_tests`), with on-device dispatch/perf calibration still awaiting
//! real hardware.

/// Importance-fold dispatch parameters. `count` bounds the dispatch (one
/// invocation per render strand). `max_len` / `max_curv` are the groom-global
/// maxima of the blended length and curvature (host/other-kernel reduction).
/// `w_length` / `w_curvature` / `w_authored` are the artist importance weights;
/// only their ratios matter after the weight-sum renormalization.
struct HairImportanceParams {
    count: u32,
    max_len: f32,
    max_curv: f32,
    w_length: f32,
    w_curvature: f32,
    w_authored: f32,
}

// Per-strand blended density-LOD metric triple from `hair_binding_metrics`:
// `.x` = blended arc length, `.y` = blended curvature, `.z` = blended authored
// thickness (already in `[0, 1]`), `.w` = unused.
@group(0) @binding(0) var<storage, read> blend: array<vec4<f32>>;

// Per-strand normalized importance in `[0, 1]`, consumed by the CPU
// `build_decimation_order` ranking sort.
@group(0) @binding(1) var<storage, read_write> out_importance: array<f32>;

var<immediate> params: HairImportanceParams;

@compute @workgroup_size(64)
fn hair_importance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count {
        return;
    }

    // Recompute the weight sum in CPU order (length + curvature + authored).
    let weight_sum = params.w_length + params.w_curvature + params.w_authored;
    if weight_sum <= 0.0 {
        out_importance[i] = 0.0;
        return;
    }

    let triple = blend[i];

    // Normalize length / curvature by their groom-global maxima, guarding the
    // all-zero case exactly as the CPU (`max > 0` else contribute `0`).
    var len_n = 0.0;
    if params.max_len > 0.0 {
        len_n = triple.x / params.max_len;
    }
    var curv_n = 0.0;
    if params.max_curv > 0.0 {
        curv_n = triple.y / params.max_curv;
    }
    let auth = clamp(triple.z, 0.0, 1.0);

    let blended =
        params.w_length * len_n + params.w_curvature * curv_n + params.w_authored * auth;
    out_importance[i] = clamp(blended / weight_sum, 0.0, 1.0);
}
