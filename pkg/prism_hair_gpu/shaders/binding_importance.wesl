// Density-LOD binding-importance twin: evaluates on-device the same per-render-
// strand importance the CPU golden
// `prism_render_architecture::hair::density_lod::binding_importances` computes,
// one thread per binding.
//
// The density-LOD driver measures each *guide* strand's arc length, accumulated
// curvature and authored thickness once, then must propagate those guide metrics
// onto every *render* strand before it can rank which strands survive as the
// groom recedes. A render strand is skinned to up to four guides with
// barycentric-style weights, so it inherits the weight-blended length, curvature
// and thickness of the guides it is bound to (out-of-range guides and
// non-positive weights contribute nothing) — exactly the guide->render
// propagation UE5 Groom applies before biasing decimation. The blended triple is
// then folded into one importance in [0, 1] by the same normalize / weight /
// renormalize step as the `compute_importance` fold. This kernel owns the whole
// per-binding gather + fold, the part that maps cleanly to one GPU thread per
// render strand.
//
// Groom-level reductions stay on the host: the two maxima `max_len` / `max_curv`
// are reduced over the *blended* per-binding metrics host-side (bit-for-bit the
// golden's own reduction inside `compute_importance`) and passed as uniform
// params. A maximum over finite `f32` is exact and order-independent, so the
// host reduction is bit-identical to any GPU reduction; the divergence sources
// are the weighted gather sums and the final blend, both multiply-add chains a
// GPU may fuse, so CPU and GPU agree to within the documented fma tolerance
// rather than bit-for-bit.
//
// Buffer layout: `params` carries the three artist weights, the two groom-level
// maxima, the binding count and the guide count; `guides[t]` / `binding_weights[t]`
// are render strand `t`'s four guide indices and weights; `metric_lengths` /
// `metric_curvatures` / `metric_authored` are the per-guide measured metrics;
// `importances` receives one folded scalar per binding at index `t`.
//
// Provenance: standard guide->render metric propagation plus importance-weighted
// density-LOD fold plus wgpu compute dispatch; no Unreal Engine source or
// derived code.

struct Params {
    // Artist weight on normalized arc length (the CPU `weights.length`).
    w_length: f32,
    // Artist weight on normalized curvature (the CPU `weights.curvature`).
    w_curvature: f32,
    // Artist weight on the authored priority (the CPU `weights.authored`).
    w_authored: f32,
    // Groom-wide maximum blended arc length; `<= 0` disables length norm.
    max_len: f32,
    // Groom-wide maximum blended curvature; `<= 0` disables curvature norm.
    max_curv: f32,
    // Number of render-strand bindings, bounding the dispatch.
    binding_count: u32,
    // Number of measured guides; guide indices `>= guide_count` are ignored.
    guide_count: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-binding guide indices, nearest first; entries `>= guide_count` are ignored.
@group(0) @binding(1) var<storage, read> guides: array<vec4<u32>>;
// Per-binding guide weights, matching `guides`; non-positive entries are ignored.
@group(0) @binding(2) var<storage, read> binding_weights: array<vec4<f32>>;
// Per-guide measured arc length (mirrors `GuideMetrics::lengths`).
@group(0) @binding(3) var<storage, read> metric_lengths: array<f32>;
// Per-guide measured accumulated curvature (mirrors `GuideMetrics::curvatures`).
@group(0) @binding(4) var<storage, read> metric_curvatures: array<f32>;
// Per-guide normalized authored thickness (mirrors `GuideMetrics::authored`).
@group(0) @binding(5) var<storage, read> metric_authored: array<f32>;
// Per-binding folded importance in [0, 1] the host then ranks.
@group(0) @binding(6) var<storage, read_write> importances: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.binding_count {
        return;
    }

    let g = guides[t];
    let w = binding_weights[t];

    // Weighted gather over the four bound guides, in the same component order as
    // the CPU `binding.guides.iter().zip(binding.weights.iter())` loop: a guide
    // only contributes when its weight is positive and its index is in range.
    var len_acc = 0.0;
    var curv_acc = 0.0;
    var auth_acc = 0.0;

    if w.x > 0.0 && g.x < params.guide_count {
        len_acc = len_acc + w.x * metric_lengths[g.x];
        curv_acc = curv_acc + w.x * metric_curvatures[g.x];
        auth_acc = auth_acc + w.x * metric_authored[g.x];
    }
    if w.y > 0.0 && g.y < params.guide_count {
        len_acc = len_acc + w.y * metric_lengths[g.y];
        curv_acc = curv_acc + w.y * metric_curvatures[g.y];
        auth_acc = auth_acc + w.y * metric_authored[g.y];
    }
    if w.z > 0.0 && g.z < params.guide_count {
        len_acc = len_acc + w.z * metric_lengths[g.z];
        curv_acc = curv_acc + w.z * metric_curvatures[g.z];
        auth_acc = auth_acc + w.z * metric_authored[g.z];
    }
    if w.w > 0.0 && g.w < params.guide_count {
        len_acc = len_acc + w.w * metric_lengths[g.w];
        curv_acc = curv_acc + w.w * metric_curvatures[g.w];
        auth_acc = auth_acc + w.w * metric_authored[g.w];
    }

    // Fold the blended triple exactly like `compute_importance`. A non-positive
    // weight sum mutes every metric, so the importance is the lowest rank 0.
    let weight_sum = params.w_length + params.w_curvature + params.w_authored;
    if weight_sum <= 0.0 {
        importances[t] = 0.0;
        return;
    }
    var len_n = 0.0;
    if params.max_len > 0.0 {
        len_n = len_acc / params.max_len;
    }
    var curv_n = 0.0;
    if params.max_curv > 0.0 {
        curv_n = curv_acc / params.max_curv;
    }
    let auth = clamp(auth_acc, 0.0, 1.0);
    // Same left-associative blend order as the CPU golden.
    let blended = params.w_length * len_n + params.w_curvature * curv_n + params.w_authored * auth;
    importances[t] = clamp(blended / weight_sum, 0.0, 1.0);
}
