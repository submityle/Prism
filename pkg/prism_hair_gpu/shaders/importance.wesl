// Density-LOD importance-fold twin: evaluates on-device the same per-strand
// importance scalar the CPU golden
// `prism_render_architecture::hair::decimation::compute_importance` computes,
// one thread per strand.
//
// Continuous density LOD keeps fewer render strands as a groom recedes; to pick
// which strands survive to the lowest counts the reference folds three
// per-strand metrics into a single importance in [0, 1]: normalized arc length
// (longer strands read more on the silhouette), normalized accumulated
// curvature (curls / flyaways matter) and an already-normalized authored artist
// priority. Each metric is normalized by the groom-wide maximum so scene scale
// drops out, blended by artist weights, then renormalized by the weight sum —
// the density-LOD bias UE5 Groom and HairWorks apply. The importance then feeds
// `decimation_priority` and the host ranking sort that builds the nested,
// pop-free decimation order; this kernel is only the per-strand fold, the part
// that maps cleanly to one GPU thread per strand.
//
// Groom-level reductions stay on the host: the two maxima `max_len` / `max_curv`
// and the `weight_sum` are computed host-side and passed as uniform params, the
// same split the sibling `hair_importance.wesl` (compile-time twin in
// `prism_render_scene`) documents. A maximum over finite `f32` is exact and
// order-independent, so the host reduction is bit-identical to any GPU
// reduction; only the per-strand blend `w_l*len_n + w_c*curv_n + w_a*auth` is a
// multiply-add chain a GPU may fuse, so CPU and GPU agree to within the
// documented fma tolerance rather than bit-for-bit. The division and the two
// normalizations are plain IEEE operations, so the sole divergence source is
// that final fused blend.
//
// Buffer layout: `params` carries the three artist weights, the two groom-level
// maxima and the strand count; `lengths[i]` / `curvatures[i]` / `authored[i]`
// are strand `i`'s raw arc length, accumulated curvature and authored priority;
// `importances` receives one folded scalar per strand at index `i`.
//
// Provenance: standard importance-weighted density-LOD metric fold plus wgpu
// compute dispatch; no Unreal Engine source or derived code.

struct Params {
    // Artist weight on normalized arc length (the CPU `weights.length`).
    w_length: f32,
    // Artist weight on normalized curvature (the CPU `weights.curvature`).
    w_curvature: f32,
    // Artist weight on the authored priority (the CPU `weights.authored`).
    w_authored: f32,
    // Groom-wide maximum arc length; `<= 0` disables length normalization.
    max_len: f32,
    // Groom-wide maximum accumulated curvature; `<= 0` disables curvature norm.
    max_curv: f32,
    // Number of strands, bounding the dispatch.
    strand_count: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Per-strand raw arc length (mirrors the CPU `lengths` slice).
@group(0) @binding(1) var<storage, read> lengths: array<f32>;
// Per-strand accumulated curvature (mirrors `curvatures`).
@group(0) @binding(2) var<storage, read> curvatures: array<f32>;
// Per-strand authored priority, clamped to [0, 1] here (mirrors `authored`).
@group(0) @binding(3) var<storage, read> authored: array<f32>;
// Per-strand folded importance in [0, 1] the host then ranks.
@group(0) @binding(4) var<storage, read_write> importances: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.strand_count {
        return;
    }
    // Weight sum matches the CPU left-associative `l + c + a`; a non-positive
    // sum means every metric is muted, so the importance is the lowest rank 0.
    let weight_sum = params.w_length + params.w_curvature + params.w_authored;
    if weight_sum <= 0.0 {
        importances[i] = 0.0;
        return;
    }
    // Normalize by the groom-wide maxima (scene scale drops out); a zero maximum
    // yields a zero normalized metric, mirroring the CPU guards exactly.
    var len_n = 0.0;
    if params.max_len > 0.0 {
        len_n = lengths[i] / params.max_len;
    }
    var curv_n = 0.0;
    if params.max_curv > 0.0 {
        curv_n = curvatures[i] / params.max_curv;
    }
    let auth = clamp(authored[i], 0.0, 1.0);
    // Same left-associative blend order as the CPU golden; the only fma-fusable
    // step in the whole kernel.
    let blended = params.w_length * len_n + params.w_curvature * curv_n + params.w_authored * auth;
    importances[i] = clamp(blended / weight_sum, 0.0, 1.0);
}
