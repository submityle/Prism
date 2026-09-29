//! GPU per-binding density-LOD metric blend.
//!
//! Compute twin of the per-binding blend loop inside the CPU golden reference
//! `prism_render_architecture::hair::density_lod::binding_importances` (the part
//! before the groom-global `compute_importance` normalization). Density LOD
//! propagates each guide's measured metrics (arc length, curvature, authored
//! thickness — see `hair_guide_metrics`) onto every *render* strand through its
//! binding weights: a render strand inherits the weight-blended metrics of the
//! up-to-four guides it is skinned to, exactly as `UE5` Groom carries guide
//! attributes onto the interpolated strands. There are far more render strands
//! than guides, so this blend is the widest fan-out in the density-LOD path and
//! benefits most from running on the `GPU`.
//!
//! One invocation owns one render-strand binding. It reads that binding's four
//! guide indices and weights, accumulates `weight * metric` for each valid
//! guide (`weight > 0` and index in range, contributing nothing otherwise — the
//! same guard as the CPU), and writes the blended triple to `out_blend[b]` as
//! `vec4(length, curvature, authored, 0)`.
//!
//! Host / other-kernel work: `guide_metrics` (the input `guide_metrics` buffer)
//! carries the *normalized* authored thickness already (its groom-global
//! max-radius reduction stays host-side, see `hair_guide_metrics`), and the
//! final fold — `compute_importance`'s max-normalization of the blended length
//! and curvature across every binding plus the weighted mix into a single
//! `[0, 1]` importance — is a groom-global reduction that likewise stays
//! host/other-kernel. This kernel owns only the parallel per-binding blend.
//!
//! Bit-faithfulness: the four guides are accumulated in index order `0..4` with
//! the same `weight * metric` products and left-to-right addition as the CPU
//! loop, and the `weight > 0.0 && idx < guide_count` guard matches exactly, so
//! the blended triple is bit-for-bit the CPU `len_acc` / `curv_acc` / `auth_acc`.
//! A `GPU` is now available, so this shader is compile- and type-checked
//! through the same render-world `ShaderCache` / `wesl` pipeline the runtime
//! uses (see `shader_tests`), with on-device dispatch/perf calibration still
//! awaiting real hardware.

/// Number of guides that influence one render strand. Mirrors the CPU
/// `interpolation::GUIDE_INFLUENCE_COUNT`.
const HAIR_GUIDE_INFLUENCE_COUNT: u32 = 4u;

/// Blend dispatch parameters. `binding_count` bounds the dispatch (one
/// invocation per render-strand binding); `guide_count` is the number of valid
/// guide metrics, so out-of-range guide indices contribute nothing.
struct HairBindingMetricParams {
    binding_count: u32,
    guide_count: u32,
}

/// One render-strand binding's guide influences: up to four guide indices and
/// their re-normalized weights. Mirrors the density-LOD-relevant fields of the
/// CPU `interpolation::RenderStrandBinding`.
struct HairBindingInfluence {
    guides: vec4<u32>,
    weights: vec4<f32>,
}

// Per-guide normalized metric triple from `hair_guide_metrics` + host
// normalization: `.x` = arc length, `.y` = curvature, `.z` = normalized
// authored thickness in `[0, 1]`.
@group(0) @binding(0) var<storage, read> guide_metrics: array<vec4<f32>>;

// Per-binding guide influences.
@group(0) @binding(1) var<storage, read> bindings: array<HairBindingInfluence>;

// Per-binding blended metric triple: `vec4(length, curvature, authored, 0)`.
@group(0) @binding(2) var<storage, read_write> out_blend: array<vec4<f32>>;

var<immediate> params: HairBindingMetricParams;

@compute @workgroup_size(64)
fn hair_binding_metrics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= params.binding_count {
        return;
    }

    let guides = bindings[b].guides;
    let weights = bindings[b].weights;

    var len_acc = 0.0;
    var curv_acc = 0.0;
    var auth_acc = 0.0;

    // Accumulate the four influences in index order, matching the CPU zip loop.
    for (var k = 0u; k < HAIR_GUIDE_INFLUENCE_COUNT; k = k + 1u) {
        let weight = weights[k];
        let idx = guides[k];
        // Same guard as the CPU: only strictly-positive weights and in-range
        // guide indices contribute.
        if weight > 0.0 && idx < params.guide_count {
            let m = guide_metrics[idx];
            len_acc = len_acc + weight * m.x;
            curv_acc = curv_acc + weight * m.y;
            auth_acc = auth_acc + weight * m.z;
        }
    }

    out_blend[b] = vec4<f32>(len_acc, curv_acc, auth_acc, 0.0);
}
