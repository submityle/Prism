//! GPU per-guide density-LOD metric measurement.
//!
//! Compute twin of the per-guide half of the CPU golden reference
//! `prism_render_architecture::hair::density_lod::guide_metrics` (which in turn
//! calls `decimation::strand_arc_length` and `decimation::strand_curvature`).
//! Importance-weighted density LOD (the bias `UE5` Groom and `HairWorks` apply
//! so long, curly or thick guides survive to the lowest strand counts) needs a
//! per-guide metric triple measured once from the resampled groom: arc length,
//! accumulated curvature and authored root-radius thickness. That measurement
//! is embarrassingly parallel — one invocation owns one guide strand and reads
//! only that strand's own control points — so it fans out cleanly here while the
//! solver's heavier passes run.
//!
//! One invocation owns one guide strand. It reads the strand's control-point
//! slice from the shared flat `points` pool (via a host-validated
//! `strand_ranges[s]` offset, the same compacted-range discipline as
//! `hair_resample`), accumulates arc length and curvature, and reads the
//! authored root radius, writing the raw triple to `out_metrics[s]` as
//! `vec4(arc_length, curvature, root_radius, 0)`.
//!
//! Host bookkeeping stays on the CPU. The `authored` field of `GuideMetrics`
//! is the root radius normalized by the groom's *maximum* root radius — a
//! groom-global reduction — so this kernel emits the raw `max(root_radius, 0)`
//! per guide and the host divides by the reduced maximum (uniformly `0` when
//! every guide has zero radius), exactly as `guide_metrics` does after the
//! per-guide loop. Likewise the downstream per-binding weight blend and the
//! `compute_importance` max-normalization stay host/other-kernel work; this
//! kernel owns only the parallel per-guide measurement.
//!
//! Bit-faithfulness: arc-length segments use hand-written `sqrt(dot(d, d))` to
//! match `interpolation::Vec3::length`, summed left-to-right in the same order
//! as the CPU `for i in 1..len` loop; curvature sums `1 - dot(t_in, t_out)`
//! over interior vertices with the same `normalize_or(ZERO)` guard
//! (`len_sq <= f32::EPSILON`, spelled here as the exact `0x1p-23`), so a
//! degenerate coincident segment contributes its full `1.0` turn just like the
//! CPU. A `GPU` is now available, so this shader is compile- and type-checked
//! through the same render-world `ShaderCache` / `wesl` pipeline the runtime
//! uses (see `shader_tests`), with on-device dispatch/perf calibration still
//! awaiting real hardware.

/// Squared-length guard for tangent normalization. Exactly `f32::EPSILON`
/// (`2^-23`), mirroring `interpolation::Vec3::normalize_or`, so the GPU and CPU
/// agree bit-for-bit on which segments collapse to a zero tangent.
const HAIR_METRIC_EPS: f32 = 0x1p-23;

/// Metric dispatch parameters. `strand_count` bounds the dispatch (one
/// invocation per guide strand).
struct HairGuideMetricParams {
    strand_count: u32,
}

/// One compacted control-point slice into `points`: `start .. start + len`.
/// The host has already validated the range, mirroring the way `guide_metrics`
/// reads `groom.strand(i)`.
struct HairMetricStrandRange {
    start: u32,
    len: u32,
}

// Shared flat buffer of resampled guide control points; `.xyz` is the position,
// `.w` is unused padding.
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;

// Per-strand compacted ranges into `points`.
@group(0) @binding(1) var<storage, read> strand_ranges: array<HairMetricStrandRange>;

// Per-strand authored root radius (raw; the host clamps and normalizes).
@group(0) @binding(2) var<storage, read> root_radii: array<f32>;

// Per-strand raw metric triple: `vec4(arc_length, curvature, root_radius, 0)`.
@group(0) @binding(3) var<storage, read_write> out_metrics: array<vec4<f32>>;

var<immediate> params: HairGuideMetricParams;

// Unit tangent or the zero vector when the segment is (near) degenerate.
// Mirrors `interpolation::Vec3::normalize_or(ZERO)`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if len_sq <= HAIR_METRIC_EPS {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return v * (1.0 / sqrt(len_sq));
}

@compute @workgroup_size(64)
fn hair_guide_metrics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.strand_count {
        return;
    }

    let start = strand_ranges[s].start;
    let len = strand_ranges[s].len;

    // Arc length: sum of segment lengths, left-to-right. `< 2` points -> 0.
    var arc_length = 0.0;
    if len >= 2u {
        for (var i = 0u; i + 1u < len; i = i + 1u) {
            let d = points[start + i + 1u].xyz - points[start + i].xyz;
            arc_length = arc_length + sqrt(dot(d, d));
        }
    }

    // Curvature: sum of `1 - dot(t_in, t_out)` over interior vertices. `< 3`
    // points -> 0. A coincident segment yields a zero tangent (dot 0) and thus
    // a full `1.0` turn, exactly like the CPU golden.
    var curvature = 0.0;
    if len >= 3u {
        for (var i = 1u; i + 1u < len; i = i + 1u) {
            let t_in = normalize_or_zero(points[start + i].xyz - points[start + i - 1u].xyz);
            let t_out = normalize_or_zero(points[start + i + 1u].xyz - points[start + i].xyz);
            curvature = curvature + (1.0 - dot(t_in, t_out));
        }
    }

    // Authored priority proxy: clamp negative radii to 0 (`root_radius.max(0)`);
    // the host normalizes by the groom-global maximum.
    let radius = max(root_radii[s], 0.0);

    out_metrics[s] = vec4<f32>(arc_length, curvature, radius, 0.0);
}
