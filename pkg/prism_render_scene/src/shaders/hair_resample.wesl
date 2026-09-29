//! GPU per-strand arc-length resampling for hair groom import.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::groom_import::resample_strand`. Authored
//! grooms (Alembic caches from `UE5` Groom, AMD `TressFX` assets, DCC exports)
//! arrive as guide *polylines* with arbitrary, non-uniform control-point counts
//! and uneven spacing, but the strand solver and the guide-to-render
//! interpolator both assume a *uniform* control-point count per guide with
//! roughly even arc-length spacing (so rest lengths are stable and blends line
//! up point-for-point). Production engines resample every guide on import; this
//! kernel is the massively parallel form of that step: one invocation owns one
//! resampled strand slot and reparameterizes its raw polyline to a fixed
//! `points_per_strand` count by even arc length, pinning the first and last
//! output to the true root and tip so no strand shrinks (design §2 数据模型).
//!
//! Host bookkeeping stays on the CPU: the caller compacts and validates raw
//! strand ranges (dropping empty or out-of-range guides, exactly like
//! `resample_groom`) and hands this kernel a dense `strand_ranges` array plus a
//! shared flat `raw_points` buffer, then reads back a fixed-stride
//! `out_points`. Only the per-strand resampling math fans out here; the
//! `len == 1` and zero-total-length (all-coincident) degenerate cases both emit
//! `points_per_strand` copies of the root, bit-for-bit the CPU golden.
//!
//! Bit-faithfulness: arc lengths use hand-written `sqrt(dot(d, d))` rather than
//! a built-in `length()` to match the solver's `dynamics::Vec3::length`, the
//! cumulative arc length is accumulated with the same left-to-right addition
//! order as the CPU `windows(2)` loop via an incremental segment cursor, and
//! the target distance is spelled `total * f32(j) / f32(ppts - 1)` with the
//! same left-associative grouping. A `GPU` is now available, so this shader is
//! compile- and type-checked through the same render-world `ShaderCache` /
//! `wesl` pipeline the runtime uses (see `shader_tests`), with on-device
//! dispatch/perf calibration still awaiting real hardware.

/// Resample dispatch parameters. `strand_count` bounds the dispatch (one
/// invocation per resampled strand); `points_per_strand` is the uniform output
/// stride, clamped to at least `2` (a strand needs a root and a tip).
struct HairResampleParams {
    strand_count: u32,
    points_per_strand: u32,
}

/// One compacted raw guide slice into `raw_points`: control points
/// `start .. start + len`. The host has already validated the range (non-empty,
/// in-bounds), mirroring `resample_groom`.
struct HairRawStrandRange {
    start: u32,
    len: u32,
}

// Shared flat buffer of raw authored control points; `.xyz` is the position,
// `.w` is unused padding.
@group(0) @binding(0) var<storage, read> raw_points: array<vec4<f32>>;

// Per-strand compacted ranges into `raw_points`.
@group(0) @binding(1) var<storage, read> strand_ranges: array<HairRawStrandRange>;

// Fixed-stride resampled output: strand `s` writes
// `out_points[s * points_per_strand .. + points_per_strand]`, each a
// `vec4(pos, 0)`.
@group(0) @binding(2) var<storage, read_write> out_points: array<vec4<f32>>;

var<immediate> params: HairResampleParams;

@compute @workgroup_size(64)
fn hair_resample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.strand_count {
        return;
    }

    // `target = target_points.max(2)`: a strand needs a root and a tip.
    var ppts = params.points_per_strand;
    if ppts < 2u {
        ppts = 2u;
    }

    let start = strand_ranges[s].start;
    let len = strand_ranges[s].len;
    let base = s * ppts;

    // Single raw point: emit `ppts` copies of the root (a valid zero-length
    // strand). The host never feeds `len == 0` (empty ranges are skipped).
    if len == 1u {
        let root = raw_points[start];
        for (var j = 0u; j < ppts; j = j + 1u) {
            out_points[base + j] = vec4<f32>(root.xyz, 0.0);
        }
        return;
    }

    let last_raw = len - 1u;

    // Pass 1: total arc length, accumulated in the same order as the CPU
    // `windows(2)` loop.
    var total = 0.0;
    for (var i = 0u; i < last_raw; i = i + 1u) {
        let d = raw_points[start + i + 1u].xyz - raw_points[start + i].xyz;
        total = total + sqrt(dot(d, d));
    }

    let root = raw_points[start];

    // All points coincide (zero total length): emit the root `ppts` times.
    if total <= 0.0 {
        for (var j = 0u; j < ppts; j = j + 1u) {
            out_points[base + j] = vec4<f32>(root.xyz, 0.0);
        }
        return;
    }

    // Pass 2: walk an incremental segment cursor. `cum0 == cumulative[seg]` is
    // rebuilt by the same left-to-right addition as pass 1, so it matches the
    // CPU `cumulative` array value-for-value; `seg` only ever advances.
    var seg = 0u;
    var cum0 = 0.0;
    for (var j = 0u; j < ppts; j = j + 1u) {
        // Target arc-length distance for output `j`, exact at both ends.
        var d = total;
        if j + 1u != ppts {
            d = total * f32(j) / f32(ppts - 1u);
        }

        // Advance while `cumulative[seg + 1] < d` (i.e. `cum0 + seglen < d`).
        loop {
            if seg >= last_raw {
                break;
            }
            let e = raw_points[start + seg + 1u].xyz - raw_points[start + seg].xyz;
            let seglen = sqrt(dot(e, e));
            let cum1 = cum0 + seglen;
            if cum1 < d {
                cum0 = cum1;
                seg = seg + 1u;
            } else {
                break;
            }
        }

        let a = raw_points[start + seg].xyz;
        let b = raw_points[start + seg + 1u].xyz;
        // `span == cumulative[seg + 1] - cumulative[seg]`, recomputed the same
        // way as the arc-length accumulation above.
        let sd = b - a;
        let span = sqrt(dot(sd, sd));
        var t = 0.0;
        if span > 0.0 {
            t = (d - cum0) / span;
        }
        // lerp(a, b, t) = a + (b - a) * t, in the same fixed order as CPU.
        let p = a + (b - a) * t;
        out_points[base + j] = vec4<f32>(p, 0.0);
    }
}
