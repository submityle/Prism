//! `wgpu` compute twin of Prism's import-time arc-length resampler
//! (`prism_render_architecture::hair::groom_import::resample_strand` /
//! `resample_groom`).
//!
//! Groom import reparameterizes every raw guide polyline — authored with an
//! arbitrary control-point count — into a fixed `target_points` stride spaced
//! evenly by arc length, so the whole engine (dynamics edge rest lengths,
//! interpolation, LOD, raster) consumes one uniform-stride buffer. `UE5` Groom
//! and `TressFX` both resample at import for exactly this reason. This kernel is
//! that bake's isolated twin: one thread per strand walks the raw polyline's
//! cumulative arc length and emits `target_points` control points, so a passing
//! real-device parity test is direct evidence the arc-length walk and the
//! endpoint pinning port bit-faithfully.
//!
//! # Algorithm
//!
//! `target` is clamped to `>= 2` (a strand needs a root and a tip). The kernel
//! first sums the segment lengths to `total` in root→tip order, then walks a
//! persistent segment cursor: output `j`'s target distance is `total` at the tip
//! (`j + 1 == target`) and `total * j / (target - 1)` otherwise, and the point is
//! the linear interpolation inside the segment that distance lands in. The
//! cursor and its running cumulative distance advance in the same fixed order as
//! the reference's `cumulative[]` array, so the segment pick and the comparison
//! `cum0 + seglen < d` are bit-identical to the scalar side.
//!
//! # Bit-faithfulness
//!
//! The endpoints are pinned exactly: `j == 0` gives `d == 0`, so `t == 0` and the
//! output is the root vertex; `j == target - 1` gives `d == total == d1`, so
//! `(d - d0) / span == span / span == 1.0` (IEEE `x / x`) and the output is the
//! tip vertex — both bit-exact. Interior points divide `(d - d0) / span`, which a
//! `GPU` may evaluate with a fused multiply-add the scalar reference cannot, and
//! `length` may fuse its `sqrt(dot)` too, so interior positions can differ by a
//! few low-mantissa `ULP`. The parity test keeps interior samples in segment
//! interiors (away from vertices and exact segment boundaries), asserts the
//! endpoints exactly and interior points within a small point-distance and
//! per-component `abs < 1e-4` or `rel < 1e-3` tolerance.
//!
//! # Portability
//!
//! The kernel uses only subtraction, `dot`, `sqrt`, multiply, compare and one
//! divide in the portable core-`WGSL` subset — no atomics, no dynamic local
//! array, no optional device feature — so the twin runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! Provenance: standard arc-length polyline resampling plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

/// Strand count and per-strand target point count for the one-thread-per-strand
/// dispatch, padded to `16` bytes.
struct Params {
    strand_count: u32,
    target_points: u32,
    pad0: u32,
    pad1: u32,
}

/// One raw strand's slice inside the shared raw-point buffer: `start..start+len`
/// (host pre-filtered so the slice is always in bounds and `len >= 1`), padded
/// to `16` bytes.
struct Range {
    start: u32,
    len: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ranges: array<Range>;
@group(0) @binding(2) var<storage, read> raw_points: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out_points: array<vec4<f32>>;

// Segment length between raw points `base + i` and `base + i + 1`. Recomputed on
// demand (WGSL has no dynamic local array); `sqrt(dot(d, d))` mirrors the
// reference `Vec3::length` (`sqrt` of the fixed-order dot) so both sides read the
// same float.
fn seg_len(base: u32, i: u32) -> f32 {
    let d = raw_points[base + i + 1u].xyz - raw_points[base + i].xyz;
    return sqrt(dot(d, d));
}

// Fixed-order linear interpolation matching the reference `lerp`
// (`a + (b - a) * t`).
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if s >= params.strand_count {
        return;
    }

    var target_pts = params.target_points;
    if target_pts < 2u {
        target_pts = 2u;
    }

    let range = ranges[s];
    let base = range.start;
    let n = range.len;
    let out_base = s * target_pts;
    let root = raw_points[base].xyz;

    // Single raw point: emit `target_pts` copies of the root (zero-length strand).
    if n == 1u {
        for (var j = 0u; j < target_pts; j = j + 1u) {
            out_points[out_base + j] = vec4<f32>(root, 0.0);
        }
        return;
    }

    let last_raw = n - 1u;

    // Total arc length, accumulated root→tip in the same order as the
    // reference's `cumulative[]` build.
    var total = 0.0;
    for (var i = 0u; i < last_raw; i = i + 1u) {
        total = total + seg_len(base, i);
    }

    // All points coincide: emit `target_pts` copies of the root.
    if total <= 0.0 {
        for (var j = 0u; j < target_pts; j = j + 1u) {
            out_points[out_base + j] = vec4<f32>(root, 0.0);
        }
        return;
    }

    // Persistent segment cursor `seg` with running cumulative distance `cum0`
    // (== reference `cumulative[seg]`). `seg` only advances across `j`, so the
    // partial sums land bit-identically on the reference array.
    var seg = 0u;
    var cum0 = 0.0;
    let denom = f32(target_pts - 1u);
    for (var j = 0u; j < target_pts; j = j + 1u) {
        var d = total;
        if j + 1u != target_pts {
            d = total * f32(j) / denom;
        }
        // Advance until `d` lies in [cum0, cum0 + seg_len(seg)]. The cursor is
        // capped at the last real segment (`last_raw - 1`): the reference never
        // advances past it (its `cumulative[last_raw] == total >= d` always),
        // and the cap keeps `seg + 1` — the `seg_len`/`lerp` index — in bounds
        // even when the independently summed `total` differs from the running
        // `cum0` by a `ULP`.
        loop {
            if seg + 1u >= last_raw {
                break;
            }
            let cum1 = cum0 + seg_len(base, seg);
            if cum1 < d {
                cum0 = cum1;
                seg = seg + 1u;
            } else {
                break;
            }
        }
        // `d0`/`d1` mirror the reference `cumulative[seg]`/`cumulative[seg+1]`.
        let d0 = cum0;
        let d1 = cum0 + seg_len(base, seg);
        let span = d1 - d0;
        var t = 0.0;
        if j + 1u == target_pts {
            // Tip pinned exactly: the reference's `t = (total - d0) / span`
            // reduces to `span / span == 1.0`, so emit the true tip vertex.
            t = 1.0;
        } else if span > 0.0 {
            t = (d - d0) / span;
        }
        let p = lerp3(raw_points[base + seg].xyz, raw_points[base + seg + 1u].xyz, t);
        out_points[out_base + j] = vec4<f32>(p, 0.0);
    }
}
