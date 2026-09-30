//! `wgpu` compute twin of Prism's forward-scatter curve decode
//! (`prism_render_architecture::hair::dual_scattering::sample_forward_scatter`).
//!
//! `forward_scatter` (and the golden `build_forward_scatter`) *pack* a light
//! ray's coverage-weighted crossing curve into a fixed layer stack — far-boundary
//! depth `layer_depths[i]` and cumulative crossing count `layer_crossings[i]`.
//! The dual-scattering shading pass then *decodes* it: given a receiver at
//! light-space `depth`, how many strands `n` did the light already cross to
//! reach it? That count drives the global multiple-scattering transmittance
//! `a_f^n` and angular spread `n * beta_f^2` (design §5/§11). This kernel is the
//! on-device twin of that decode — one thread per receiver query walks the
//! identical bracket-and-interpolate the golden does, so a passing real-device
//! parity test is direct evidence the ported decode reads the same crossing
//! count as the reference, not merely that its shader compiles.
//!
//! Where `voxel_transmittance` decodes a per-voxel slab as a discrete prefix
//! walk, this decodes a *layered* curve with **linear interpolation** between
//! the two bracketing boundaries — a genuinely distinct sampling kernel. It is
//! the read side of the `forward_scatter` packing: pack once, decode per shaded
//! receiver.
//!
//! # Batched decode shape
//!
//! Several curves are packed back-to-back into shared `layer_depths` /
//! `layer_crossings` pools; each curve owns a compacted `[start, count)` range.
//! A flat query buffer names, per receiver, its `curve` index and `depth`. One
//! thread per query resolves its curve range and writes one crossing count into
//! `out_crossings[q]`, so every query is independent and race-free.
//!
//! # Bit-faithfulness
//!
//! The decode mirrors the golden branch-for-branch: an empty curve
//! (`count == 0`) and a receiver in front of the frontmost boundary
//! (`depth <= layer_depths[start]`) both read `0`; a receiver at or beyond the
//! deepest boundary (`depth >= layer_depths[start + count - 1]`) reads the last
//! (largest) crossing; otherwise the first window `[d0, d1]` with
//! `depth <= d1` interpolates `c0 + (c1 - c0) * t` with `t = (depth - d0) /
//! (d1 - d0)` and `t = 0` when the span is non-positive. The interpolation is a
//! closed-form multiply/add/divide with no transcendental, so `CPU`/`GPU`
//! differ only by a fused-multiply-add `ULP` or two — the parity test asserts a
//! per-value tolerance (`abs_diff < 1e-4` or `rel_diff < 1e-3`) while the
//! front/empty/beyond reads are exact.

/// Decode parameters. `query_count` bounds the dispatch (one invocation per
/// receiver query); `curve_count` guards an out-of-range `curve` index so the
/// kernel is panic-free on malformed input. Scalar-packed and padded to `16`
/// bytes so the uniform block is a multiple of 16.
struct HairScatterSampleParams {
    query_count: u32,
    curve_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One compacted per-curve slice into the shared `layer_depths` /
/// `layer_crossings` pools: entries `start .. start + count`, ascending in
/// light-space depth. A `count == 0` curve means "no occluders" and decodes to
/// `0` everywhere.
struct HairScatterCurveRange {
    start: u32,
    count: u32,
}

/// One receiver query: which packed curve to read (`curve`) and the receiver's
/// light-space depth (`depth`).
struct HairScatterQuery {
    curve: u32,
    depth: f32,
}

@group(0) @binding(0) var<uniform> params: HairScatterSampleParams;

// Per-curve compacted ranges into the shared pools.
@group(0) @binding(1) var<storage, read> curve_ranges: array<HairScatterCurveRange>;

// Shared flat far-boundary depths, non-decreasing within each curve's slice.
@group(0) @binding(2) var<storage, read> layer_depths: array<f32>;

// Shared flat cumulative crossing counts, parallel to `layer_depths`.
@group(0) @binding(3) var<storage, read> layer_crossings: array<f32>;

// Flat receiver queries; length `query_count`.
@group(0) @binding(4) var<storage, read> queries: array<HairScatterQuery>;

// Per-query decoded crossing count; length `query_count`.
@group(0) @binding(5) var<storage, read_write> out_crossings: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if q >= params.query_count {
        return;
    }

    let query = queries[q];
    let curve = query.curve;
    // Guard a malformed curve index: no golden equivalent exists, so treat an
    // out-of-range curve as "no occluders".
    if curve >= params.curve_count {
        out_crossings[q] = 0.0;
        return;
    }

    let range = curve_ranges[curve];
    let start = range.start;
    let count = range.count;

    // Empty curve: no occluders anywhere.
    if count == 0u {
        out_crossings[q] = 0.0;
        return;
    }

    let depth = query.depth;

    // In front of the frontmost boundary: nothing crossed yet.
    let front = layer_depths[start];
    if depth <= front {
        out_crossings[q] = 0.0;
        return;
    }

    // At or beyond the deepest boundary: take the last (largest) count.
    let last = count - 1u;
    let back = layer_depths[start + last];
    if depth >= back {
        out_crossings[q] = layer_crossings[start + last];
        return;
    }

    // Interior: interpolate across the first window `[d0, d1]` with
    // `depth <= d1`, matching the golden's `windows(2)` scan order.
    for (var i = 0u; i < last; i = i + 1u) {
        let d0 = layer_depths[start + i];
        let d1 = layer_depths[start + i + 1u];
        if depth <= d1 {
            let span = d1 - d0;
            var t = 0.0;
            if span > 0.0 {
                t = (depth - d0) / span;
            }
            let c0 = layer_crossings[start + i];
            let c1 = layer_crossings[start + i + 1u];
            out_crossings[q] = c0 + (c1 - c0) * t;
            return;
        }
    }

    // Unreachable given `front < depth < back`, but keep a defined write.
    out_crossings[q] = layer_crossings[start + last];
}
