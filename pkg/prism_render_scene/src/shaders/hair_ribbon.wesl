//! GPU strand-to-ribbon meshing kernel: the `Cards` LOD proxy geometry stage.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::ribbon::build_ribbon`. The sandbox has no
//! GPU, so this shader is validated by naga compilation only (see
//! `shader_tests`); it is the GPU-driven *card meshing* stage (design §10.4)
//! that swaps a too-thin / too-distant render strand for a flat quad strip
//! following its centerline, mirroring `UE5` Groom card LODs and `AMD` `TressFX`
//! /`NVIDIA` `HairWorks` proxy strips at the algorithm level without reusing
//! their code.
//!
//! One invocation owns one strand. It reads that strand's flat control-point
//! range plus the coherent rotation-minimizing per-vertex frames produced by
//! the [`hair frames`] stage (the bitangent is the widening direction; the
//! tangent rides along for anisotropic shading) and a per-point radius, then
//! emits exactly the same indexed triangle-list ribbon as the CPU golden, in
//! the identical vertex/UV/index order so the GPU path stays a faithful twin:
//!
//! - two edge vertices per control point (`2*i` = left = `p - bitangent*r`,
//!   `2*i + 1` = right = `p + bitangent*r`),
//! - `u` = `0` on the left edge, `1` on the right; `v` = normalized arc length
//!   from `0` at the root to `1` at the tip (even parameter spacing on a
//!   degenerate zero-length strand, exactly like the CPU fallback),
//! - two triangles per segment with the consistent
//!   `(l0, r0, l1) / (r0, r1, l1)` winding.
//!
//! The ribbon is *view-independent*: it widens along the strand's own
//! rotation-minimizing bitangent rather than toward the camera, so the mesh is
//! stable across frames. Output is batched: a strand's vertices start at
//! `vertex_offset` and its indices at `index_offset`, and emitted indices are
//! biased by `vertex_offset` so every strand's triangle list addresses its own
//! slice of the shared output pool. When a strand writes to `vertex_offset = 0`
//! the index arithmetic is bit-identical to the single-strand CPU golden, so
//! that golden validates this kernel.
//!
//! Buffer layout: `points` is the flat pool of all strand control points
//! (`xyz` used); `frame_tangents` and `frame_bitangents` run parallel to it;
//! `radii` carries the per-point (already tapered) half-width. `strands[s]`
//! slices those pools for strand `s` and names its output offsets. Everything
//! is bounds-guarded: a strand with fewer than two points (or a point count
//! past the scratch bound) emits nothing rather than indexing out of range,
//! matching the CPU `RibbonMesh::default()` fallback.

/// Upper bound on control points per strand, bounding the fixed-count loops so
/// the kernel needs no dynamic allocation. Matches the interpolation twin.
const MAX_HAIR_STRAND_POINTS: u32 = 256u;

/// Per-strand slice descriptor into the flat input pools and output buffers.
struct HairRibbonStrand {
    // First control point of this strand in `points` / `frame_*` / `radii`.
    point_offset: u32,
    // Number of control points; a value below two emits an empty ribbon.
    point_count: u32,
    // First output vertex slot (`out_positions` / `out_tangents` / `out_uvs`).
    vertex_offset: u32,
    // First output index slot in `out_indices`.
    index_offset: u32,
}

/// Tunable parameters shared by every strand of a groom. Only the strand count
/// that bounds the dispatch is needed; the geometry is fully data-driven.
struct HairRibbonParams {
    // Number of strands, bounding the dispatch.
    strand_count: u32,
}

// Flat pool of all strand control points (`xyz` used).
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
// Per-vertex rotation-minimizing frame tangent (`xyz`), parallel to `points`.
@group(0) @binding(1) var<storage, read> frame_tangents: array<vec4<f32>>;
// Per-vertex frame bitangent (`xyz`), the ribbon widening direction.
@group(0) @binding(2) var<storage, read> frame_bitangents: array<vec4<f32>>;
// Per-vertex ribbon half-width (already tapered by the caller).
@group(0) @binding(3) var<storage, read> radii: array<f32>;
// Per-strand slice descriptors.
@group(0) @binding(4) var<storage, read> strands: array<HairRibbonStrand>;
// Output ribbon edge positions (`xyz` = position, `w` = 1.0); two per point.
@group(0) @binding(5) var<storage, read_write> out_positions: array<vec4<f32>>;
// Output per-vertex tangents (`xyz`, `w` = 0.0) for anisotropic shading.
@group(0) @binding(6) var<storage, read_write> out_tangents: array<vec4<f32>>;
// Output per-vertex UVs (`u` in {0,1} for left/right, `v` = arc length).
@group(0) @binding(7) var<storage, read_write> out_uvs: array<vec2<f32>>;
// Output triangle-list indices, biased by each strand's `vertex_offset`.
@group(0) @binding(8) var<storage, read_write> out_indices: array<u32>;

var<immediate> params: HairRibbonParams;

@compute @workgroup_size(64)
fn hair_ribbon(@builtin(global_invocation_id) gid: vec3<u32>) {
    let strand = gid.x;
    if strand >= params.strand_count {
        return;
    }

    let desc = strands[strand];
    let n = desc.point_count;
    // A single point (or fewer) cannot form a quad strip: empty ribbon, exactly
    // like the CPU `build_ribbon` length guard.
    if n < 2u || n > MAX_HAIR_STRAND_POINTS {
        return;
    }

    let pbase = desc.point_offset;
    let vbase = desc.vertex_offset;
    let ibase = desc.index_offset;

    // Pass 1: total centerline arc length (drives the `v` coordinate).
    var total = 0.0;
    for (var i = 1u; i < n; i = i + 1u) {
        let d = points[pbase + i].xyz - points[pbase + i - 1u].xyz;
        total = total + length(d);
    }

    // Pass 2: emit two edge vertices per control point plus per-segment tris.
    let inv_span = f32(n - 1u);
    var cumulative = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        if i > 0u {
            let d = points[pbase + i].xyz - points[pbase + i - 1u].xyz;
            cumulative = cumulative + length(d);
        }

        let p = points[pbase + i].xyz;
        let tangent = frame_tangents[pbase + i].xyz;
        let offset = frame_bitangents[pbase + i].xyz * radii[pbase + i];

        let left = vbase + 2u * i;
        let right = left + 1u;
        out_positions[left] = vec4<f32>(p - offset, 1.0);
        out_positions[right] = vec4<f32>(p + offset, 1.0);
        out_tangents[left] = vec4<f32>(tangent, 0.0);
        out_tangents[right] = vec4<f32>(tangent, 0.0);

        var v = 0.0;
        if total > 0.0 {
            v = cumulative / total;
        } else {
            // Zero-length strand: even parameter spacing (CPU fallback).
            v = f32(i) / inv_span;
        }
        out_uvs[left] = vec2<f32>(0.0, v);
        out_uvs[right] = vec2<f32>(1.0, v);
    }

    // Two triangles per segment, consistent winding, biased into this strand's
    // output slice so a batched pool stays correctly indexed.
    for (var i = 0u; i < n - 1u; i = i + 1u) {
        let l0 = vbase + 2u * i;
        let r0 = l0 + 1u;
        let l1 = l0 + 2u;
        let r1 = l0 + 3u;
        let o = ibase + 6u * i;
        out_indices[o + 0u] = l0;
        out_indices[o + 1u] = r0;
        out_indices[o + 2u] = l1;
        out_indices[o + 3u] = r0;
        out_indices[o + 4u] = r1;
        out_indices[o + 5u] = l1;
    }
}
