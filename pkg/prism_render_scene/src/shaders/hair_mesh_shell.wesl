//! GPU strand-to-shell meshing kernel: the coarsest `Mesh` LOD proxy stage.
//!
//! Compute twin of the CPU golden reference
//! `prism_render_architecture::hair::mesh_shell::build_from_sections` (the
//! shared core behind `build_shell` / `build_shell_tapered`). A `GPU` is now
//! available, so this shader is compile- and type-checked through the same
//! render-world `ShaderCache` / `wesl` pipeline the runtime uses (see
//! `shader_tests`), with on-device dispatch/perf calibration still awaiting
//! real hardware; it is the GPU-driven *shell meshing* stage (design §10.4)
//! that collapses a too-distant render strand onto a low triangle-count solid
//! tube, mirroring `UE5` Groom mesh LODs and comparable `AMD` `TressFX`
//! /`NVIDIA` `HairWorks` proxy meshes at the algorithm level without reusing
//! their code.
//!
//! One invocation owns one strand. It reads that strand's flat control-point
//! range plus the coherent rotation-minimizing per-vertex frames produced by
//! the [`hair frames`] stage (the bitangent spans the section width, the normal
//! spans the section thickness, the tangent rides along for anisotropic
//! shading) and a per-point half-width / half-thickness, then emits exactly the
//! same indexed triangle-list shell as the CPU golden, in the identical
//! vertex/normal/UV/index order so the GPU path stays a faithful twin:
//!
//! - four rectangle-corner vertices per control point ("ring"): ring `r` owns
//!   vertices `4*r .. 4*r + 4`, one per `(width_sign, thickness_sign)` walked
//!   once around the section so consecutive corners share an edge,
//! - per-corner diagonal outward normal (`sign_w*bitangent + sign_h*normal`
//!   normalized, falling back to the frame normal on a degenerate frame, just
//!   like the CPU `normalize_or`),
//! - `u` walks the perimeter in quarter steps (`0`, `0.25`, `0.5`, `0.75`),
//!   `v` = normalized arc length from `0` at the root to `1` at the tip (even
//!   parameter spacing on a degenerate zero-length strand, exactly like the CPU
//!   fallback),
//! - eight side triangles per segment (two per quad face) plus two triangles
//!   for each of the flat root/tip caps, with the same outward winding.
//!
//! The shell is *view-independent*: the section is oriented by the strand's own
//! rotation-minimizing frame rather than toward the camera, so the mesh is
//! stable across frames. Output is batched: a strand's vertices start at
//! `vertex_offset` and its indices at `index_offset`, and emitted indices are
//! biased by `vertex_offset` so every strand's triangle list addresses its own
//! slice of the shared output pool. When a strand writes to `vertex_offset = 0`
//! the index arithmetic is bit-identical to the single-strand CPU golden, so
//! that golden validates this kernel.
//!
//! Buffer layout: `points` is the flat pool of all strand control points
//! (`xyz` used); `frame_tangents`, `frame_normals`, and `frame_bitangents` run
//! parallel to it; `half_widths` / `half_thicknesses` carry the per-point
//! (already tapered) section extents. `strands[s]` slices those pools for
//! strand `s` and names its output offsets. Everything is bounds-guarded: a
//! strand with fewer than two points (or a point count past the scratch bound)
//! emits nothing rather than indexing out of range, matching the CPU
//! `ShellMesh::default()` fallback.

/// Upper bound on control points per strand, bounding the fixed-count loops so
/// the kernel needs no dynamic allocation. Matches the ribbon / interpolation
/// twins.
const MAX_HAIR_STRAND_POINTS: u32 = 256u;

/// Squared-length threshold below which the corner outward direction is treated
/// as degenerate and the frame normal is used instead. Mirrors the CPU
/// `Vec3::normalize_or` guard (`len_sq <= f32::EPSILON`).
const NORMALIZE_EPSILON: f32 = 1.1920929e-7;

/// Per-strand slice descriptor into the flat input pools and output buffers.
struct HairShellStrand {
    // First control point of this strand in `points` / `frame_*` / sections.
    point_offset: u32,
    // Number of control points; a value below two emits an empty shell.
    point_count: u32,
    // First output vertex slot (`out_positions` / `out_normals` / `out_uvs`).
    vertex_offset: u32,
    // First output index slot in `out_indices`.
    index_offset: u32,
}

/// Tunable parameters shared by every strand of a groom. Only the strand count
/// that bounds the dispatch is needed; the geometry is fully data-driven.
struct HairShellParams {
    // Number of strands, bounding the dispatch.
    strand_count: u32,
}

// Flat pool of all strand control points (`xyz` used).
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
// Per-vertex rotation-minimizing frame tangent (`xyz`), parallel to `points`.
@group(0) @binding(1) var<storage, read> frame_tangents: array<vec4<f32>>;
// Per-vertex frame normal (`xyz`), the section thickness direction.
@group(0) @binding(2) var<storage, read> frame_normals: array<vec4<f32>>;
// Per-vertex frame bitangent (`xyz`), the section width direction.
@group(0) @binding(3) var<storage, read> frame_bitangents: array<vec4<f32>>;
// Per-vertex section half-width along the bitangent (already tapered).
@group(0) @binding(4) var<storage, read> half_widths: array<f32>;
// Per-vertex section half-thickness along the normal (already tapered).
@group(0) @binding(5) var<storage, read> half_thicknesses: array<f32>;
// Per-strand slice descriptors.
@group(0) @binding(6) var<storage, read> strands: array<HairShellStrand>;
// Output shell positions (`xyz` = position, `w` = 1.0); four per point.
@group(0) @binding(7) var<storage, read_write> out_positions: array<vec4<f32>>;
// Output per-vertex outward normals (`xyz`, `w` = 0.0).
@group(0) @binding(8) var<storage, read_write> out_normals: array<vec4<f32>>;
// Output per-vertex UVs (`u` walks the perimeter, `v` = arc length).
@group(0) @binding(9) var<storage, read_write> out_uvs: array<vec2<f32>>;
// Output triangle-list indices, biased by each strand's `vertex_offset`.
@group(0) @binding(10) var<storage, read_write> out_indices: array<u32>;

var<immediate> params: HairShellParams;

@compute @workgroup_size(64)
fn hair_mesh_shell(@builtin(global_invocation_id) gid: vec3<u32>) {
    let strand = gid.x;
    if strand >= params.strand_count {
        return;
    }

    let desc = strands[strand];
    let n = desc.point_count;
    // Fewer than two points cannot sweep a tube: empty shell, exactly like the
    // CPU `build_from_sections` length guard.
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

    // Pass 2: emit four corner vertices per ring.
    let inv_span = f32(n - 1u);
    var cumulative = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        if i > 0u {
            let d = points[pbase + i].xyz - points[pbase + i - 1u].xyz;
            cumulative = cumulative + length(d);
        }

        let p = points[pbase + i].xyz;
        let normal = frame_normals[pbase + i].xyz;
        let bitangent = frame_bitangents[pbase + i].xyz;
        let half_width = half_widths[pbase + i];
        let half_thickness = half_thicknesses[pbase + i];

        var v = 0.0;
        if total > 0.0 {
            v = cumulative / total;
        } else {
            // Zero-length strand: even parameter spacing (CPU fallback).
            v = f32(i) / inv_span;
        }

        // Corner `j` walks the rectangle so consecutive corners share an edge,
        // matching the CPU `CORNER_SIGNS = [(1,1),(-1,1),(-1,-1),(1,-1)]`. The
        // signs are derived arithmetically (no array indexing) to stay a plain
        // const-expression twin: width is positive on corners 0/3, thickness on
        // corners 0/1.
        for (var j = 0u; j < 4u; j = j + 1u) {
            let sign_w = select(-1.0, 1.0, j == 0u || j == 3u);
            let sign_h = select(-1.0, 1.0, j == 0u || j == 1u);
            let offset = bitangent * (sign_w * half_width)
                + normal * (sign_h * half_thickness);
            let slot = vbase + 4u * i + j;
            out_positions[slot] = vec4<f32>(p + offset, 1.0);

            // Diagonal outward normal at the rectangle corner; `normalize_or`
            // falls back to the frame normal on a degenerate frame.
            let outward = bitangent * sign_w + normal * sign_h;
            let len_sq = dot(outward, outward);
            var unit_outward = normal;
            if len_sq > NORMALIZE_EPSILON {
                unit_outward = outward * (1.0 / sqrt(len_sq));
            }
            out_normals[slot] = vec4<f32>(unit_outward, 0.0);
            out_uvs[slot] = vec2<f32>(f32(j) * 0.25, v);
        }
    }

    // Eight side triangles per segment, biased into this strand's output slice.
    for (var r = 0u; r < n - 1u; r = r + 1u) {
        let base = vbase + 4u * r;
        let next = vbase + 4u * (r + 1u);
        let o = ibase + 24u * r;
        for (var s = 0u; s < 4u; s = s + 1u) {
            let sn = (s + 1u) % 4u;
            let a = base + s;
            let b = base + sn;
            let c = next + sn;
            let d = next + s;
            let q = o + 6u * s;
            // Quad (a, b, c, d) wound so its face normal points outward.
            out_indices[q + 0u] = a;
            out_indices[q + 1u] = b;
            out_indices[q + 2u] = c;
            out_indices[q + 3u] = a;
            out_indices[q + 4u] = c;
            out_indices[q + 5u] = d;
        }
    }

    // Caps follow the side triangles: 24 indices per segment for `n - 1`
    // segments, then two triangles for each of the root and tip caps.
    let cap_base = ibase + 24u * (n - 1u);
    // Root cap (ring 0): wound so its face normal points along -tangent.
    out_indices[cap_base + 0u] = vbase + 0u;
    out_indices[cap_base + 1u] = vbase + 2u;
    out_indices[cap_base + 2u] = vbase + 1u;
    out_indices[cap_base + 3u] = vbase + 0u;
    out_indices[cap_base + 4u] = vbase + 3u;
    out_indices[cap_base + 5u] = vbase + 2u;
    // Tip cap (last ring): reversed winding, face normal along +tangent.
    let tip = vbase + 4u * (n - 1u);
    out_indices[cap_base + 6u] = tip;
    out_indices[cap_base + 7u] = tip + 1u;
    out_indices[cap_base + 8u] = tip + 2u;
    out_indices[cap_base + 9u] = tip;
    out_indices[cap_base + 10u] = tip + 2u;
    out_indices[cap_base + 11u] = tip + 3u;
}
