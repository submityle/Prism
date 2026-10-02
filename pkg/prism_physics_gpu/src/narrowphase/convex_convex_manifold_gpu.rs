//! Real-device `wgpu` compute implementation of the convex-versus-convex
//! multi-point narrow phase.
//!
//! [`GpuConvexConvexManifoldNarrowphase`] compiles
//! `shaders/narrowphase_convex_convex_manifold.wgsl` once and exposes
//! [`GpuConvexConvexManifoldNarrowphase::query`], which turns a batch of
//! `(hull, hull)` couples into one contact manifold each on the device. One
//! invocation handles one couple: it runs the full narrow-phase pipeline of the
//! [`cpu_convex_convex_manifold`](super::convex_convex_manifold::cpu_convex_convex_manifold)
//! twin on the device, walking a Gilbert-Johnson-Keerthi simplex of the
//! Minkowski difference toward the origin, blowing the terminal simplex up to a
//! tetrahedron and expanding it with the expanding-polytope algorithm to
//! recover the minimum-translation normal and penetration depth, then promoting
//! that penetration to a manifold by clipping the incident hull's most
//! anti-parallel face against the reference face and reducing the survivors to
//! the widest, deepest four. A passing real-device parity test is therefore
//! direct evidence the kernel builds the same manifolds as the reference.
//!
//! # Buffer layout
//!
//! The `CPU`-side packing here is byte-for-byte with the `WGSL` structs:
//!
//! * hull headers upload as one [`GpuHullHeader`] each: `(vert_offset,
//!   vert_count, face_offset, face_count)` as a `vec4<u32>`, locating the body's
//!   slices in the flattened vertex and face arrays;
//! * every hull's local-space vertices concatenate into one `vec4<f32>` array
//!   (`xyz` used, `w` padding), each body's slice beginning at its `vert_offset`;
//! * every hull's faces concatenate into one [`GpuFace`] array: the outward
//!   unit normal in `plane.xyz` with the plane offset in `plane.w`, then
//!   `(loop_offset, loop_count, pad, pad)` in `loop_info`;
//! * every face's counter-clockwise loop of hull-local vertex indices
//!   concatenates into one `u32` array, each loop beginning at its `loop_offset`;
//! * poses upload as one [`GpuPose`] each: the world translation in
//!   `translation.xyz` (`w` unused) and the rotation quaternion `(x, y, z, w)`
//!   in `rotation`;
//! * couples upload as one `vec2<u32>` each (body index, body index);
//! * manifolds read back as five `vec4<f32>` each ([`GpuManifold`]):
//!   `(normal.xyz, count)` then four `(position.xyz, depth)` points.
//!
//! # Dense output
//!
//! The kernel writes a slot per input couple, marking separated couples with a
//! zero point count rather than compacting them away. This keeps the manifold
//! index aligned with the couple index, which the parity test relies on, and
//! lets a downstream [`crate::scan`] compaction stream the survivors without a
//! second pass over the couples.
//!
//! Provenance: Gilbert-Johnson-Keerthi distance (1988) with Ericson's Voronoi
//! sub-distance (2005), the expanding-polytope algorithm (van den Bergen,
//! 2001) with Ericson horizon re-triangulation, and a textbook
//! reference/incident face-clipping manifold with the Sutherland-Hodgman clip
//! and four-point reduction (Ericson, 2004). No Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::convex_convex_manifold::ConvexConvexPair;
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::layout::{buffer_entry, entry};
use super::manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuManifold`] slot occupies, mirroring the `WGSL`
/// `Manifold` stride (five `vec4<f32>`).
const MANIFOLD_BYTES: u64 = size_of::<GpuManifold>() as u64;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_convex_convex_manifold.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of candidate couples queued in the pair buffer.
    num_pairs: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
    /// Padding to a 16-byte boundary.
    pad2: u32,
}

/// Upload form of one hull header, matching the `WGSL` `HullHeader` struct:
/// `(vert_offset, vert_count, face_offset, face_count)` as a `vec4<u32>`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuHullHeader {
    /// `(vert_offset, vert_count, face_offset, face_count)`.
    data: [u32; 4],
}

/// Upload form of one hull face, matching the `WGSL` `GpuFace` struct: the
/// outward unit normal in `plane.xyz` with the plane offset in `plane.w`, then
/// `(loop_offset, loop_count, pad, pad)` in `loop_info`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuFace {
    /// Outward unit normal in `xyz`, plane offset in `w`.
    plane: [f32; 4],
    /// `(loop_offset, loop_count, pad, pad)`.
    loop_info: [u32; 4],
}

/// Upload form of one [`ConvexPose`], matching the `WGSL` `GpuPose` struct: the
/// world translation in `translation.xyz` (`w` unused) and the rotation
/// quaternion `(x, y, z, w)` in `rotation`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPose {
    /// World translation in `xyz`; `w` is unused padding.
    translation: [f32; 4],
    /// Rotation quaternion `(x, y, z, w)`.
    rotation: [f32; 4],
}

/// Read-back form of one contact manifold, matching the `WGSL` `Manifold`
/// struct: `(normal.xyz, count)` then four `(position.xyz, depth)` points.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuManifold {
    /// Shared contact normal in `xyz`, live point count in `w`.
    normal_count: [f32; 4],
    /// First contact point in `xyz`, its penetration depth in `w`.
    p0: [f32; 4],
    /// Second contact point in `xyz`, its penetration depth in `w`.
    p1: [f32; 4],
    /// Third contact point in `xyz`, its penetration depth in `w`.
    p2: [f32; 4],
    /// Fourth contact point in `xyz`, its penetration depth in `w`.
    p3: [f32; 4],
}

/// A compiled, reusable `GPU` convex-versus-convex manifold pipeline.
pub struct GpuConvexConvexManifoldNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, headers, vertices, faces, loops,
    /// poses, pairs, and manifolds.
    layout: BindGroupLayout,
    /// The manifold kernel: one invocation per candidate couple.
    manifolds: ComputePipeline,
}

impl GpuConvexConvexManifoldNarrowphase {
    /// Compiles the convex-versus-convex manifold kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuConvexConvexManifoldNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_convex_convex_manifold"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_convex_convex_manifold.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_convex_convex_manifold_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: true }),
                buffer_entry(7, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_convex_convex_manifold_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let manifolds = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_convex_convex_manifold_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_convex_convex_manifold"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuConvexConvexManifoldNarrowphase {
            module,
            layout,
            manifolds,
        }
    }

    /// Generates convex-versus-convex contact manifolds for `pairs` over
    /// `hulls` and `poses` on the `GPU`.
    ///
    /// Returns one slot per couple, in input order: [`Some`] carrying a one- to
    /// four-point manifold when the hulls penetrate, or [`None`] when they are
    /// separated. The output matches the
    /// [`cpu_convex_convex_manifold`](super::convex_convex_manifold::cpu_convex_convex_manifold)
    /// twin: the point count agrees exactly and the normal, positions, and
    /// depths agree to within the tight float tolerance the support argmax, the
    /// normalise reciprocals, and the barycentric divisions impose.
    ///
    /// An empty `pairs` batch returns an empty vector without touching the
    /// device.
    ///
    /// # Panics
    ///
    /// Panics if `hulls` and `poses` have different lengths, since the kernel
    /// indexes a body's hull and pose in lockstep.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        pairs: &[ConvexConvexPair],
    ) -> Vec<Option<ContactManifold>> {
        assert_eq!(
            hulls.len(),
            poses.len(),
            "hull and pose slices must align one body per index"
        );
        if pairs.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();
        let num_pairs = pairs.len();

        let params = Params {
            num_pairs: u32::try_from(num_pairs).unwrap_or(u32::MAX),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = buffer::uniform(
            device,
            "prism_narrowphase_convex_convex_manifold_params",
            &params,
        );

        // Flatten every hull into shared headers, vertices, faces, and loops.
        // Each header records where this body's vertex and face slices begin in
        // the concatenated arrays and how long they are; a face records where
        // its hull-local loop begins in the concatenated loop array.
        let mut headers: Vec<GpuHullHeader> = Vec::with_capacity(hulls.len());
        let mut packed_vertices: Vec<[f32; 4]> = Vec::new();
        let mut packed_faces: Vec<GpuFace> = Vec::new();
        let mut packed_loops: Vec<u32> = Vec::new();
        for hull in hulls {
            let vert_offset = u32::try_from(packed_vertices.len()).unwrap_or(u32::MAX);
            let face_offset = u32::try_from(packed_faces.len()).unwrap_or(u32::MAX);
            for v in hull.vertices() {
                packed_vertices.push([v.x, v.y, v.z, 0.0]);
            }
            for face in hull.faces() {
                let loop_offset = u32::try_from(packed_loops.len()).unwrap_or(u32::MAX);
                let loop_count = u32::try_from(face.loop_indices.len()).unwrap_or(u32::MAX);
                packed_loops.extend_from_slice(&face.loop_indices);
                packed_faces.push(GpuFace {
                    plane: [face.normal.x, face.normal.y, face.normal.z, face.offset],
                    loop_info: [loop_offset, loop_count, 0, 0],
                });
            }
            let vert_count = u32::try_from(hull.vertices().len()).unwrap_or(u32::MAX);
            let face_count = u32::try_from(hull.faces().len()).unwrap_or(u32::MAX);
            headers.push(GpuHullHeader {
                data: [vert_offset, vert_count, face_offset, face_count],
            });
        }

        // A zero-length storage buffer is invalid; a single padding element is
        // never read because the body that would reference it has no vertices,
        // faces, or loops of its own.
        if headers.is_empty() {
            headers.push(GpuHullHeader { data: [0, 0, 0, 0] });
        }
        if packed_vertices.is_empty() {
            packed_vertices.push([0.0, 0.0, 0.0, 0.0]);
        }
        if packed_faces.is_empty() {
            packed_faces.push(GpuFace {
                plane: [0.0, 0.0, 0.0, 0.0],
                loop_info: [0, 0, 0, 0],
            });
        }
        if packed_loops.is_empty() {
            packed_loops.push(0);
        }

        let headers_buf = buffer::storage_read(
            device,
            "prism_narrowphase_convex_convex_manifold_hulls",
            &headers,
        );
        let vertices_buf = buffer::storage_read(
            device,
            "prism_narrowphase_convex_convex_manifold_vertices",
            &packed_vertices,
        );
        let faces_buf = buffer::storage_read(
            device,
            "prism_narrowphase_convex_convex_manifold_faces",
            &packed_faces,
        );
        let loops_buf = buffer::storage_read(
            device,
            "prism_narrowphase_convex_convex_manifold_face_loops",
            &packed_loops,
        );

        // Poses pack to a GpuPose each: xyz translation then the rotation
        // quaternion (x, y, z, w).
        let packed_poses: Vec<GpuPose> = poses
            .iter()
            .map(|p| GpuPose {
                translation: [p.translation.x, p.translation.y, p.translation.z, 0.0],
                rotation: [p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w],
            })
            .collect();
        let poses_buf = buffer::storage_read(
            device,
            "prism_narrowphase_convex_convex_manifold_poses",
            &packed_poses,
        );

        // Pairs pack to a vec2<u32> each: the two body indices.
        let packed_pairs: Vec<[u32; 2]> = pairs.iter().map(|pair| [pair.a, pair.b]).collect();
        let pairs_buf = buffer::storage_read(
            device,
            "prism_narrowphase_convex_convex_manifold_pairs",
            &packed_pairs,
        );

        let manifolds_bytes = MANIFOLD_BYTES * num_pairs as u64;
        let manifolds_buf = buffer::storage_rw_zeroed(
            device,
            "prism_narrowphase_convex_convex_manifold_out",
            manifolds_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_convex_convex_manifold_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &headers_buf),
                entry(2, &vertices_buf),
                entry(3, &faces_buf),
                entry(4, &loops_buf),
                entry(5, &poses_buf),
                entry(6, &pairs_buf),
                entry(7, &manifolds_buf),
            ],
        });

        let manifolds_stage = buffer::staging(
            device,
            "prism_narrowphase_convex_convex_manifold_stage",
            manifolds_bytes,
        );

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_convex_convex_manifold_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_convex_convex_manifold_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.manifolds);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(
            &mut encoder,
            &manifolds_buf,
            &manifolds_stage,
            manifolds_bytes,
        );
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<GpuManifold>(ctx, &manifolds_stage);
        raw.iter()
            .zip(pairs)
            .map(|(slot, pair)| decode_manifold(slot, pair))
            .collect()
    }
}

/// Rebuilds a [`ContactManifold`] from one raw `GPU` slot, or [`None`] when the
/// slot reports zero live points.
fn decode_manifold(slot: &GpuManifold, pair: &ConvexConvexPair) -> Option<ContactManifold> {
    let count = slot.normal_count[3] as usize;
    if count == 0 {
        return None;
    }
    let normal = Vec3::new(
        slot.normal_count[0],
        slot.normal_count[1],
        slot.normal_count[2],
    );
    let rows = [slot.p0, slot.p1, slot.p2, slot.p3];
    let mut points = [ManifoldPoint::new(Vec3::ZERO, 0.0); MAX_MANIFOLD_POINTS];
    for (point, row) in points.iter_mut().zip(rows.iter()).take(count) {
        *point = ManifoldPoint::new(Vec3::new(row[0], row[1], row[2]), row[3]);
    }
    Some(ContactManifold::new(pair.a, pair.b, normal, count, &points))
}
