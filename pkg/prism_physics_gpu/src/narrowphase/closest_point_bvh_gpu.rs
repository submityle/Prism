//! Real-device `wgpu` compute implementation of the closest-point distance
//! scene query: the device twin of
//! [`closest_point_bvh`](super::closest_point_bvh::closest_point_bvh).
//!
//! [`GpuSceneClosestPoint`] compiles `shaders/narrowphase_closest_point.wgsl`
//! once and exposes [`GpuSceneClosestPoint::closest`], which finds, for one
//! world-space query point, the single nearest target and the exact closest
//! point on its (optionally rounded) surface. One device invocation handles one
//! target: it runs a single Gilbert-Johnson-Keerthi distance walk between the
//! query point (modelled as a one-vertex hull) and the target core, matching the
//! `CPU` [`evaluate`](super::closest_point_bvh) step operation for operation.
//! The host then folds in each target's rounding radius through the shared
//! [`closest_hit_from_core`](super::closest_point_bvh::closest_hit_from_core)
//! rule and reduces to the nearest target with the shared
//! [`better`](super::closest_point_bvh::better) order, so the result is the
//! identical hit the `CPU` brute and `BVH` queries return.
//!
//! # Honest brute reduction
//!
//! A nearest query has no device-side priority queue, so this kernel evaluates
//! every target (one lane each) and the host keeps the least-distance hit. That
//! is the `GPU` analogue of the `CPU` brute golden, not the `BVH` branch and
//! bound; the `BVH` prune lives on the `CPU` query, and the device result is
//! pinned to it exactly by the parity suite. There is deliberately no resident
//! variant: the honest device form is a full brute sweep per query.
//!
//! # Body layout
//!
//! Body `0` is the query point's one-vertex hull posed at the query position;
//! bodies `1..=n` are the targets in slice order. Each lane `i` solves the pair
//! `(0, i + 1)` and reports target `i`, matching the `0`-based target index the
//! `CPU` query returns.
//!
//! # Buffer layout
//!
//! The `CPU`-side packing here is byte-for-byte with the `WGSL` structs:
//!
//! * [`Params`] carries the target count in one uniform;
//! * hull headers upload as one [`GpuHullHeader`] each: `(vert_offset,
//!   vert_count, 0, 0)` as a `vec4<u32>`, locating the body's vertex slice in
//!   the flattened vertex array (the radius is folded in on the host, so no
//!   extra binding is needed);
//! * every hull's local-space vertices concatenate into one `vec4<f32>` array
//!   (`xyz` used, `w` padding), each body's slice beginning at its `vert_offset`;
//! * poses upload as one [`GpuPose`] each: the world translation in
//!   `translation.xyz` (`w` unused) and the rotation quaternion `(x, y, z, w)`
//!   in `rotation`;
//! * pairs upload as one `vec2<u32>` each (query body index `0`, target body
//!   index);
//! * results read back as two `vec4<f32>` each ([`GpuClosestOut`]):
//!   `(point_b.xyz, core_distance)` then `(normal.xyz, intersecting_flag)`,
//!   where an `intersecting_flag` of `1.0` marks a query point inside the target
//!   core.
//!
//! # Provenance
//!
//! Closest-point distance via a Gilbert-Johnson-Keerthi distance walk (Gilbert,
//! Johnson, and Keerthi, 1988) with Ericson's Voronoi sub-distance (2005). No
//! Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::closest_point_bvh::{better, closest_hit_from_core, ClosestPointHit, SceneClosestPoint};
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::layout::{buffer_entry, entry};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuClosestOut`] slot occupies, mirroring the `WGSL`
/// `ClosestOut` stride (two `vec4<f32>`).
const OUT_BYTES: u64 = size_of::<GpuClosestOut>() as u64;

/// `intersecting_flag` value the kernel writes when the query point lies within
/// the target core.
const INTERSECTING_FLAG: f32 = 1.0;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_closest_point.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of targets queued in the pair buffer (one lane each).
    num_pairs: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
    /// Padding to a 16-byte boundary.
    pad2: u32,
}

/// Upload form of one hull header, matching the `WGSL` `HullHeader` struct:
/// `(vert_offset, vert_count, 0, 0)` as a `vec4<u32>`. The closest-point kernel
/// reads only the support hull, so the trailing slots stay zero and the rounding
/// radius is applied on the host.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuHullHeader {
    /// `(vert_offset, vert_count, 0, 0)`.
    data: [u32; 4],
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

/// Read-back form of one closest-point result, matching the `WGSL` `ClosestOut`
/// struct: `(point_b.xyz, core_distance)` then `(normal.xyz, intersecting_flag)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuClosestOut {
    /// Core closest point on the target in `xyz`, core distance in `w`.
    point_dist: [f32; 4],
    /// Outward unit normal in `xyz`, intersecting flag in `w` (`1.0` inside the
    /// target core, `0.0` separated).
    normal_flag: [f32; 4],
}

/// A compiled, reusable `GPU` closest-point distance query pipeline. Build it
/// once per device and reuse it across queries; the pipeline compiles on
/// construction.
pub struct GpuSceneClosestPoint {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, headers, vertices, poses, pairs, and
    /// results.
    layout: BindGroupLayout,
    /// The closest-point kernel: one invocation per target.
    pipeline: ComputePipeline,
}

impl GpuSceneClosestPoint {
    /// Compiles the closest-point distance kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSceneClosestPoint {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_closest_point"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_closest_point.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_closest_point_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_closest_point_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_closest_point_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_closest_point"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSceneClosestPoint {
            module,
            layout,
            pipeline,
        }
    }

    /// `GPU`-driven form of
    /// [`closest_point_bvh`](super::closest_point_bvh::closest_point_bvh): the
    /// nearest target to `query`, with the closest point on its rounded surface,
    /// the surface distance, the outward normal there, and whether the query
    /// point is inside. Evaluates every target on the device (one lane each) and
    /// reduces on the host, matching the `CPU` brute and `BVH` queries exactly.
    ///
    /// Returns `None` only when there are no targets.
    ///
    /// # Panics
    ///
    /// Panics if `target_hulls`, `target_poses`, and `target_radii` do not all
    /// share one length, since each target's hull, pose, and radius are indexed
    /// in lockstep.
    #[must_use]
    pub fn closest(
        &self,
        ctx: &GpuContext,
        target_hulls: &[ConvexHull],
        target_poses: &[ConvexPose],
        target_radii: &[f32],
        query: &SceneClosestPoint,
    ) -> Option<ClosestPointHit> {
        assert_eq!(
            target_hulls.len(),
            target_poses.len(),
            "target hull and pose slices must align one target per index"
        );
        assert_eq!(
            target_hulls.len(),
            target_radii.len(),
            "target hull and radius slices must align one target per index"
        );
        let n = target_hulls.len();
        if n == 0 {
            return None;
        }

        let device = ctx.device();

        let params = Params {
            num_pairs: u32::try_from(n).unwrap_or(u32::MAX),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = buffer::uniform(device, "prism_narrowphase_closest_point_params", &params);

        // Body 0 is the query point's one-vertex hull; bodies 1..=n are the
        // targets. Flatten every hull into shared headers and vertices.
        let point_hull = ConvexHull::from_point();
        let mut headers: Vec<GpuHullHeader> = Vec::with_capacity(n + 1);
        let mut packed_vertices: Vec<[f32; 4]> = Vec::new();
        let push_hull = |hull: &ConvexHull, headers: &mut Vec<GpuHullHeader>, verts: &mut Vec<[f32; 4]>| {
            let vert_offset = u32::try_from(verts.len()).unwrap_or(u32::MAX);
            for v in hull.vertices() {
                verts.push([v.x, v.y, v.z, 0.0]);
            }
            let vert_count = u32::try_from(hull.vertices().len()).unwrap_or(u32::MAX);
            headers.push(GpuHullHeader {
                data: [vert_offset, vert_count, 0, 0],
            });
        };
        push_hull(&point_hull, &mut headers, &mut packed_vertices);
        for hull in target_hulls {
            push_hull(hull, &mut headers, &mut packed_vertices);
        }

        let headers_buf =
            buffer::storage_read(device, "prism_narrowphase_closest_point_hulls", &headers);
        let vertices_buf = buffer::storage_read(
            device,
            "prism_narrowphase_closest_point_vertices",
            &packed_vertices,
        );

        // Poses: body 0 at the query position (identity rotation), then the
        // targets in slice order.
        let query_pose = ConvexPose::new(query.position, Quat::IDENTITY);
        let mut packed_poses: Vec<GpuPose> = Vec::with_capacity(n + 1);
        let push_pose = |p: &ConvexPose, poses: &mut Vec<GpuPose>| {
            poses.push(GpuPose {
                translation: [p.translation.x, p.translation.y, p.translation.z, 0.0],
                rotation: [p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w],
            });
        };
        push_pose(&query_pose, &mut packed_poses);
        for pose in target_poses {
            push_pose(pose, &mut packed_poses);
        }
        let poses_buf =
            buffer::storage_read(device, "prism_narrowphase_closest_point_poses", &packed_poses);

        // Pairs: lane i solves (query body 0, target body i + 1).
        let packed_pairs: Vec<[u32; 2]> = (0..n)
            .map(|i| [0_u32, u32::try_from(i + 1).unwrap_or(u32::MAX)])
            .collect();
        let pairs_buf =
            buffer::storage_read(device, "prism_narrowphase_closest_point_pairs", &packed_pairs);

        let out_bytes = OUT_BYTES * n as u64;
        let out_buf =
            buffer::storage_rw_zeroed(device, "prism_narrowphase_closest_point_out", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_closest_point_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &headers_buf),
                entry(2, &vertices_buf),
                entry(3, &poses_buf),
                entry(4, &pairs_buf),
                entry(5, &out_buf),
            ],
        });

        let out_stage =
            buffer::staging(device, "prism_narrowphase_closest_point_stage", out_bytes);

        let groups = u32::try_from(n.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_closest_point_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_closest_point_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &out_buf, &out_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<GpuClosestOut>(ctx, &out_stage);

        // Fold the rounding radius into each core result through the shared host
        // rule, then reduce to the nearest target with the shared order.
        let mut best: Option<ClosestPointHit> = None;
        for (i, slot) in raw.iter().enumerate().take(n) {
            let intersecting = slot.normal_flag[3] == INTERSECTING_FLAG;
            let core_distance = slot.point_dist[3];
            let point_b = Vec3::new(slot.point_dist[0], slot.point_dist[1], slot.point_dist[2]);
            let normal = Vec3::new(slot.normal_flag[0], slot.normal_flag[1], slot.normal_flag[2]);
            let hit = closest_hit_from_core(
                u32::try_from(i).unwrap_or(u32::MAX),
                target_radii[i],
                intersecting,
                core_distance,
                point_b,
                normal,
                query.position,
            );
            best = Some(match best {
                Some(b) if !better(&hit, &b) => b,
                _ => hit,
            });
        }
        best
    }
}
