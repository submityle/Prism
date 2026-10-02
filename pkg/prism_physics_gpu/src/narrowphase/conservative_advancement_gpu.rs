//! Real-device `wgpu` compute implementation of the convex-versus-convex
//! conservative-advancement continuous collision narrow phase.
//!
//! [`GpuConvexConvexToiNarrowphase`] compiles
//! `shaders/narrowphase_convex_convex_toi.wgsl` once and exposes
//! [`GpuConvexConvexToiNarrowphase::query`], which turns a batch of swept
//! `(hull, hull)` couples into one earliest time of impact each on the device.
//! One invocation handles one couple: it runs the full conservative-advancement
//! loop of the
//! [`cpu_convex_convex_toi`](super::conservative_advancement::cpu_convex_convex_toi)
//! twin on the device, repeatedly sampling the two bodies' screw motion at the
//! current time, measuring the gap with a Gilbert-Johnson-Keerthi walk over the
//! Minkowski difference, bounding the fastest rate the gap can close under both
//! bodies' linear and angular velocities, and advancing time by the largest
//! step that provably cannot overshoot the first contact. A passing real-device
//! parity test is therefore direct evidence the kernel finds the same impact
//! times, points, and normals as the reference.
//!
//! # Buffer layout
//!
//! The `CPU`-side packing here is byte-for-byte with the `WGSL` structs:
//!
//! * [`Params`] carries the couple count, the substep length `dt`, and the
//!   `target` separation in one uniform, so every lane reads the same
//!   advance parameters without threading them through the per-body padding;
//! * hull headers upload as one [`GpuHullHeader`] each: `(vert_offset,
//!   vert_count, 0, 0)` as a `vec4<u32>`, locating the body's vertex slice in
//!   the flattened vertex array (the conservative-advancement kernel reads only
//!   the support hull, so it never needs the face or loop tables the manifold
//!   kernel packs);
//! * every hull's local-space vertices concatenate into one `vec4<f32>` array
//!   (`xyz` used, `w` padding), each body's slice beginning at its `vert_offset`;
//! * poses upload as one [`GpuPose`] each: the world translation in
//!   `translation.xyz` (`w` unused) and the rotation quaternion `(x, y, z, w)`
//!   in `rotation`;
//! * motions upload as one [`GpuMotion`] each: the world linear velocity in
//!   `linear.xyz` and the angular velocity (axis scaled by turn rate) in
//!   `angular.xyz`;
//! * couples upload as one `vec2<u32>` each (body index, body index);
//! * times of impact read back as two `vec4<f32>` each ([`GpuToi`]):
//!   `(point.xyz, time)` then `(normal.xyz, hit_flag)`, where a `hit_flag` of
//!   `1.0` marks a reported impact and `0.0` a miss.
//!
//! # Dense output
//!
//! The kernel writes a slot per input couple, marking misses with a zero
//! `hit_flag` rather than compacting them away. This keeps the result index
//! aligned with the couple index, which the parity test relies on, and lets a
//! downstream [`crate::scan`] compaction stream the hits without a second pass
//! over the couples.
//!
//! Provenance: conservative advancement after Brian Mirtich, *Timewarp Rigid
//! Body Simulation* (2000), and the ray-casting formulation of Gino van den
//! Bergen, *Ray Casting against General Convex Objects with Application to
//! Continuous Collision Detection* (2004), over a Gilbert-Johnson-Keerthi
//! distance walk (1988) with Ericson's Voronoi sub-distance (2005). No Unreal
//! Engine source or derived code.

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

use super::body_motion::BodyMotion;
use super::conservative_advancement::{ConvexConvexSweepPair, Toi};
use super::convex_hull::ConvexHull;
use super::convex_pose::ConvexPose;
use super::layout::{buffer_entry, entry};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuToi`] slot occupies, mirroring the `WGSL` `Toi`
/// stride (two `vec4<f32>`).
const TOI_BYTES: u64 = size_of::<GpuToi>() as u64;

/// `hit_flag` value the kernel writes for a reported impact.
const HIT_FLAG: f32 = 1.0;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_convex_convex_toi.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of candidate couples queued in the pair buffer.
    num_pairs: u32,
    /// Substep length; the advance aborts once the sampled time exceeds it.
    dt: f32,
    /// Target separation (speculative margin); `0.0` for touching contact.
    target: f32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
}

/// Upload form of one hull header, matching the `WGSL` `HullHeader` struct:
/// `(vert_offset, vert_count, 0, 0)` as a `vec4<u32>`. The conservative-
/// advancement kernel reads only the support hull, so the face and loop fields
/// the manifold kernel uses stay zero here.
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

/// Upload form of one [`BodyMotion`], matching the `WGSL` `GpuMotion` struct:
/// the world linear velocity in `linear.xyz` and the angular velocity (axis
/// scaled by turn rate) in `angular.xyz`; both `w` lanes are padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMotion {
    /// World linear velocity in `xyz`; `w` is unused padding.
    linear: [f32; 4],
    /// World angular velocity in `xyz`; `w` is unused padding.
    angular: [f32; 4],
}

/// Read-back form of one time of impact, matching the `WGSL` `Toi` struct:
/// `(point.xyz, time)` then `(normal.xyz, hit_flag)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuToi {
    /// World-space contact point in `xyz`, impact time in `w`.
    point_time: [f32; 4],
    /// Contact normal in `xyz`, hit flag in `w` (`1.0` hit, `0.0` miss).
    normal_hit: [f32; 4],
}

/// A compiled, reusable `GPU` convex-versus-convex conservative-advancement
/// pipeline.
pub struct GpuConvexConvexToiNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, headers, vertices, poses, motions,
    /// pairs, and times of impact.
    layout: BindGroupLayout,
    /// The time-of-impact kernel: one invocation per candidate couple.
    tois: ComputePipeline,
}

impl GpuConvexConvexToiNarrowphase {
    /// Compiles the convex-versus-convex conservative-advancement kernel on
    /// `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuConvexConvexToiNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_convex_convex_toi"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_convex_convex_toi.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_convex_convex_toi_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_convex_convex_toi_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let tois = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_convex_convex_toi_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_convex_convex_toi"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuConvexConvexToiNarrowphase {
            module,
            layout,
            tois,
        }
    }

    /// Computes the earliest convex-versus-convex time of impact for `pairs`
    /// over `hulls`, `poses`, and `motions` on the `GPU`.
    ///
    /// Returns one slot per couple, in input order: [`Some`] carrying the
    /// first [`Toi`] within `dt` when the swept hulls reach `target`
    /// separation, or [`None`] when they stay farther apart than `target` for
    /// the whole substep. The output matches the
    /// [`cpu_convex_convex_toi`](super::conservative_advancement::cpu_convex_convex_toi)
    /// twin: the hit decision agrees exactly and the time, point, and normal
    /// agree to within the tight float tolerance the support argmax, the
    /// normalise reciprocals, and the advance division impose.
    ///
    /// An empty `pairs` batch returns an empty vector without touching the
    /// device.
    ///
    /// # Panics
    ///
    /// Panics if `hulls`, `poses`, and `motions` do not all share one length,
    /// since the kernel indexes a body's hull, pose, and motion in lockstep.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        hulls: &[ConvexHull],
        poses: &[ConvexPose],
        motions: &[BodyMotion],
        pairs: &[ConvexConvexSweepPair],
        dt: f32,
        target: f32,
    ) -> Vec<Option<Toi>> {
        assert_eq!(
            hulls.len(),
            poses.len(),
            "hull and pose slices must align one body per index"
        );
        assert_eq!(
            hulls.len(),
            motions.len(),
            "hull and motion slices must align one body per index"
        );
        if pairs.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();
        let num_pairs = pairs.len();

        let params = Params {
            num_pairs: u32::try_from(num_pairs).unwrap_or(u32::MAX),
            dt,
            target,
            pad0: 0,
        };
        let params_buf = buffer::uniform(device, "prism_narrowphase_convex_convex_toi_params", &params);

        // Flatten every hull into shared headers and vertices. Each header
        // records where this body's vertex slice begins in the concatenated
        // array and how long it is; the conservative-advancement kernel reads
        // only the support hull, so no face or loop tables are packed.
        let mut headers: Vec<GpuHullHeader> = Vec::with_capacity(hulls.len());
        let mut packed_vertices: Vec<[f32; 4]> = Vec::new();
        for hull in hulls {
            let vert_offset = u32::try_from(packed_vertices.len()).unwrap_or(u32::MAX);
            for v in hull.vertices() {
                packed_vertices.push([v.x, v.y, v.z, 0.0]);
            }
            let vert_count = u32::try_from(hull.vertices().len()).unwrap_or(u32::MAX);
            headers.push(GpuHullHeader {
                data: [vert_offset, vert_count, 0, 0],
            });
        }

        // A zero-length storage buffer is invalid; a single padding element is
        // never read because the body that would reference it has no vertices
        // of its own.
        if headers.is_empty() {
            headers.push(GpuHullHeader { data: [0, 0, 0, 0] });
        }
        if packed_vertices.is_empty() {
            packed_vertices.push([0.0, 0.0, 0.0, 0.0]);
        }

        let headers_buf =
            buffer::storage_read(device, "prism_narrowphase_convex_convex_toi_hulls", &headers);
        let vertices_buf = buffer::storage_read(
            device,
            "prism_narrowphase_convex_convex_toi_vertices",
            &packed_vertices,
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
            "prism_narrowphase_convex_convex_toi_poses",
            &packed_poses,
        );

        // Motions pack to a GpuMotion each: xyz linear then xyz angular.
        let packed_motions: Vec<GpuMotion> = motions
            .iter()
            .map(|m| GpuMotion {
                linear: [m.linear.x, m.linear.y, m.linear.z, 0.0],
                angular: [m.angular.x, m.angular.y, m.angular.z, 0.0],
            })
            .collect();
        let motions_buf = buffer::storage_read(
            device,
            "prism_narrowphase_convex_convex_toi_motions",
            &packed_motions,
        );

        // Pairs pack to a vec2<u32> each: the two body indices.
        let packed_pairs: Vec<[u32; 2]> = pairs.iter().map(|pair| [pair.a, pair.b]).collect();
        let pairs_buf = buffer::storage_read(
            device,
            "prism_narrowphase_convex_convex_toi_pairs",
            &packed_pairs,
        );

        let tois_bytes = TOI_BYTES * num_pairs as u64;
        let tois_buf =
            buffer::storage_rw_zeroed(device, "prism_narrowphase_convex_convex_toi_out", tois_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_convex_convex_toi_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &headers_buf),
                entry(2, &vertices_buf),
                entry(3, &poses_buf),
                entry(4, &motions_buf),
                entry(5, &pairs_buf),
                entry(6, &tois_buf),
            ],
        });

        let tois_stage =
            buffer::staging(device, "prism_narrowphase_convex_convex_toi_stage", tois_bytes);

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_convex_convex_toi_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_convex_convex_toi_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.tois);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &tois_buf, &tois_stage, tois_bytes);
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<GpuToi>(ctx, &tois_stage);
        raw.iter().map(decode_toi).collect()
    }
}

/// Rebuilds a [`Toi`] from one raw `GPU` slot, or [`None`] when the slot
/// reports a miss.
fn decode_toi(slot: &GpuToi) -> Option<Toi> {
    if slot.normal_hit[3] != HIT_FLAG {
        return None;
    }
    Some(Toi {
        time: slot.point_time[3],
        point: Vec3::new(slot.point_time[0], slot.point_time[1], slot.point_time[2]),
        normal: Vec3::new(slot.normal_hit[0], slot.normal_hit[1], slot.normal_hit[2]),
    })
}
