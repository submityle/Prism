//! Real-device `wgpu` compute implementation of the two-point capsule-OBB
//! narrow phase.
//!
//! [`GpuCapsuleObbManifoldNarrowphase`] compiles
//! `shaders/narrowphase_capsule_obb_manifold.wgsl` once and exposes
//! [`GpuCapsuleObbManifoldNarrowphase::query`], which turns a batch of
//! `(capsule, box)` couples into one contact manifold each on the device. One
//! invocation handles one couple: it runs the same single deepest-feature test
//! as the single-point kernel, then clips the capsule axis to the reference box
//! face exactly as the
//! [`cpu_capsule_obb_manifold`](super::capsule_obb_manifold::cpu_capsule_obb_manifold)
//! golden twin does, so a passing real-device parity test is direct evidence the
//! kernel builds the same manifolds as the reference.
//!
//! # Buffer layout
//!
//! The `CPU`-side packing here is byte-for-byte with the `WGSL` structs:
//!
//! * capsules upload as two `vec4<f32>` each ([`GpuCapsule`]): `(p0.xyz, radius)`
//!   then `(p1.xyz, pad)`;
//! * boxes upload as four `vec4<f32>` each ([`GpuObb`]): the centre in row 0
//!   (`w` unused), then each local axis in `xyz` with its half extent riding in
//!   that row's `w` lane;
//! * pairs upload as one `vec2<u32>` each (capsule index, box index);
//! * manifolds read back as five `vec4<f32>` each ([`GpuManifold`]):
//!   `(normal.xyz, count)` then four `(position.xyz, depth)` points.
//!
//! # Dense output
//!
//! The kernel writes a slot per input couple, marking non-penetrating couples
//! with a zero point count rather than compacting them away. This keeps the
//! manifold index aligned with the couple index, which the parity test relies
//! on, and lets a downstream [`crate::scan`] compaction stream the survivors
//! without a second pass over the couples.
//!
//! Provenance: textbook capsule-versus-oriented-bounding-box closest-feature
//! collision plus Liang-Barsky segment-rectangle clipping; no Unreal Engine
//! source or derived code.

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

use super::capsule::Capsule;
use super::capsule_obb::CapsuleObbPair;
use super::layout::{buffer_entry, entry};
use super::manifold::{ContactManifold, ManifoldPoint, MAX_MANIFOLD_POINTS};
use super::obb::Obb;

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuManifold`] slot occupies, mirroring the `WGSL`
/// `Manifold` stride (five `vec4<f32>`).
const MANIFOLD_BYTES: u64 = size_of::<GpuManifold>() as u64;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_capsule_obb_manifold.wgsl`.
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

/// Upload form of one [`Capsule`], matching the `WGSL` `Capsule` struct:
/// `(p0.xyz, radius)` then `(p1.xyz, pad)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCapsule {
    /// Segment start in `xyz`, swept radius in `w`.
    p0_radius: [f32; 4],
    /// Segment end in `xyz`; `w` is unused padding.
    p1_pad: [f32; 4],
}

/// Upload form of one [`Obb`], matching the `WGSL` `Obb` struct: the centre in
/// `center` (`w` unused), then each local axis in the `xyz` of its row with the
/// matching half extent in that row's `w` lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuObb {
    /// Box centre in `xyz`; `w` is unused padding.
    center: [f32; 4],
    /// Local axis 0 in `xyz`, half extent along it in `w`.
    axis0: [f32; 4],
    /// Local axis 1 in `xyz`, half extent along it in `w`.
    axis1: [f32; 4],
    /// Local axis 2 in `xyz`, half extent along it in `w`.
    axis2: [f32; 4],
}

/// Readback form of one manifold slot, matching the `WGSL` `Manifold` layout:
/// `(normal.xyz, count)` then four `(position.xyz, depth)` points.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuManifold {
    /// Shared contact normal in `xyz` (the reference face's outward push-out
    /// direction, from the box toward the capsule), live point count in `w`.
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

/// A compiled, reusable `GPU` two-point capsule-OBB narrow-phase pipeline.
pub struct GpuCapsuleObbManifoldNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, capsules, boxes, pairs, and
    /// manifolds.
    layout: BindGroupLayout,
    /// The manifold kernel: one invocation per candidate couple.
    manifolds: ComputePipeline,
}

impl GpuCapsuleObbManifoldNarrowphase {
    /// Compiles the two-point capsule-OBB manifold kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapsuleObbManifoldNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_capsule_obb_manifold"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_capsule_obb_manifold.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_capsule_obb_manifold_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_capsule_obb_manifold_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let manifolds = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_capsule_obb_manifold_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_capsule_obb_manifold"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCapsuleObbManifoldNarrowphase {
            module,
            layout,
            manifolds,
        }
    }

    /// Generates capsule-OBB contact manifolds for `pairs` over `capsules` and
    /// `boxes` on the `GPU`.
    ///
    /// Returns one slot per couple, in input order: [`Some`] carrying a one- or
    /// two-point manifold when the capsule penetrates the box, or [`None`] when
    /// it is separated or exactly grazing. The output matches the
    /// [`cpu_capsule_obb_manifold`](super::capsule_obb_manifold::cpu_capsule_obb_manifold)
    /// twin: the point count agrees exactly and the normal, positions, and
    /// depths to within a tight float tolerance (the only inexact steps are the
    /// square roots and reciprocals in the closest-feature search, the
    /// outside-face normalise, and the edge-crossing solves).
    ///
    /// An empty `pairs` batch returns an empty vector without touching the
    /// device.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        capsules: &[Capsule],
        boxes: &[Obb],
        pairs: &[CapsuleObbPair],
    ) -> Vec<Option<ContactManifold>> {
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
            "prism_narrowphase_capsule_obb_manifold_params",
            &params,
        );

        // Capsules pack to two vec4 each: (p0.xyz, radius) and (p1.xyz, pad).
        let packed_capsules: Vec<GpuCapsule> = capsules
            .iter()
            .map(|cap| GpuCapsule {
                p0_radius: [cap.p0.x, cap.p0.y, cap.p0.z, cap.radius],
                p1_pad: [cap.p1.x, cap.p1.y, cap.p1.z, 0.0],
            })
            .collect();
        let capsules_buf = buffer::storage_read(
            device,
            "prism_narrowphase_capsule_obb_manifold_capsules",
            &packed_capsules,
        );

        let packed_boxes: Vec<GpuObb> = boxes
            .iter()
            .map(|b| GpuObb {
                center: [b.center.x, b.center.y, b.center.z, 0.0],
                axis0: [b.axes[0].x, b.axes[0].y, b.axes[0].z, b.half_extents.x],
                axis1: [b.axes[1].x, b.axes[1].y, b.axes[1].z, b.half_extents.y],
                axis2: [b.axes[2].x, b.axes[2].y, b.axes[2].z, b.half_extents.z],
            })
            .collect();
        let boxes_buf = buffer::storage_read(
            device,
            "prism_narrowphase_capsule_obb_manifold_boxes",
            &packed_boxes,
        );

        let packed_pairs: Vec<[u32; 2]> =
            pairs.iter().map(|pair| [pair.capsule, pair.obb]).collect();
        let pairs_buf = buffer::storage_read(
            device,
            "prism_narrowphase_capsule_obb_manifold_pairs",
            &packed_pairs,
        );

        let manifolds_bytes = MANIFOLD_BYTES * num_pairs as u64;
        let manifolds_buf = buffer::storage_rw_zeroed(
            device,
            "prism_narrowphase_capsule_obb_manifold_out",
            manifolds_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_capsule_obb_manifold_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &capsules_buf),
                entry(2, &boxes_buf),
                entry(3, &pairs_buf),
                entry(4, &manifolds_buf),
            ],
        });

        let manifolds_stage = buffer::staging(
            device,
            "prism_narrowphase_capsule_obb_manifold_stage",
            manifolds_bytes,
        );

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_capsule_obb_manifold_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_capsule_obb_manifold_pass"),
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
fn decode_manifold(slot: &GpuManifold, pair: &CapsuleObbPair) -> Option<ContactManifold> {
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
    Some(ContactManifold::new(
        pair.capsule,
        pair.obb,
        normal,
        count,
        &points,
    ))
}
