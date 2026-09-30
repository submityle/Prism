//! Real-device `wgpu` compute implementation of the OBB-OBB narrow phase.
//!
//! [`GpuObbObbNarrowphase`] compiles `shaders/narrowphase_obb_obb.wgsl` once and
//! exposes [`GpuObbObbNarrowphase::query`], which turns a batch of `(box, box)`
//! couples into one contact slot each on the device. One invocation handles one
//! couple, reading both boxes' centres, axes, and half extents, and running the
//! same fifteen-axis separating-axis test and minimum-translation manifold
//! construction the
//! [`cpu_obb_obb_narrowphase`](super::obb_obb::cpu_obb_obb_narrowphase) golden
//! twin runs, so a passing real-device parity test is direct evidence the kernel
//! builds the same contacts as the reference.
//!
//! # Buffer layout
//!
//! The `CPU`-side packing here is byte-for-byte with the `WGSL` structs:
//!
//! * boxes upload as four `vec4<f32>` each ([`GpuObb`]): the centre in row 0
//!   (`w` unused), then each local axis in `xyz` with its half extent riding in
//!   that row's `w` lane;
//! * pairs upload as one `vec2<u32>` each (box `a` index, box `b` index);
//! * contacts read back as two `vec4<f32>` each ([`GpuContact`]):
//!   `(normal.xyz, depth)` then `(point.xyz, valid)`.
//!
//! # Dense output
//!
//! The kernel writes a slot per input couple, marking non-penetrating couples
//! with a zero validity flag rather than compacting them away. This keeps the
//! contact index aligned with the couple index (which the parity test relies on)
//! and lets a downstream [`crate::scan`] compaction stream the survivors without
//! a second pass over the couples.
//!
//! Provenance: textbook separating-axis oriented-bounding-box collision
//! manifold; no Unreal Engine source or derived code.

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

use super::contact::Contact;
use super::layout::{buffer_entry, entry};
use super::obb::Obb;
use super::obb_obb::ObbObbPair;

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuContact`] slot occupies, mirroring the `WGSL` `Contact`
/// stride (two `vec4<f32>`).
const CONTACT_BYTES: u64 = size_of::<GpuContact>() as u64;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_obb_obb.wgsl`.
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

/// Readback form of one contact slot, matching the `WGSL` `Contact` layout:
/// `(normal.xyz, depth)` then `(point.xyz, valid)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuContact {
    /// Contact normal in `xyz` (from box `a` toward box `b`), penetration depth
    /// in `w`.
    normal_depth: [f32; 4],
    /// Mid-overlap contact point in `xyz`, validity flag in `w` (nonzero means a
    /// reported contact).
    point_valid: [f32; 4],
}

/// A compiled, reusable `GPU` OBB-OBB narrow-phase pipeline.
pub struct GpuObbObbNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, boxes, pairs, and contacts.
    layout: BindGroupLayout,
    /// The contact kernel: one invocation per candidate couple.
    contacts: ComputePipeline,
}

impl GpuObbObbNarrowphase {
    /// Compiles the OBB-OBB contact kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuObbObbNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_obb_obb"),
            source: ShaderSource::Wgsl(include_str!("../shaders/narrowphase_obb_obb.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_obb_obb_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_obb_obb_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let contacts = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_obb_obb_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_obb_obb"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuObbObbNarrowphase {
            module,
            layout,
            contacts,
        }
    }

    /// Generates OBB-OBB contacts for `pairs` over `boxes` on the `GPU`.
    ///
    /// Returns one slot per couple, in input order: [`Some`] carrying the
    /// minimum-translation manifold when the two boxes penetrate, or [`None`]
    /// when a separating axis exists or they exactly graze. The output matches
    /// the
    /// [`cpu_obb_obb_narrowphase`](super::obb_obb::cpu_obb_obb_narrowphase) twin:
    /// the validity flag agrees exactly and the normal, depth, and point to
    /// within a tight float tolerance (the only inexact step is the reciprocal
    /// square root in the edge-axis normalise).
    ///
    /// An empty `pairs` batch returns an empty vector without touching the
    /// device.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        boxes: &[Obb],
        pairs: &[ObbObbPair],
    ) -> Vec<Option<Contact>> {
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
        let params_buf = buffer::uniform(device, "prism_narrowphase_obb_obb_params", &params);

        // Boxes pack to four vec4 each: centre then axis rows carrying the half
        // extents in their w lanes.
        let packed_boxes: Vec<GpuObb> = boxes
            .iter()
            .map(|b| GpuObb {
                center: [b.center.x, b.center.y, b.center.z, 0.0],
                axis0: [b.axes[0].x, b.axes[0].y, b.axes[0].z, b.half_extents.x],
                axis1: [b.axes[1].x, b.axes[1].y, b.axes[1].z, b.half_extents.y],
                axis2: [b.axes[2].x, b.axes[2].y, b.axes[2].z, b.half_extents.z],
            })
            .collect();
        let boxes_buf =
            buffer::storage_read(device, "prism_narrowphase_obb_obb_boxes", &packed_boxes);

        // Couples pack to a vec2<u32> each: the two box indices.
        let packed_pairs: Vec<[u32; 2]> = pairs.iter().map(|pair| [pair.a, pair.b]).collect();
        let pairs_buf =
            buffer::storage_read(device, "prism_narrowphase_obb_obb_pairs", &packed_pairs);

        let contacts_bytes = CONTACT_BYTES * num_pairs as u64;
        let contacts_buf =
            buffer::storage_rw_zeroed(device, "prism_narrowphase_obb_obb_contacts", contacts_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_obb_obb_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &boxes_buf),
                entry(2, &pairs_buf),
                entry(3, &contacts_buf),
            ],
        });

        let contacts_stage = buffer::staging(
            device,
            "prism_narrowphase_obb_obb_contacts_stage",
            contacts_bytes,
        );

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_obb_obb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_obb_obb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.contacts);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        buffer::copy(&mut encoder, &contacts_buf, &contacts_stage, contacts_bytes);
        ctx.queue().submit([encoder.finish()]);

        let raw = buffer::read_back::<GpuContact>(ctx, &contacts_stage);
        raw.iter()
            .zip(pairs)
            .map(|(slot, pair)| {
                if slot.point_valid[3] == 0.0 {
                    None
                } else {
                    let nd = slot.normal_depth;
                    let pv = slot.point_valid;
                    Some(Contact::new(
                        pair.a,
                        pair.b,
                        Vec3::new(nd[0], nd[1], nd[2]),
                        nd[3],
                        Vec3::new(pv[0], pv[1], pv[2]),
                    ))
                }
            })
            .collect()
    }
}
