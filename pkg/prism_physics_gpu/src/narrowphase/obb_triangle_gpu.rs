//! Real-device `wgpu` compute implementation of the OBB-versus-triangle narrow
//! phase.
//!
//! [`GpuObbTriangleNarrowphase`] compiles
//! `shaders/narrowphase_obb_triangle.wgsl` once and exposes
//! [`GpuObbTriangleNarrowphase::query`], which turns a batch of candidate
//! box-versus-triangle couples into one contact slot each on the device. One
//! invocation handles one couple, reading the box's centre, axes, and half
//! extents and the triangle's three vertices, and running the same thirteen-axis
//! separating-axis test and manifold construction the
//! [`cpu_obb_triangle_narrowphase`](super::obb_triangle::cpu_obb_triangle_narrowphase)
//! golden twin runs, so a passing real-device parity test is direct evidence the
//! kernel builds the same contacts as the reference.
//!
//! # Buffer layout
//!
//! The `CPU`-side packing here is byte-for-byte with the `WGSL` structs:
//!
//! * boxes upload as four `vec4<f32>` each ([`GpuObb`]): the centre in row 0
//!   (`w` unused), then each local axis in `xyz` with its half extent riding in
//!   that row's `w` lane;
//! * triangles upload as three `vec4<f32>` each ([`GpuTriangle`]): one vertex in
//!   the `xyz` of each row, the `w` lanes unused padding;
//! * pairs upload as one `vec2<u32>` each (box index, triangle index);
//! * contacts read back as two `vec4<f32>` each ([`GpuContact`]):
//!   `(normal.xyz, depth)` then `(point.xyz, valid)`.
//!
//! # Dense output
//!
//! The kernel writes a slot per input couple, marking non-penetrating couples
//! with a zero validity flag rather than compacting them away. This keeps the
//! contact index aligned with the couple index (which the parity test relies on)
//! and lets a downstream [`crate::scan`] compaction stream the survivors without
//! a second pass over the pairs.
//!
//! Provenance: the thirteen-axis box-triangle separating-axis set is Tomas
//! Akenine-Möller, *Fast 3D Triangle-Box Overlap Testing* (2001); the
//! separating-axis minimum-translation manifold and the closest-point-on-
//! triangle Voronoi cascade are Christer Ericson, *Real-Time Collision
//! Detection* (2004), sections 5.2.9 and 5.1.5. No Unreal Engine source or
//! derived code.

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
use super::obb_triangle::ObbTrianglePair;
use super::sphere_triangle::Triangle;

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuContact`] slot occupies, mirroring the `WGSL` `Contact`
/// stride (two `vec4<f32>`).
const CONTACT_BYTES: u64 = size_of::<GpuContact>() as u64;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_obb_triangle.wgsl`.
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

/// Upload form of one [`Triangle`], matching the `WGSL` `Triangle` struct: each
/// vertex in the `xyz` of its row, the `w` lanes unused padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTriangle {
    /// First vertex in `xyz`; `w` is unused padding.
    a: [f32; 4],
    /// Second vertex in `xyz`; `w` is unused padding.
    b: [f32; 4],
    /// Third vertex in `xyz`; `w` is unused padding.
    c: [f32; 4],
}

/// Readback form of one contact slot, matching the `WGSL` `Contact` layout:
/// `(normal.xyz, depth)` then `(point.xyz, valid)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuContact {
    /// Unit contact normal in `xyz`, penetration depth in `w`.
    normal_depth: [f32; 4],
    /// World contact point in `xyz`, validity flag in `w` (nonzero means a
    /// reported contact).
    point_valid: [f32; 4],
}

/// A compiled, reusable `GPU` OBB-versus-triangle narrow-phase pipeline.
pub struct GpuObbTriangleNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, boxes, triangles, pairs, and
    /// contacts.
    layout: BindGroupLayout,
    /// The contact kernel: one invocation per candidate couple.
    contacts: ComputePipeline,
}

impl GpuObbTriangleNarrowphase {
    /// Compiles the OBB-versus-triangle contact kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuObbTriangleNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_obb_triangle"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_obb_triangle.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_obb_triangle_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_obb_triangle_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let contacts = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_obb_triangle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_obb_triangle"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuObbTriangleNarrowphase {
            module,
            layout,
            contacts,
        }
    }

    /// Generates OBB-versus-triangle contacts for `pairs` on the `GPU`.
    ///
    /// Returns one slot per couple, in input order: [`Some`] carrying the
    /// minimum-translation manifold when the box penetrates the triangle, or
    /// [`None`] when a separating axis exists or they exactly graze. The output
    /// matches the
    /// [`cpu_obb_triangle_narrowphase`](super::obb_triangle::cpu_obb_triangle_narrowphase)
    /// twin: the validity flag agrees exactly and the normal, depth, and point
    /// agree within the tight tolerance the reciprocal square roots and the
    /// barycentric reciprocals impose.
    ///
    /// An empty `pairs` batch returns an empty vector without touching the
    /// device.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        boxes: &[Obb],
        triangles: &[Triangle],
        pairs: &[ObbTrianglePair],
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
        let params_buf =
            buffer::uniform(device, "prism_narrowphase_obb_triangle_params", &params);

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
            buffer::storage_read(device, "prism_narrowphase_obb_triangle_boxes", &packed_boxes);

        // Triangles pack to three vec4 each: one vertex per row, w unused.
        let packed_triangles: Vec<GpuTriangle> = triangles
            .iter()
            .map(|t| GpuTriangle {
                a: [t.a.x, t.a.y, t.a.z, 0.0],
                b: [t.b.x, t.b.y, t.b.z, 0.0],
                c: [t.c.x, t.c.y, t.c.z, 0.0],
            })
            .collect();
        let triangles_buf = buffer::storage_read(
            device,
            "prism_narrowphase_obb_triangle_triangles",
            &packed_triangles,
        );

        // Pairs pack to a vec2<u32> each: the box index then the triangle index.
        let packed_pairs: Vec<[u32; 2]> =
            pairs.iter().map(|pair| [pair.obb, pair.triangle]).collect();
        let pairs_buf =
            buffer::storage_read(device, "prism_narrowphase_obb_triangle_pairs", &packed_pairs);

        let contacts_bytes = CONTACT_BYTES * num_pairs as u64;
        let contacts_buf = buffer::storage_rw_zeroed(
            device,
            "prism_narrowphase_obb_triangle_contacts",
            contacts_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_obb_triangle_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &boxes_buf),
                entry(2, &triangles_buf),
                entry(3, &pairs_buf),
                entry(4, &contacts_buf),
            ],
        });

        let contacts_stage = buffer::staging(
            device,
            "prism_narrowphase_obb_triangle_contacts_stage",
            contacts_bytes,
        );

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_obb_triangle_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_obb_triangle_pass"),
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
                        pair.obb,
                        pair.triangle,
                        Vec3::new(nd[0], nd[1], nd[2]),
                        nd[3],
                        Vec3::new(pv[0], pv[1], pv[2]),
                    ))
                }
            })
            .collect()
    }
}
