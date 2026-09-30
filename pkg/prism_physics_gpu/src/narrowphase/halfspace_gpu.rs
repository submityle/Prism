//! Real-device `wgpu` compute implementation of the sphere-halfspace narrow phase.
//!
//! [`GpuHalfspaceNarrowphase`] compiles `shaders/narrowphase_halfspace.wgsl`
//! once and exposes [`GpuHalfspaceNarrowphase::query`], which turns a batch of
//! `(sphere, plane)` couples into one contact slot each on the device. One
//! invocation handles one couple, reading the sphere and the plane and running
//! the same overlap test and manifold construction the
//! [`cpu_halfspace_narrowphase`](super::halfspace::cpu_halfspace_narrowphase)
//! golden twin runs, so a passing real-device parity test is direct evidence the
//! kernel builds the same contacts as the reference.
//!
//! # Dense output
//!
//! The kernel writes a slot per input couple, marking non-penetrating couples
//! with a zero validity flag rather than compacting them away. This keeps the
//! contact index aligned with the couple index (which the parity test relies on)
//! and lets a downstream [`crate::scan`] compaction stream the survivors without
//! a second pass over the couples.
//!
//! Provenance: textbook sphere-plane collision manifold; no Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::broadphase::Particle;
use crate::buffer;
use crate::context::GpuContext;

use super::contact::Contact;
use super::halfspace::{Plane, SpherePlanePair};
use super::layout::{buffer_entry, entry};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuContact`] slot occupies, mirroring the `WGSL` `Contact`
/// stride (two `vec4<f32>`).
const CONTACT_BYTES: u64 = size_of::<GpuContact>() as u64;

/// Uniform parameters shared with `Params` in `shaders/narrowphase_halfspace.wgsl`.
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

/// Readback form of one contact slot, matching the `WGSL` `Contact` layout:
/// `(normal.xyz, depth)` then `(point.xyz, valid)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuContact {
    /// Outward plane normal in `xyz`, penetration depth in `w`.
    normal_depth: [f32; 4],
    /// Surface contact point in `xyz`, validity flag in `w` (nonzero means a
    /// reported contact).
    point_valid: [f32; 4],
}

/// A compiled, reusable `GPU` sphere-halfspace narrow-phase pipeline.
pub struct GpuHalfspaceNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, spheres, planes, pairs, and contacts.
    layout: BindGroupLayout,
    /// The contact kernel: one invocation per candidate couple.
    contacts: ComputePipeline,
}

impl GpuHalfspaceNarrowphase {
    /// Compiles the contact kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHalfspaceNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_halfspace"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_halfspace.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_halfspace_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_halfspace_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let contacts = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_halfspace_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_halfspace"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHalfspaceNarrowphase {
            module,
            layout,
            contacts,
        }
    }

    /// Generates sphere-halfspace contacts for `pairs` over `spheres` and
    /// `planes` on the `GPU`.
    ///
    /// Returns one slot per couple, in input order: [`Some`] carrying the
    /// manifold when the sphere penetrates the plane's halfspace, or [`None`]
    /// when it floats clear. The output matches the
    /// [`cpu_halfspace_narrowphase`](super::halfspace::cpu_halfspace_narrowphase)
    /// twin: the validity flag agrees exactly and the normal, depth, and point
    /// to within the tightest float tolerance (this path carries no square root
    /// or reciprocal).
    ///
    /// An empty `pairs` batch returns an empty vector without touching the device.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        spheres: &[Particle],
        planes: &[Plane],
        pairs: &[SpherePlanePair],
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
        let params_buf = buffer::uniform(device, "prism_narrowphase_halfspace_params", &params);

        // Spheres pack to a single vec4 each: xyz centre, w radius.
        let packed_spheres: Vec<[f32; 4]> = spheres
            .iter()
            .map(|p| [p.position.x, p.position.y, p.position.z, p.radius])
            .collect();
        let spheres_buf = buffer::storage_read(
            device,
            "prism_narrowphase_halfspace_spheres",
            &packed_spheres,
        );

        // Planes pack to a single vec4 each: xyz outward normal, w offset.
        let packed_planes: Vec<[f32; 4]> = planes
            .iter()
            .map(|p| [p.normal.x, p.normal.y, p.normal.z, p.offset])
            .collect();
        let planes_buf =
            buffer::storage_read(device, "prism_narrowphase_halfspace_planes", &packed_planes);

        // Couples pack to a vec2<u32> each: the sphere and plane indices.
        let packed_pairs: Vec<[u32; 2]> =
            pairs.iter().map(|pair| [pair.sphere, pair.plane]).collect();
        let pairs_buf =
            buffer::storage_read(device, "prism_narrowphase_halfspace_pairs", &packed_pairs);

        let contacts_bytes = CONTACT_BYTES * num_pairs as u64;
        let contacts_buf = buffer::storage_rw_zeroed(
            device,
            "prism_narrowphase_halfspace_contacts",
            contacts_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_halfspace_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &spheres_buf),
                entry(2, &planes_buf),
                entry(3, &pairs_buf),
                entry(4, &contacts_buf),
            ],
        });

        let contacts_stage = buffer::staging(
            device,
            "prism_narrowphase_halfspace_contacts_stage",
            contacts_bytes,
        );

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_halfspace_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_halfspace_pass"),
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
                        pair.sphere,
                        pair.plane,
                        Vec3::new(nd[0], nd[1], nd[2]),
                        nd[3],
                        Vec3::new(pv[0], pv[1], pv[2]),
                    ))
                }
            })
            .collect()
    }
}
