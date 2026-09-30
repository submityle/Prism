//! Real-device `wgpu` compute implementation of the sphere-sphere narrow phase.
//!
//! [`GpuNarrowphase`] compiles `shaders/narrowphase_contacts.wgsl` once and
//! exposes [`GpuNarrowphase::query`], which turns a batch of broad-phase
//! candidate pairs into one contact slot each on the device. One invocation
//! handles one pair, reading the two particles' bounding spheres and running the
//! same overlap test and manifold construction the
//! [`cpu_narrowphase`](super::cpu::cpu_narrowphase) golden twin runs, so a
//! passing real-device parity test is direct evidence the kernel builds the same
//! contacts as the reference.
//!
//! # Dense output
//!
//! The kernel writes a slot per input pair, marking non-penetrating pairs with a
//! zero validity flag rather than compacting them away. This keeps the contact
//! index aligned with the pair index (which the parity test relies on) and lets
//! a downstream [`crate::scan`] compaction stream the survivors without a second
//! pass over the pairs.
//!
//! Provenance: textbook sphere-sphere collision manifold; no Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::broadphase::{CandidatePair, Particle};
use crate::buffer;
use crate::context::GpuContext;

use super::contact::Contact;
use super::layout::{buffer_entry, entry};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuContact`] slot occupies, mirroring the `WGSL` `Contact`
/// stride (two `vec4<f32>`).
const CONTACT_BYTES: u64 = size_of::<GpuContact>() as u64;

/// Uniform parameters shared with `Params` in `shaders/narrowphase_contacts.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of candidate pairs queued in the pair buffer.
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
    /// Unit contact normal in `xyz`, penetration depth in `w`.
    normal_depth: [f32; 4],
    /// World contact point in `xyz`, validity flag in `w` (nonzero means a
    /// reported contact).
    point_valid: [f32; 4],
}

/// A compiled, reusable `GPU` sphere-sphere narrow-phase pipeline.
pub struct GpuNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, particles, pairs, and the contacts.
    layout: BindGroupLayout,
    /// The contact kernel: one invocation per candidate pair.
    contacts: ComputePipeline,
}

impl GpuNarrowphase {
    /// Compiles the contact kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_contacts"),
            source: ShaderSource::Wgsl(include_str!("../shaders/narrowphase_contacts.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let contacts = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_contacts"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuNarrowphase {
            module,
            layout,
            contacts,
        }
    }

    /// Generates sphere-sphere contacts for `pairs` over `particles` on the `GPU`.
    ///
    /// Returns one slot per pair, in input order: [`Some`] carrying the manifold
    /// when the two spheres penetrate, or [`None`] when they are separated or
    /// exactly touching. The output matches the
    /// [`cpu_narrowphase`](super::cpu::cpu_narrowphase) twin: the validity flag
    /// agrees exactly and the normal, depth, and point agree within the tight
    /// tolerance the square root and reciprocal impose.
    ///
    /// An empty `pairs` batch returns an empty vector without touching the device.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        particles: &[Particle],
        pairs: &[CandidatePair],
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
        let params_buf = buffer::uniform(device, "prism_narrowphase_params", &params);

        // Particles pack to a single vec4 each: xyz centre, w radius.
        let packed_particles: Vec<[f32; 4]> = particles
            .iter()
            .map(|p| [p.position.x, p.position.y, p.position.z, p.radius])
            .collect();
        let particles_buf =
            buffer::storage_read(device, "prism_narrowphase_particles", &packed_particles);

        // Pairs pack to a vec2<u32> each: the two particle indices.
        let packed_pairs: Vec<[u32; 2]> = pairs.iter().map(|pair| [pair.a, pair.b]).collect();
        let pairs_buf = buffer::storage_read(device, "prism_narrowphase_pairs", &packed_pairs);

        let contacts_bytes = CONTACT_BYTES * num_pairs as u64;
        let contacts_buf =
            buffer::storage_rw_zeroed(device, "prism_narrowphase_contacts", contacts_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &particles_buf),
                entry(2, &pairs_buf),
                entry(3, &contacts_buf),
            ],
        });

        let contacts_stage =
            buffer::staging(device, "prism_narrowphase_contacts_stage", contacts_bytes);

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_pass"),
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
