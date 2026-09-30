//! Real-device `wgpu` compute implementation of the capsule-capsule narrow phase.
//!
//! [`GpuCapsuleCapsuleNarrowphase`] compiles
//! `shaders/narrowphase_capsule_capsule.wgsl` once and exposes
//! [`GpuCapsuleCapsuleNarrowphase::query`], which turns a batch of candidate
//! capsule-capsule pairs into one contact slot each on the device. One
//! invocation handles one pair, reading the two capsule proxies and running the
//! same segment-segment closest-point test and manifold construction the
//! [`cpu_capsule_capsule_narrowphase`](super::capsule_capsule::cpu_capsule_capsule_narrowphase)
//! golden twin runs, so a passing real-device parity test is direct evidence
//! the kernel builds the same contacts as the reference.
//!
//! # Dense output
//!
//! The kernel writes a slot per input pair, marking non-penetrating pairs with a
//! zero validity flag rather than compacting them away. This keeps the contact
//! index aligned with the pair index (which the parity test relies on) and lets
//! a downstream [`crate::scan`] compaction stream the survivors without a second
//! pass over the pairs.
//!
//! Provenance: textbook capsule-capsule (segment-segment) closest-feature
//! collision manifold; no Unreal Engine source or derived code.

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
use super::capsule_capsule::CapsuleCapsulePair;
use super::contact::Contact;
use super::layout::{buffer_entry, entry};

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Bytes each output [`GpuContact`] slot occupies, mirroring the `WGSL`
/// `Contact` stride (two `vec4<f32>`).
const CONTACT_BYTES: u64 = size_of::<GpuContact>() as u64;

/// Uniform parameters shared with `Params` in
/// `shaders/narrowphase_capsule_capsule.wgsl`.
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

/// Upload form of one capsule, matching the `WGSL` `Capsule` layout:
/// `(p0.xyz, radius)` then `(p1.xyz, pad)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCapsule {
    /// First endpoint in `xyz`, swept radius in `w`.
    p0_radius: [f32; 4],
    /// Second endpoint in `xyz`, unused padding in `w`.
    p1_pad: [f32; 4],
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

/// A compiled, reusable `GPU` capsule-capsule narrow-phase pipeline.
pub struct GpuCapsuleCapsuleNarrowphase {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring params, capsules, pairs, and the contacts.
    layout: BindGroupLayout,
    /// The contact kernel: one invocation per candidate pair.
    contacts: ComputePipeline,
}

impl GpuCapsuleCapsuleNarrowphase {
    /// Compiles the capsule-capsule contact kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCapsuleCapsuleNarrowphase {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_narrowphase_capsule_capsule"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/narrowphase_capsule_capsule.wgsl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_narrowphase_capsule_capsule_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_narrowphase_capsule_capsule_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let contacts = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_narrowphase_capsule_capsule_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("narrowphase_capsule_capsule"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCapsuleCapsuleNarrowphase {
            module,
            layout,
            contacts,
        }
    }

    /// Generates capsule-capsule contacts for `pairs` over `capsules` on the `GPU`.
    ///
    /// Returns one slot per pair, in input order: [`Some`] carrying the manifold
    /// when the two capsules penetrate, or [`None`] when they are separated or
    /// exactly touching. The output matches the
    /// [`cpu_capsule_capsule_narrowphase`](super::capsule_capsule::cpu_capsule_capsule_narrowphase)
    /// twin: the validity flag agrees exactly and the normal, depth, and point
    /// within the tight tolerance the square root and reciprocal impose.
    ///
    /// An empty `pairs` batch returns an empty vector without touching the device.
    #[must_use]
    pub fn query(
        &self,
        ctx: &GpuContext,
        capsules: &[Capsule],
        pairs: &[CapsuleCapsulePair],
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
            buffer::uniform(device, "prism_narrowphase_capsule_capsule_params", &params);

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
            "prism_narrowphase_capsule_capsule_capsules",
            &packed_capsules,
        );

        // Pairs pack to a vec2<u32> each: the two capsule indices.
        let packed_pairs: Vec<[u32; 2]> = pairs.iter().map(|pair| [pair.a, pair.b]).collect();
        let pairs_buf = buffer::storage_read(
            device,
            "prism_narrowphase_capsule_capsule_pairs",
            &packed_pairs,
        );

        let contacts_bytes = CONTACT_BYTES * num_pairs as u64;
        let contacts_buf = buffer::storage_rw_zeroed(
            device,
            "prism_narrowphase_capsule_capsule_contacts",
            contacts_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_narrowphase_capsule_capsule_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &capsules_buf),
                entry(2, &pairs_buf),
                entry(3, &contacts_buf),
            ],
        });

        let contacts_stage = buffer::staging(
            device,
            "prism_narrowphase_capsule_capsule_contacts_stage",
            contacts_bytes,
        );

        let groups = u32::try_from(num_pairs.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_narrowphase_capsule_capsule_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_narrowphase_capsule_capsule_pass"),
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
