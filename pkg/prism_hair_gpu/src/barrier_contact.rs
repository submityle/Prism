//! `wgpu` compute twin of Prism's barrier-based frictional hair contact
//! ([`resolve_contact`](prism_render_architecture::hair::barrier_contact::resolve_contact)).
//!
//! [`resolve_contact`](prism_render_architecture::hair::barrier_contact::resolve_contact)
//! is the per-contact core of the real-time simplified Codimensional `IPC`
//! (`C-IPC`, Li 2021) contact tier: a C¹ rational soft barrier pushes two
//! endpoints apart along the contact normal and a semi-implicit `Coulomb`
//! friction impulse cancels the tangential slide, clamped to the friction cone.
//! This crate is that primitive's isolated twin: one thread per contact reads a
//! pair `(a, b, normal, distance, rel_velocity)` from its *original* state,
//! evaluates the resolution and the corrected positions/velocities, and writes
//! them into its own disjoint output slot.
//!
//! # `Jacobi`, not `Gauss-Seidel`
//!
//! The reference also ships a batch [`resolve_contacts`] wrapper that resolves a
//! contact list in index order, so endpoints shared between contacts see each
//! other's partial corrections (`Gauss-Seidel`). That ordering is inherently
//! sequential and is deliberately *not* twinned here. Instead this kernel
//! twins the pure per-contact [`reference_resolve`] evaluated from each
//! contact's original endpoint state (`Jacobi`), which is embarrassingly
//! parallel and disjoint-write — the correctness unit the batch loop cannot
//! isolate.
//!
//! # Barrier (no `ln`)
//!
//! Prism's workspace bans `ln`/`exp`/`pow`, so the textbook `C-IPC` log barrier
//! is off-limits. The reference uses a rational soft barrier evaluated on the
//! clamped distance `d_e = clamp(d, d_floor, dhat)` whose repulsive force
//! vanishes at `d = dhat` (C¹) and stays large-but-finite at the floor. The
//! host uploads *already-sanitized* params, so the kernel skips the
//! `sanitized()`/non-finite guards and only reproduces the `d >= dhat -> 0`
//! short-circuit and the clamped evaluation.
//!
//! # Correctness model
//!
//! Core math is closed-form (one `sqrt`, divides, mul-add), but a `GPU` may fuse
//! multiply-adds the scalar reference leaves separate, so positions, velocities
//! and impulses can differ by a few low-mantissa `ULP`; the parity test asserts
//! each within `abs_diff < 1e-4` or `rel_diff < 1e-3`. The `slipping` flag is a
//! branch pick (friction clamped to the `Coulomb` cone or not), so the test
//! cases sit well clear of the cone threshold and the flag is asserted exactly.
//!
//! # Portability
//!
//! The kernel uses only `sqrt`/`dot`/`min`/`max`/`clamp`/multiply/divide in the
//! portable core-`WGSL` subset — no `exp`/`pow`/trig, atomics or optional device
//! feature — so the twin runs unmodified on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: real-time simplified `C-IPC` (Li 2021) barrier + semi-implicit
//! `Coulomb` friction, plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::barrier_contact::{
    resolve_contact, BarrierParams, ContactPoint, Vec3,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One contact to resolve: the two endpoints plus the contact normal, gap and
/// relative velocity.
#[derive(Clone, Copy, Debug)]
pub struct ContactInput {
    /// First endpoint (pushed along `+normal`).
    pub a: ContactPoint,
    /// Second endpoint (pushed along `-normal`).
    pub b: ContactPoint,
    /// Contact normal pointing from `b` toward `a` (normalized internally).
    pub normal: Vec3,
    /// Current contact gap; at/beyond `dhat` the contact is a no-op.
    pub distance: f32,
    /// Velocity of `a` relative to `b`, used for the friction impulse.
    pub rel_velocity: Vec3,
}

/// One resolved contact: the corrected endpoints plus the scalar outcome,
/// mirroring the reference `(a, b, ContactResolution)`.
#[derive(Clone, Copy, Debug)]
pub struct ContactOutput {
    /// First endpoint after the barrier push and friction impulse.
    pub a: ContactPoint,
    /// Second endpoint after the barrier push and friction impulse.
    pub b: ContactPoint,
    /// Scalar normal (repulsive) impulse applied along the normal.
    pub normal_impulse: f32,
    /// Scalar friction impulse applied in the tangent plane.
    pub friction_impulse: f32,
    /// `true` when the friction impulse was clamped to the `Coulomb` cone.
    pub slipping: bool,
}

/// Already-sanitized params plus the contact count, padded to `32` bytes to
/// match `Params` in `shaders/barrier_contact.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    dhat: f32,
    stiffness: f32,
    d_floor: f32,
    friction_mu: f32,
    contact_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// One contact in the shader upload layout: positions carry inverse mass in
/// `w`, the normal carries the gap in `w`; matches `Contact` in
/// `shaders/barrier_contact.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuContact {
    a_pos: [f32; 4],
    a_vel: [f32; 4],
    b_pos: [f32; 4],
    b_vel: [f32; 4],
    normal: [f32; 4],
    rel_velocity: [f32; 4],
}

/// One resolved contact in the shader upload layout, matching `Resolved` in
/// `shaders/barrier_contact.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResolved {
    a_pos: [f32; 4],
    a_vel: [f32; 4],
    b_pos: [f32; 4],
    b_vel: [f32; 4],
    scalars: [f32; 4],
}

/// A compiled, reusable barrier-contact pipeline.
pub struct GpuHairBarrierContact {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairBarrierContact {
    /// Compiles the barrier-contact kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairBarrierContact {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_barrier_contact"),
            source: ShaderSource::Wgsl(include_str!("../shaders/barrier_contact.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_barrier_contact_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_barrier_contact_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_barrier_contact_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairBarrierContact {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every contact independently from its original endpoint state
    /// (`Jacobi`), one output per input in order.
    ///
    /// The result for contact `i` equals the `CPU` golden [`reference_resolve`]
    /// (built on
    /// [`resolve_contact`](prism_render_architecture::hair::barrier_contact::resolve_contact))
    /// to within the tolerance documented on this module, with `slipping`
    /// bit-exact. An empty slice yields an empty vector without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        contacts: &[ContactInput],
        params: BarrierParams,
    ) -> Vec<ContactOutput> {
        let contact_count = contacts.len();
        if contact_count == 0 {
            return Vec::new();
        }

        // The host sanitizes params once; the kernel trusts them.
        let p = params.sanitized();

        let uploads: Vec<GpuContact> = contacts
            .iter()
            .map(|c| GpuContact {
                a_pos: [c.a.position.x, c.a.position.y, c.a.position.z, c.a.inv_mass],
                a_vel: [c.a.velocity.x, c.a.velocity.y, c.a.velocity.z, 0.0],
                b_pos: [c.b.position.x, c.b.position.y, c.b.position.z, c.b.inv_mass],
                b_vel: [c.b.velocity.x, c.b.velocity.y, c.b.velocity.z, 0.0],
                normal: [c.normal.x, c.normal.y, c.normal.z, c.distance],
                rel_velocity: [c.rel_velocity.x, c.rel_velocity.y, c.rel_velocity.z, 0.0],
            })
            .collect();

        let device = ctx.device();
        let uniforms = Params {
            dhat: p.dhat,
            stiffness: p.stiffness,
            d_floor: p.d_floor,
            friction_mu: p.friction_mu,
            contact_count: contact_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let out_bytes = (contact_count as u64) * (size_of::<GpuResolved>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_barrier_contact_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let contacts_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_barrier_contact_contacts"),
            contents: bytemuck::cast_slice(&uploads),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_barrier_contact_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_barrier_contact_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_barrier_contact_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: contacts_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_barrier_contact_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_barrier_contact_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (contact_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResolved>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        raw.iter()
            .zip(contacts.iter())
            .map(|(r, input)| ContactOutput {
                a: ContactPoint::new(
                    Vec3::new(r.a_pos[0], r.a_pos[1], r.a_pos[2]),
                    Vec3::new(r.a_vel[0], r.a_vel[1], r.a_vel[2]),
                    input.a.inv_mass,
                ),
                b: ContactPoint::new(
                    Vec3::new(r.b_pos[0], r.b_pos[1], r.b_pos[2]),
                    Vec3::new(r.b_vel[0], r.b_vel[1], r.b_vel[2]),
                    input.b.inv_mass,
                ),
                normal_impulse: r.scalars[0],
                friction_impulse: r.scalars[1],
                slipping: r.scalars[2] > 0.5,
            })
            .collect()
    }
}

/// Runs the golden per-contact resolution directly on a fresh copy of the
/// endpoints; a thin reference so the parity test can name one golden path.
///
/// This is the `Jacobi` unit the kernel twins: it evaluates
/// [`resolve_contact`](prism_render_architecture::hair::barrier_contact::resolve_contact)
/// from `input`'s original endpoint state and returns the corrected endpoints
/// plus the resolution.
#[must_use]
pub fn reference_resolve(input: &ContactInput, params: BarrierParams) -> ContactOutput {
    let mut a = input.a;
    let mut b = input.b;
    let res = resolve_contact(
        &mut a,
        &mut b,
        input.normal,
        input.distance,
        input.rel_velocity,
        params,
    );
    ContactOutput {
        a,
        b,
        normal_impulse: res.normal_impulse,
        friction_impulse: res.friction_impulse,
        slipping: res.slipping,
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
