//! `wgpu` compute twin of Prism's in-place strand body-collision resolve
//! ([`resolve_strand_collisions`](prism_render_architecture::hair::collision::resolve_strand_collisions)).
//!
//! `TressFX`-class strand solvers keep hair off the body by projecting every
//! free guide particle out of a small set of analytic collider proxies (spheres
//! and capsules fitted to the head, neck and shoulders) after each constraint
//! sweep (design §6.2). The `CPU` golden `resolve_strand_collisions` visits each
//! particle and, for a free particle, applies every collider in order via
//! [`Collider::push_out`](prism_render_architecture::hair::collision::Collider::push_out)
//! — the last collider to push wins for that particle. A pinned particle
//! (`inverse_mass <= 0`, the skinned root) is never moved, and an empty collider
//! set is a no-op. This twin runs that whole pass batch-wide on the device, one
//! thread per particle folding a per-dispatch *shared* collider array.
//!
//! # Relationship to the sibling collider twin
//!
//! This is deliberately a different pass from
//! [`GpuColliderProjector`](crate::GpuColliderProjector), which evaluates a flat
//! batch of independent `(point, collider)` queries with no fold. Here each
//! thread owns one particle and sequentially folds the shared colliders,
//! reproducing the real per-pass body solve
//! (`pos_{c+1} = colliders[c].push_out(pos_c)`), including the pinned skip and
//! the last-push-wins ordering — not just a single push-out.
//!
//! # What the kernel evaluates
//!
//! [`GpuStrandCollisionResolve::eval`] takes a batch of guide particles (only
//! position and inverse mass matter) and a single shared collider array, and
//! returns one resolved position per particle in input order. The particle
//! index is the invocation id (`@compute @workgroup_size(64)`, a
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! particle count early-return.
//!
//! # Correctness model
//!
//! The per-collider push-out contains no transcendental call, so the `CPU` and
//! `GPU` evaluate the same closed-form geometry and diverge only through legal
//! fused-multiply-add contraction in the sphere `normalize` (`sqrt` +
//! reciprocal). Each component is matched against a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`), not bit-for-bit. The pinned skip,
//! the inert-radius and already-exterior guards, the center-escape fallback and
//! the capsule segment clamp are all reproduced exactly, so a swapped branch or
//! a missing guard still fails the parity test.
//!
//! # Portability
//!
//! The kernel uses only compares, `dot`, `sqrt`, a reciprocal multiply, `clamp`,
//! `min`/`max` and add in the portable core-`WGSL` subset — no `exp`, `pow` or
//! optional device feature — so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard analytic sphere/capsule collider push-out plus an
//! in-order fold and `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use prism_render_architecture::hair::collision::{resolve_strand_collisions, Collider};
use prism_render_architecture::hair::dynamics::StrandParticle;

use crate::context::GpuContext;

/// Collider kind discriminant for a sphere, matching `KIND_SPHERE` in the
/// shader.
const KIND_SPHERE: f32 = 0.0;
/// Collider kind discriminant for a capsule, matching `KIND_CAPSULE` in the
/// shader.
const KIND_CAPSULE: f32 = 1.0;

/// Uniform parameters for one resolve dispatch. Layout matches `Params` in
/// `shaders/strand_collision_resolve.wesl`: the particle and collider counts,
/// padded to one `16`-byte uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    particle_count: u32,
    collider_count: u32,
    pad0: u32,
    pad1: u32,
}

/// One shared analytic collider, uploaded as two `vec4<f32>` (`32`-byte stride).
///
/// For a sphere, `center_a.xyz` is the center and `center_a[3]` the radius, with
/// `b_kind[3] == 0`. For a capsule, `center_a.xyz` is endpoint `a`, `b_kind.xyz`
/// endpoint `b`, `center_a[3]` the radius and `b_kind[3] == 1`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCollider {
    center_a: [f32; 4],
    b_kind: [f32; 4],
}

/// Encodes one golden [`Collider`] into its device [`GpuCollider`] layout.
fn pack_collider(collider: Collider) -> GpuCollider {
    match collider {
        Collider::Sphere { center, radius } => GpuCollider {
            center_a: [center.x, center.y, center.z, radius],
            b_kind: [0.0, 0.0, 0.0, KIND_SPHERE],
        },
        Collider::Capsule { a, b, radius } => GpuCollider {
            center_a: [a.x, a.y, a.z, radius],
            b_kind: [b.x, b.y, b.z, KIND_CAPSULE],
        },
    }
}

/// A compiled, reusable per-particle body-collision resolve pipeline.
pub struct GpuStrandCollisionResolve {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuStrandCollisionResolve {
    /// Compiles the per-particle resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuStrandCollisionResolve {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_strand_collision_resolve"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/strand_collision_resolve.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_strand_collision_resolve_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_strand_collision_resolve_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_strand_collision_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuStrandCollisionResolve {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every free particle out of the shared `colliders`, returning one
    /// `[x, y, z]` position per particle in input order.
    ///
    /// The resolved position for particle `i` matches the `CPU` golden
    /// [`resolve_strand_collisions`](prism_render_architecture::hair::collision::resolve_strand_collisions)
    /// applied to the same inputs within the fma tolerance (`abs_diff < 1e-4` or
    /// `rel_diff < 1e-3`): pinned particles (`inverse_mass <= 0`) are returned
    /// unchanged, free particles are folded through every collider in order, and
    /// the last collider to push wins. An empty particle batch yields an empty
    /// vector without a dispatch; an empty collider set is the golden's no-op, so
    /// each particle's original position is returned without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        particles: &[StrandParticle],
        colliders: &[Collider],
    ) -> Vec<[f32; 3]> {
        let particle_count = particles.len();
        if particle_count == 0 {
            return Vec::new();
        }
        // An empty collider set is the golden's early-return no-op: every
        // particle keeps its original position. Short-circuit so we never bind a
        // zero-sized collider storage buffer.
        if colliders.is_empty() {
            return particles
                .iter()
                .map(|p| [p.position.x, p.position.y, p.position.z])
                .collect();
        }

        let device = ctx.device();

        let uniforms = Params {
            particle_count: particle_count as u32,
            collider_count: colliders.len() as u32,
            pad0: 0,
            pad1: 0,
        };

        let packed: Vec<GpuCollider> = colliders.iter().map(|&c| pack_collider(c)).collect();
        let particle_rows: Vec<[f32; 4]> = particles
            .iter()
            .map(|p| [p.position.x, p.position.y, p.position.z, p.inverse_mass])
            .collect();

        let out_bytes = (particle_count as u64) * 3 * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_collision_resolve_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let colliders_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_collision_resolve_colliders"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let particles_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_strand_collision_resolve_particles"),
            contents: bytemuck::cast_slice(&particle_rows),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_collision_resolve_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_strand_collision_resolve_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_strand_collision_resolve_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: colliders_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: particles_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_strand_collision_resolve_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_strand_collision_resolve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (particle_count as u32).div_ceil(64);
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    }
}

/// The `CPU` golden strand body-collision resolve, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
///
/// Runs [`resolve_strand_collisions`](prism_render_architecture::hair::collision::resolve_strand_collisions)
/// on a copy of `particles` and returns the resolved positions in order.
#[must_use]
pub fn reference_resolve_strand_collisions(
    particles: &[StrandParticle],
    colliders: &[Collider],
) -> Vec<[f32; 3]> {
    let mut working = particles.to_vec();
    resolve_strand_collisions(&mut working, colliders);
    working
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect()
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
