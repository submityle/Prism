//! Real-device `wgpu` compute implementation of two-way particle/rigid
//! coupling, the faithful twin of [`prism_physics_core`]'s
//! `resolve_two_way_coupling`.
//!
//! [`GpuClothCoupling`] compiles `shaders/cloth_coupling.wgsl` once and exposes
//! a single [`GpuClothCoupling::solve`] that pushes particles out of each rigid
//! proxy, splits every push-out between the particle and the body by inverse
//! mass, and accumulates the Newton reaction impulse the body receives.
//!
//! # Correctness model
//!
//! Bodies are processed in slice order: the host drives **one dispatch per
//! body**, and the shared `positions` buffer persists across dispatches, so
//! each body sees the particle positions after the previous body's corrections
//! — exactly the sequential golden. Within a dispatch the pass is Jacobi and
//! *per-particle independent*: thread `i` writes its own mass-weighted share of
//! the push-out back to `positions[i]` and emits its `body_delta` / `impulse`
//! contributions to per-particle output slots. The host then sums those slots
//! **in index order** (matching the golden's accumulation order), translates
//! the body once, and accumulates its reaction impulse. Keeping the per-body
//! reduction on the host preserves the golden's exact summation order, so the
//! only float divergence is a few `ULP` in `inverseSqrt`/division inside the
//! projection; parity is verified within a tight relative tolerance.
//!
//! # Provenance
//!
//! The inverse-mass-weighted contact split and the Newton reaction impulse are
//! textbook position-based-dynamics / rigid-body contact mechanics. No Unreal
//! Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use prism_physics_core::{BodyCollider, CouplingBody};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::super::body::collider::{pack_convex_planes, GpuBodyCollider, GpuConvexPlane};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size(64)` in the kernel.
const WORKGROUP: usize = 64;

/// Minimum squared push-out length treated as a real contact.
///
/// Mirrors `EPS_LEN_SQ` in both the kernel and [`prism_physics_core`]; a body
/// translation below this threshold is ignored (the golden's Jacobi gate).
const EPS_LEN_SQ: Real = 1.0e-12;

/// Uniform parameters shared with `Params` in `shaders/cloth_coupling.wgsl`
/// (16 bytes / 4 words plus the 64-byte collider record).
///
/// The field order mirrors the `WGSL` struct exactly; reordering silently
/// corrupts the solve.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// The body's current pose for this dispatch (64 bytes, 16-byte aligned).
    collider: GpuBodyCollider,
    /// Number of addressable particles (positions length).
    particle_count: u32,
    /// The body's (already non-negative) inverse mass.
    w_body: f32,
    /// The substep.
    dt: f32,
    /// Padding to a 16-byte word.
    _pad: u32,
}

/// A compiled, reusable `GPU` cloth two-way coupling pipeline.
pub struct GpuClothCoupling {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the pass shares.
    layout: BindGroupLayout,
    /// Per-particle mass-weighted push-out + body/impulse contribution.
    pipeline: ComputePipeline,
}

impl GpuClothCoupling {
    /// Compiles the cloth two-way coupling kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothCoupling {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_coupling"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_coupling.wgsl").into()),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_coupling_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, write),
                buffer_entry(2, read),
                buffer_entry(3, write),
                buffer_entry(4, write),
                buffer_entry(6, read),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_coupling_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cloth_coupling_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothCoupling {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves two-way particle/rigid contact for every particle against every
    /// body and returns the applied particle positions together with the
    /// mutated bodies (translated colliders and accumulated reaction impulses).
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_coupling`](super::cpu::cpu_cloth_coupling): bodies are
    /// processed in slice order, each seeing the particle positions after the
    /// previous body's corrections.
    ///
    /// An empty particle or body list, or an `inverse_masses` slice whose length
    /// differs from `positions`, returns the inputs unchanged. (The engine
    /// pass tolerates a shorter `inverse_masses` by truncating; this twin
    /// instead requires equal lengths and leaves ragged input untouched, so the
    /// parity suite always compares well-formed, equal-length columns.)
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        bodies: &[CouplingBody],
        dt: Real,
    ) -> (Vec<Vec3>, Vec<CouplingBody>) {
        if positions.is_empty()
            || bodies.is_empty()
            || inverse_masses.len() != positions.len()
            || dt <= 0.0
        {
            return (positions.to_vec(), bodies.to_vec());
        }

        let device = ctx.device();
        let count = positions.len();
        let slot_bytes = (count as u64) * 16;

        // Shared, persistent position buffer: each body's dispatch reads the
        // corrections left by the previous body (sequential golden).
        let packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let positions_buf = buffer::storage_rw_init(device, "prism_cloth_coupling_pos", &packed);
        let inv_mass_buf =
            buffer::storage_read(device, "prism_cloth_coupling_invmass", inverse_masses);
        // Per-particle contribution slots, fully overwritten by every dispatch.
        let body_delta_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_coupling_bodydelta", slot_bytes);
        let impulse_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_coupling_impulse", slot_bytes);

        let groups = u32::try_from(count.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut out_bodies = bodies.to_vec();
        for body in &mut out_bodies {
            let collider = body.collider;
            let w_body = body.inverse_mass.max(0.0);
            let params = Params {
                collider: GpuBodyCollider::from_collider(collider),
                particle_count: u32::try_from(count).unwrap_or(u32::MAX),
                w_body,
                dt,
                _pad: 0,
            };
            let params_buf = buffer::uniform(device, "prism_cloth_coupling_params", &params);

            // The convex-hull arm reads its face planes from a storage pool;
            // this dispatch handles exactly one body, so the planes live at
            // offset 0 (an empty 1-element dummy keeps the binding legal for
            // the analytic primitives, which never index it).
            let convex_planes = pack_convex_planes(&[collider]);
            let convex_dummy = [GpuConvexPlane::zeroed()];
            let convex_slice: &[GpuConvexPlane] = if convex_planes.is_empty() {
                &convex_dummy
            } else {
                &convex_planes
            };
            let convex_buf =
                buffer::storage_read(device, "prism_cloth_coupling_convex", convex_slice);

            let bind = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_cloth_coupling_bind"),
                layout: &self.layout,
                entries: &[
                    entry(0, &params_buf),
                    entry(1, &positions_buf),
                    entry(2, &inv_mass_buf),
                    entry(3, &body_delta_buf),
                    entry(4, &impulse_buf),
                    entry(6, &convex_buf),
                ],
            });

            let bd_stage = buffer::staging(device, "prism_cloth_coupling_bd_stage", slot_bytes);
            let im_stage = buffer::staging(device, "prism_cloth_coupling_im_stage", slot_bytes);

            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("prism_cloth_coupling_encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism_cloth_coupling_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups.max(1), 1, 1);
            }
            buffer::copy(&mut encoder, &body_delta_buf, &bd_stage, slot_bytes);
            buffer::copy(&mut encoder, &impulse_buf, &im_stage, slot_bytes);
            ctx.queue().submit([encoder.finish()]);

            // Sum the per-particle contributions in index order, matching the
            // golden's sequential accumulation.
            let bd_slots = buffer::read_back::<[f32; 4]>(ctx, &bd_stage);
            let im_slots = buffer::read_back::<[f32; 4]>(ctx, &im_stage);
            let mut bd_sum = Vec3::ZERO;
            let mut im_sum = Vec3::ZERO;
            for slot in &bd_slots {
                bd_sum += Vec3::new(slot[0], slot[1], slot[2]);
            }
            for slot in &im_slots {
                im_sum += Vec3::new(slot[0], slot[1], slot[2]);
            }

            if w_body > 0.0 && bd_sum.length_squared() > EPS_LEN_SQ {
                body.collider = translate_collider(collider, bd_sum);
            }
            body.reaction_impulse += im_sum;
        }

        // Read the final particle positions after every body's corrections.
        let pos_stage = buffer::staging(device, "prism_cloth_coupling_pos_stage", slot_bytes);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_coupling_pos_encoder"),
        });
        buffer::copy(&mut encoder, &positions_buf, &pos_stage, slot_bytes);
        ctx.queue().submit([encoder.finish()]);
        let read = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        let out_positions = read.iter().map(|q| Vec3::new(q[0], q[1], q[2])).collect();

        (out_positions, out_bodies)
    }
}

/// Returns `collider` translated by `delta`.
///
/// Mirrors the private `translate_collider` in
/// [`prism_physics_core`]'s coupling pass: spheres, capsules, and oriented
/// boxes move their centers/endpoints rigidly (an `OBB` keeps its orientation
/// and half-extents), and a half-space shifts its `offset` along the
/// (unchanged) normal by `normal.dot(delta)`.
fn translate_collider(collider: BodyCollider, delta: Vec3) -> BodyCollider {
    match collider {
        BodyCollider::Sphere { center, radius } => BodyCollider::Sphere {
            center: center + delta,
            radius,
        },
        BodyCollider::Capsule { p0, p1, radius } => BodyCollider::Capsule {
            p0: p0 + delta,
            p1: p1 + delta,
            radius,
        },
        BodyCollider::HalfSpace { normal, offset } => BodyCollider::HalfSpace {
            normal,
            offset: offset + normal.dot(delta),
        },
        BodyCollider::Obb {
            center,
            orientation,
            half_extents,
        } => BodyCollider::Obb {
            center: center + delta,
            orientation,
            half_extents,
        },
        BodyCollider::ConvexHull(proxy) => BodyCollider::ConvexHull(proxy.translated(delta)),
    }
}
