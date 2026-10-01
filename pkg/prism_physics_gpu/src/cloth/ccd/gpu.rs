//! Real-device `wgpu` compute implementation of the cloth continuous-collision
//! sweep, the faithful twin of [`prism_physics_core`]'s [`resolve_ccd`](prism_physics_core::resolve_ccd).
//!
//! [`GpuClothCcd`] compiles `shaders/cloth_ccd.wgsl` once and exposes a single
//! [`GpuClothCcd::solve`] that sweeps every free particle's `prev -> curr`
//! motion against every collider, snaps a hit particle onto the surface plus a
//! skin along the outward normal, reflects its inbound normal velocity by
//! restitution, and damps its tangential slide with position-level Coulomb
//! friction.
//!
//! # Correctness model
//!
//! The pass is *per-particle independent*, so a **single dispatch** reproduces
//! the whole sweep: thread `i` reads the read-only `prev_positions[i]`, walks
//! every collider in slice order (ties broken toward the first, matching the
//! golden's strict `t < best` update), and writes only its own `positions[i]`
//! and `velocities[i]`. There is no cross-particle dependency, so no host-side
//! reduction or sequential body loop is needed (unlike [`super::super::coupling`]).
//! The only float divergence from the `CPU` is a few `ULP` in `sqrt`/division
//! inside the closed-form TOI and projection; parity is verified within a tight
//! tolerance.
//!
//! # Provenance
//!
//! The closed-form swept-primitive TOI solvers are standard analytic
//! continuous-collision geometry, and the tangential-friction projection reuses
//! the Macklin et al. (2014) primitive. No Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use prism_physics_core::{BodyCollider, CcdParams};
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::super::body::collider::pack_body_colliders;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size(64)` in the kernel.
const WORKGROUP: usize = 64;

/// Numerical floor for treating `dt` as zero (velocity reflection disabled),
/// mirroring `EPS_COEF` in [`prism_physics_core`]'s CCD pass.
const EPS_COEF: Real = 1.0e-12;

/// Uniform parameters shared with `Params` in `shaders/cloth_ccd.wgsl`
/// (32 bytes / 8 words).
///
/// The field order mirrors the `WGSL` struct exactly; reordering silently
/// corrupts the solve.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of addressable particles (positions length).
    particle_count: u32,
    /// Number of body colliders in the slice.
    collider_count: u32,
    /// Sanitized skin: distance along the outward normal to place a hit.
    skin: f32,
    /// Sanitized restitution in `0..=1`.
    restitution: f32,
    /// Sanitized Coulomb friction coefficient in `0..=1`.
    mu: f32,
    /// `1 / dt`, or `0` when `|dt|` is (near) zero.
    inv_dt: f32,
    /// Padding to a 16-byte word.
    _pad0: u32,
    /// Padding to a 16-byte word.
    _pad1: u32,
}

/// A compiled, reusable `GPU` cloth continuous-collision sweep pipeline.
pub struct GpuClothCcd {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the pass shares.
    layout: BindGroupLayout,
    /// Per-particle earliest-TOI sweep + surface snap + friction.
    pipeline: ComputePipeline,
}

impl GpuClothCcd {
    /// Compiles the cloth continuous-collision kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothCcd {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_ccd"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_ccd.wgsl").into()),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_ccd_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, write),
                buffer_entry(2, read),
                buffer_entry(3, write),
                buffer_entry(4, read),
                buffer_entry(5, read),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_ccd_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cloth_ccd_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothCcd {
            module,
            layout,
            pipeline,
        }
    }

    /// Sweeps every free particle against every collider and returns the applied
    /// particle positions together with the mutated velocities.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_ccd`](super::cpu::cpu_cloth_ccd): a single dispatch resolves
    /// the whole per-particle-independent sweep.
    ///
    /// A disabled [`CcdParams`], an empty particle or collider list, or ragged
    /// input (`prev_positions`, `velocities`, or `inverse_masses` whose length
    /// differs from `positions`) returns the inputs unchanged. The engine pass
    /// tolerates shorter read-only columns by truncating; this twin instead
    /// requires equal lengths and leaves ragged input untouched, so the parity
    /// suite always compares well-formed, equal-length columns.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        prev_positions: &[Vec3],
        velocities: &[Vec3],
        inverse_masses: &[Real],
        colliders: &[BodyCollider],
        params: CcdParams,
        dt: Real,
        friction: Real,
    ) -> (Vec<Vec3>, Vec<Vec3>) {
        let count = positions.len();
        if !params.enabled
            || colliders.is_empty()
            || count == 0
            || prev_positions.len() != count
            || velocities.len() != count
            || inverse_masses.len() != count
        {
            return (positions.to_vec(), velocities.to_vec());
        }

        // Match the engine's host-side sanitisation exactly so the uniform the
        // kernel reads is identical to what the golden computes internally.
        let params = params.sanitized();
        let mu = if friction.is_finite() {
            friction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let inv_dt = if dt.abs() <= EPS_COEF { 0.0 } else { 1.0 / dt };

        let device = ctx.device();
        let slot_bytes = (count as u64) * 16;

        let packed_pos: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let packed_prev: Vec<[f32; 4]> = prev_positions
            .iter()
            .map(|p| [p.x, p.y, p.z, 0.0])
            .collect();
        let packed_vel: Vec<[f32; 4]> = velocities.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();
        let packed_colliders = pack_body_colliders(colliders);

        let positions_buf = buffer::storage_rw_init(device, "prism_cloth_ccd_pos", &packed_pos);
        let prev_buf = buffer::storage_read(device, "prism_cloth_ccd_prev", &packed_prev);
        let velocities_buf = buffer::storage_rw_init(device, "prism_cloth_ccd_vel", &packed_vel);
        let inv_mass_buf = buffer::storage_read(device, "prism_cloth_ccd_invmass", inverse_masses);
        let colliders_buf =
            buffer::storage_read(device, "prism_cloth_ccd_colliders", &packed_colliders);

        let uniform = Params {
            particle_count: u32::try_from(count).unwrap_or(u32::MAX),
            collider_count: u32::try_from(colliders.len()).unwrap_or(u32::MAX),
            skin: params.skin,
            restitution: params.restitution,
            mu,
            inv_dt,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = buffer::uniform(device, "prism_cloth_ccd_params", &uniform);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_ccd_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &prev_buf),
                entry(3, &velocities_buf),
                entry(4, &inv_mass_buf),
                entry(5, &colliders_buf),
            ],
        });

        let groups = u32::try_from(count.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let pos_stage = buffer::staging(device, "prism_cloth_ccd_pos_stage", slot_bytes);
        let vel_stage = buffer::staging(device, "prism_cloth_ccd_vel_stage", slot_bytes);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_ccd_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_ccd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        buffer::copy(&mut encoder, &positions_buf, &pos_stage, slot_bytes);
        buffer::copy(&mut encoder, &velocities_buf, &vel_stage, slot_bytes);
        ctx.queue().submit([encoder.finish()]);

        let pos_read = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        let vel_read = buffer::read_back::<[f32; 4]>(ctx, &vel_stage);
        let out_positions = pos_read
            .iter()
            .map(|q| Vec3::new(q[0], q[1], q[2]))
            .collect();
        let out_velocities = vel_read
            .iter()
            .map(|q| Vec3::new(q[0], q[1], q[2]))
            .collect();

        (out_positions, out_velocities)
    }
}
