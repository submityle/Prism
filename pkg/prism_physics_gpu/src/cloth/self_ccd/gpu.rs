//! Real-device `wgpu` compute implementation of the cloth continuous
//! self-collision (self-CCD) sweep, the faithful twin of [`prism_physics_core`]'s
//! [`resolve_self_ccd_jacobi`](prism_physics_core::resolve_self_ccd_jacobi).
//!
//! [`GpuClothSelfCcd`] compiles `shaders/cloth_self_ccd.wgsl` once and exposes a
//! single [`GpuClothSelfCcd::solve`] that resolves every tunnelling
//! cloth-vs-cloth pair with one parallel-safe (Jacobi) iteration and returns the
//! applied particle positions together with the mutated velocities.
//!
//! # Correctness model
//!
//! The candidate pairs and the per-particle incidence list are built on the host
//! by [`prep::build`](super::prep::build) so they are integer-identical to the
//! golden's broad phase. Only the swept-pair time-of-impact resolution runs on
//! the GPU, split into two own-slot passes that each read a frozen snapshot:
//!
//!   * `phase1_pairs` — one invocation per candidate pair writes its own
//!     per-partner position/velocity deltas, reading no other pair's slot.
//!   * `phase2_apply` — one invocation per particle folds its half of every
//!     incident pair in ascending-pair-index order (the golden's reduction
//!     order), summing deltas and writing `out_positions` / `out_velocities`.
//!
//! Because every pass is own-slot and reads only frozen inputs, the two
//! dispatches reproduce the whole Jacobi pass regardless of invocation order.
//! The only float divergence from the `CPU` is a few `ULP` in `sqrt`/division
//! inside the closed-form TOI; parity is verified within a tight tolerance.
//!
//! # Provenance
//!
//! The closed-form swept-pair TOI resolution is standard analytic
//! continuous-collision geometry, and the Jacobi own-slot accumulate/apply split
//! is standard parallel position-based dynamics. No Unreal Engine source or
//! derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use prism_physics_core::SelfCcdParams;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::prep;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size(64)` in the kernel.
const WORKGROUP: usize = 64;

/// Numerical floor for treating `dt` as zero (velocity recovery disabled),
/// mirroring `EPS_REL_MOTION` / the `inv_dt` guard in [`prism_physics_core`].
const EPS_DT: Real = 1.0e-12;

/// Uniform parameters shared with `Params` in `shaders/cloth_self_ccd.wgsl`
/// (32 bytes / 8 words).
///
/// The field order mirrors the `WGSL` struct exactly; reordering silently
/// corrupts the solve.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of candidate pairs (phase-1 thread count).
    pair_count: u32,
    /// Number of addressable particles (phase-2 thread count).
    particle_count: u32,
    /// Sanitized fabric thickness: the enforced separation.
    thickness: f32,
    /// Sanitized normal restitution in `0..=1`.
    restitution: f32,
    /// `1 / dt`, or `0` when `|dt|` is (near) zero.
    inv_dt: f32,
    /// Padding to a 32-byte (8-word) boundary.
    _pad0: u32,
    /// Padding to a 32-byte (8-word) boundary.
    _pad1: u32,
    /// Padding to a 32-byte (8-word) boundary.
    _pad2: u32,
}

/// A compiled, reusable `GPU` cloth continuous self-collision pipeline pair.
pub struct GpuClothSelfCcd {
    /// Kept alive so the pipelines it produced stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the two passes share.
    layout: BindGroupLayout,
    /// Per-pair swept-pair TOI resolution writing own-slot deltas.
    phase1: ComputePipeline,
    /// Per-particle CSR gather applying the summed corrections.
    phase2: ComputePipeline,
}

impl GpuClothSelfCcd {
    /// Compiles the two cloth self-CCD kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothSelfCcd {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_self_ccd"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_self_ccd.wgsl").into()),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_self_ccd_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, read),
                buffer_entry(2, read),
                buffer_entry(3, read),
                buffer_entry(4, read),
                buffer_entry(5, read),
                buffer_entry(6, read),
                buffer_entry(7, read),
                buffer_entry(8, write),
                buffer_entry(9, write),
                buffer_entry(10, write),
                buffer_entry(11, write),
                buffer_entry(12, write),
                buffer_entry(13, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_self_ccd_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry_point: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let phase1 = make("phase1_pairs", "prism_cloth_self_ccd_phase1_pipeline");
        let phase2 = make("phase2_apply", "prism_cloth_self_ccd_phase2_pipeline");
        GpuClothSelfCcd {
            module,
            layout,
            phase1,
            phase2,
        }
    }

    /// Resolves every tunnelling cloth-vs-cloth pair with one parallel-safe
    /// (Jacobi) continuous self-collision iteration and returns the applied
    /// particle positions together with the mutated velocities.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_self_ccd`](super::cpu::cpu_cloth_self_ccd): both reproduce the
    /// engine's own parallel-safe self-CCD pass.
    ///
    /// Ragged input (`prev_positions`, `velocities`, or `inverse_masses` whose
    /// length differs from `positions`) returns the inputs unchanged, as does
    /// any pass the host broad phase reports as a no-op (disabled sweep,
    /// non-positive thickness, fewer than two particles, or no candidate pair).
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        prev_positions: &[Vec3],
        velocities: &[Vec3],
        inverse_masses: &[Real],
        params: SelfCcdParams,
        dt: Real,
    ) -> (Vec<Vec3>, Vec<Vec3>) {
        let count = positions.len();
        if prev_positions.len() != count
            || velocities.len() != count
            || inverse_masses.len() != count
        {
            return (positions.to_vec(), velocities.to_vec());
        }

        let Some(prep) = prep::build(
            positions,
            prev_positions,
            velocities,
            inverse_masses,
            params,
        ) else {
            return (positions.to_vec(), velocities.to_vec());
        };

        let inv_dt = if dt.abs() <= EPS_DT { 0.0 } else { 1.0 / dt };

        let device = ctx.device();
        let particle_count = prep.particle_count as usize;
        let pair_count = prep.pair_count as usize;
        let pair_bytes = (pair_count as u64) * 16;
        let out_bytes = (particle_count as u64) * 16;

        let uniform = Params {
            pair_count: prep.pair_count,
            particle_count: prep.particle_count,
            thickness: prep.thickness,
            restitution: prep.restitution,
            inv_dt,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = buffer::uniform(device, "prism_cloth_self_ccd_params", &uniform);

        let positions_buf =
            buffer::storage_read(device, "prism_cloth_self_ccd_pos", &prep.positions);
        let prev_buf =
            buffer::storage_read(device, "prism_cloth_self_ccd_prev", &prep.prev_positions);
        let velocities_buf =
            buffer::storage_read(device, "prism_cloth_self_ccd_vel", &prep.velocities);
        let inv_mass_buf =
            buffer::storage_read(device, "prism_cloth_self_ccd_invmass", &prep.inverse_masses);
        let pairs_buf = buffer::storage_read(device, "prism_cloth_self_ccd_pairs", &prep.pairs);
        let voff_buf =
            buffer::storage_read(device, "prism_cloth_self_ccd_vert_off", &prep.vert_offsets);
        let vent_buf =
            buffer::storage_read(device, "prism_cloth_self_ccd_vert_ent", &prep.vert_entries);

        // Phase-1 own-slot outputs; `phase1_pairs` fully overwrites every slot,
        // so zeroed initialisation only guards unreachable lanes.
        let dp_a_buf = buffer::storage_rw_zeroed(device, "prism_cloth_self_ccd_dp_a", pair_bytes);
        let dp_b_buf = buffer::storage_rw_zeroed(device, "prism_cloth_self_ccd_dp_b", pair_bytes);
        let dv_a_buf = buffer::storage_rw_zeroed(device, "prism_cloth_self_ccd_dv_a", pair_bytes);
        let dv_b_buf = buffer::storage_rw_zeroed(device, "prism_cloth_self_ccd_dv_b", pair_bytes);
        let out_pos_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_self_ccd_out_pos", out_bytes);
        let out_vel_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_self_ccd_out_vel", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_self_ccd_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &prev_buf),
                entry(3, &velocities_buf),
                entry(4, &inv_mass_buf),
                entry(5, &pairs_buf),
                entry(6, &voff_buf),
                entry(7, &vent_buf),
                entry(8, &dp_a_buf),
                entry(9, &dp_b_buf),
                entry(10, &dv_a_buf),
                entry(11, &dv_b_buf),
                entry(12, &out_pos_buf),
                entry(13, &out_vel_buf),
            ],
        });

        let pair_groups = u32::try_from(pair_count.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let vertex_groups = u32::try_from(particle_count.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let pos_stage = buffer::staging(device, "prism_cloth_self_ccd_pos_stage", out_bytes);
        let vel_stage = buffer::staging(device, "prism_cloth_self_ccd_vel_stage", out_bytes);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_self_ccd_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_self_ccd_phase1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.phase1);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(pair_groups.max(1), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_self_ccd_phase2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.phase2);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(vertex_groups.max(1), 1, 1);
        }
        buffer::copy(&mut encoder, &out_pos_buf, &pos_stage, out_bytes);
        buffer::copy(&mut encoder, &out_vel_buf, &vel_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let pos_read = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        let vel_read = buffer::read_back::<[f32; 4]>(ctx, &vel_stage);

        // The host broad phase addresses only the first `count` particles; any
        // tail beyond that is left exactly as the caller passed it in.
        let mut out_positions = positions.to_vec();
        let mut out_velocities = velocities.to_vec();
        for i in 0..particle_count {
            let q = pos_read[i];
            out_positions[i] = Vec3::new(q[0], q[1], q[2]);
            let v = vel_read[i];
            out_velocities[i] = Vec3::new(v[0], v[1], v[2]);
        }

        (out_positions, out_velocities)
    }
}
