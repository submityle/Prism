//! Real-device `wgpu` compute implementation of the cloth point (vertex-vertex)
//! self-collision pass, the faithful twin of [`prism_physics_core`]'s
//! [`resolve_self_collision_jacobi`](prism_physics_core::resolve_self_collision_jacobi)
//! and its friction variant.
//!
//! [`GpuClothSelfCollisionPoint`] compiles `shaders/cloth_self_collision_point.wgsl`
//! once and exposes a single [`GpuClothSelfCollisionPoint::solve`] that
//! resolves every penetrating cloth-vs-cloth point pair with one parallel-safe
//! (Jacobi) iteration and returns the applied particle positions.
//!
//! # Correctness model
//!
//! The candidate pairs and the per-particle incidence list are built on the host
//! by [`prep::build`](super::prep::build) so they are integer-identical to the
//! golden's broad phase. Only the separating-push arithmetic runs on the GPU,
//! split into two own-slot passes that each read a frozen snapshot:
//!
//!   * `phase1_pairs` — one invocation per candidate pair writes its own
//!     per-partner position delta, reading no other pair's slot.
//!   * `phase2_apply` — one invocation per particle folds its half of every
//!     incident pair in ascending-pair-index order (the golden's reduction
//!     order), summing deltas and writing `out_positions`.
//!
//! Because every pass is own-slot and reads only frozen inputs, the two
//! dispatches reproduce the whole Jacobi pass regardless of invocation order.
//! The only float divergence from the `CPU` is a few `ULP` in `sqrt`/division;
//! parity is verified within a tight tolerance.
//!
//! # Provenance
//!
//! The inverse-mass-weighted separation is standard position-based dynamics,
//! the tangential-friction projection is the one published by Macklin et al.
//! (2014), and the Jacobi own-slot accumulate/apply split is standard parallel
//! position-based dynamics. No Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
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

/// Uniform parameters shared with `Params` in
/// `shaders/cloth_self_collision_point.wgsl` (16 bytes / 4 words).
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
    /// Sanitized Coulomb friction in `0..=1`; `0` selects the plain push.
    friction: f32,
}

/// A compiled, reusable `GPU` cloth point self-collision pipeline pair.
pub struct GpuClothSelfCollisionPoint {
    /// Kept alive so the pipelines it produced stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the two passes share.
    layout: BindGroupLayout,
    /// Per-pair separating push writing own-slot deltas.
    phase1: ComputePipeline,
    /// Per-particle CSR gather applying the summed corrections.
    phase2: ComputePipeline,
}

impl GpuClothSelfCollisionPoint {
    /// Compiles the two cloth point self-collision kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothSelfCollisionPoint {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_self_collision_point"),
            source: ShaderSource::Wgsl(
                include_str!("../../shaders/cloth_self_collision_point.wgsl").into(),
            ),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_self_collision_point_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, read),
                buffer_entry(2, read),
                buffer_entry(3, read),
                buffer_entry(4, read),
                buffer_entry(5, read),
                buffer_entry(6, read),
                buffer_entry(7, write),
                buffer_entry(8, write),
                buffer_entry(9, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_self_collision_point_pipeline_layout"),
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
        let phase1 = make("phase1_pairs", "prism_cloth_self_collision_point_phase1_pipeline");
        let phase2 = make("phase2_apply", "prism_cloth_self_collision_point_phase2_pipeline");
        GpuClothSelfCollisionPoint {
            module,
            layout,
            phase1,
            phase2,
        }
    }

    /// Resolves every penetrating cloth-vs-cloth point pair with one
    /// parallel-safe (Jacobi) self-collision iteration and returns the applied
    /// particle positions.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_self_collision_point`](super::cpu::cpu_cloth_self_collision_point):
    /// both reproduce the engine's own parallel-safe point self-collision pass.
    ///
    /// A `friction` that sanitizes to `0` takes the plain normal-push branch;
    /// otherwise the kernel consults `prev_positions` for each partner's
    /// frame-start slide (a `prev_positions` slice shorter than `positions` is
    /// padded with the matching end position, matching the golden). Any pass the
    /// host broad phase reports as a no-op (non-positive `cell_size`/`thickness`,
    /// a mismatched `inverse_masses` length, fewer than two particles, or no
    /// candidate pair) returns `positions` unchanged.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        prev_positions: &[Vec3],
        inverse_masses: &[Real],
        cell_size: Real,
        thickness: Real,
        friction: Real,
    ) -> Vec<Vec3> {
        let Some(prep) = prep::build(
            positions,
            prev_positions,
            inverse_masses,
            cell_size,
            thickness,
            friction,
        ) else {
            return positions.to_vec();
        };

        let device = ctx.device();
        let particle_count = prep.particle_count as usize;
        let pair_count = prep.pair_count as usize;
        let pair_bytes = (pair_count as u64) * 16;
        let out_bytes = (particle_count as u64) * 16;

        let uniform = Params {
            pair_count: prep.pair_count,
            particle_count: prep.particle_count,
            thickness: prep.thickness,
            friction: prep.friction,
        };
        let params_buf = buffer::uniform(device, "prism_cloth_self_collision_point_params", &uniform);

        let positions_buf =
            buffer::storage_read(device, "prism_cloth_self_collision_point_pos", &prep.positions);
        let prev_buf = buffer::storage_read(
            device,
            "prism_cloth_self_collision_point_prev",
            &prep.prev_positions,
        );
        let inv_mass_buf = buffer::storage_read(
            device,
            "prism_cloth_self_collision_point_invmass",
            &prep.inverse_masses,
        );
        let pairs_buf =
            buffer::storage_read(device, "prism_cloth_self_collision_point_pairs", &prep.pairs);
        let voff_buf = buffer::storage_read(
            device,
            "prism_cloth_self_collision_point_vert_off",
            &prep.vert_offsets,
        );
        let vent_buf = buffer::storage_read(
            device,
            "prism_cloth_self_collision_point_vert_ent",
            &prep.vert_entries,
        );

        // Phase-1 own-slot outputs; `phase1_pairs` fully overwrites every slot,
        // so zeroed initialisation only guards unreachable lanes.
        let dp_a_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_self_collision_point_dp_a", pair_bytes);
        let dp_b_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_self_collision_point_dp_b", pair_bytes);
        let out_pos_buf = buffer::storage_rw_zeroed(
            device,
            "prism_cloth_self_collision_point_out_pos",
            out_bytes,
        );

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_self_collision_point_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &prev_buf),
                entry(3, &inv_mass_buf),
                entry(4, &pairs_buf),
                entry(5, &voff_buf),
                entry(6, &vent_buf),
                entry(7, &dp_a_buf),
                entry(8, &dp_b_buf),
                entry(9, &out_pos_buf),
            ],
        });

        let pair_groups = u32::try_from(pair_count.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let vertex_groups = u32::try_from(particle_count.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let pos_stage =
            buffer::staging(device, "prism_cloth_self_collision_point_pos_stage", out_bytes);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_self_collision_point_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_self_collision_point_phase1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.phase1);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(pair_groups.max(1), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_self_collision_point_phase2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.phase2);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(vertex_groups.max(1), 1, 1);
        }
        buffer::copy(&mut encoder, &out_pos_buf, &pos_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let pos_read = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);

        // The host broad phase addresses only the first `count` particles; any
        // tail beyond that is left exactly as the caller passed it in.
        let mut out_positions = positions.to_vec();
        for (i, slot) in out_positions.iter_mut().take(particle_count).enumerate() {
            let q = pos_read[i];
            *slot = Vec3::new(q[0], q[1], q[2]);
        }
        out_positions
    }
}
