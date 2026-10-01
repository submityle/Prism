//! Real-device `wgpu` compute implementation of the cloth self-collision pass.
//!
//! [`GpuClothSelfCollision`] compiles `shaders/cloth_self_collision.wgsl` once
//! and exposes [`GpuClothSelfCollision::solve`], which runs one parallel-safe
//! Jacobi virtual-particle self-collision pass on the `GPU` and returns the
//! applied positions. The two own-slot passes read only a frozen position
//! snapshot, so the result is order-independent and matches the
//! [`cpu_cloth_self_collision_jacobi`](super::cpu::cpu_cloth_self_collision_jacobi)
//! golden twin within a tight tolerance (`GPU` fused multiply-add and
//! division/sqrt rounding perturb the low bits of the separating push).
//!
//! The uniform hash and the phase-2 incidence list are built on the host by
//! [`super::prep`] in the golden's deterministic order, so the only `GPU`
//! floating-point work is the push arithmetic itself.
//!
//! Provenance: the virtual-particle technique is the published `NvCloth`
//! method; the Jacobi own-slot accumulate/apply split is standard parallel
//! position-based dynamics. No Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use prism_physics_core::VirtualParticle;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BufferBindingType,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource,
};

use crate::buffer;
use crate::context::GpuContext;

use super::layout::{buffer_entry, entry};
use super::prep::{self, ClothPrep};
use super::ClothSelfCollisionScope;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Coincidence guard shared with `prism_physics_core`'s `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Uniform parameters shared with `Params` in
/// `shaders/cloth_self_collision.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of real particles (the leading samples).
    real_count: u32,
    /// Total samples: real particles then in-range virtual particles.
    sample_count: u32,
    /// Number of occupied grid cells.
    key_count: u32,
    /// `0` = resolve every pair; `1` = virtual-touching pairs only.
    virtual_only: u32,
    /// Fabric thickness.
    thickness: f32,
    /// `thickness * thickness`.
    thickness_sq: f32,
    /// Coincidence guard (`EPS_LEN_SQ`).
    eps_len_sq: f32,
    /// Padding to a 16-byte boundary.
    pad0: f32,
}

/// A compiled, reusable `GPU` cloth self-collision pipeline pair.
pub struct GpuClothSelfCollision {
    /// Kept alive so the pipelines it produced stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the two passes share.
    layout: BindGroupLayout,
    /// Phase 1: one invocation per sample accumulates its own half-correction.
    phase1: ComputePipeline,
    /// Phase 2: one invocation per real vertex scatters and applies.
    phase2: ComputePipeline,
}

impl GpuClothSelfCollision {
    /// Compiles the two cloth self-collision kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothSelfCollision {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_self_collision"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/cloth_self_collision.wgsl").into(),
            ),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_self_collision_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, read),
                buffer_entry(2, read),
                buffer_entry(3, read),
                buffer_entry(4, read),
                buffer_entry(5, read),
                buffer_entry(6, read),
                buffer_entry(7, read),
                buffer_entry(8, read),
                buffer_entry(9, write),
                buffer_entry(10, read),
                buffer_entry(11, read),
                buffer_entry(12, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_self_collision_pipeline_layout"),
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
        let phase1 = make("phase1_samples", "prism_cloth_phase1_pipeline");
        let phase2 = make("phase2_apply", "prism_cloth_phase2_pipeline");
        GpuClothSelfCollision {
            module,
            layout,
            phase1,
            phase2,
        }
    }

    /// Runs one Jacobi virtual-particle self-collision pass and returns the
    /// applied positions.
    ///
    /// `scope` selects a self-contained tier
    /// ([`ClothSelfCollisionScope::All`]) or the virtual-only augment mode
    /// ([`ClothSelfCollisionScope::VirtualOnly`]). A non-positive
    /// `cell_size`/`thickness`, a mismatched `inverse_masses` length, or fewer
    /// than two samples, returns `positions` unchanged, matching the
    /// [`cpu_cloth_self_collision_jacobi`](super::cpu::cpu_cloth_self_collision_jacobi)
    /// twin.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        virtuals: &[VirtualParticle],
        cell_size: Real,
        thickness: Real,
        scope: ClothSelfCollisionScope,
    ) -> Vec<Vec3> {
        let Some(p) = prep::build(positions, inverse_masses, virtuals, cell_size, thickness) else {
            return positions.to_vec();
        };
        self.dispatch(ctx, &p, scope)
    }

    /// Uploads a prepared scene, runs both passes, and reads back positions.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        p: &ClothPrep,
        scope: ClothSelfCollisionScope,
    ) -> Vec<Vec3> {
        let device = ctx.device();

        let params = Params {
            real_count: p.real_count,
            sample_count: p.sample_count,
            key_count: p.key_count(),
            virtual_only: match scope {
                ClothSelfCollisionScope::All => 0,
                ClothSelfCollisionScope::VirtualOnly => 1,
            },
            thickness: p.thickness,
            thickness_sq: p.thickness * p.thickness,
            eps_len_sq: EPS_LEN_SQ,
            pad0: 0.0,
        };
        let params_buf = buffer::uniform(device, "prism_cloth_params", &params);

        let positions_buf = buffer::storage_read(device, "prism_cloth_pos", &p.positions);
        let inv_mass_buf = buffer::storage_read(device, "prism_cloth_invmass", &p.inverse_masses);
        let verts_buf = buffer::storage_read(device, "prism_cloth_verts", &p.sample_verts);
        let weights_buf = buffer::storage_read(device, "prism_cloth_weights", &p.sample_weights);
        let cells_buf = buffer::storage_read(device, "prism_cloth_cells", &p.sample_cells);
        let keys_buf = buffer::storage_read(device, "prism_cloth_keys", &p.grid_keys);
        let goff_buf = buffer::storage_read(device, "prism_cloth_grid_off", &p.grid_offsets);
        let gmem_buf = buffer::storage_read(device, "prism_cloth_grid_mem", &p.grid_members);
        let voff_buf = buffer::storage_read(device, "prism_cloth_vert_off", &p.vert_offsets);
        let vent_buf = buffer::storage_read(device, "prism_cloth_vert_ent", &p.vert_entries);

        let dp_bytes = (p.sample_count as u64) * 16;
        let dp_buf = buffer::storage_rw_zeroed(device, "prism_cloth_sample_dp", dp_bytes);
        let out_bytes = (p.real_count as u64) * 16;
        let out_buf = buffer::storage_rw_zeroed(device, "prism_cloth_out", out_bytes);
        let out_stage = buffer::staging(device, "prism_cloth_out_stage", out_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &inv_mass_buf),
                entry(3, &verts_buf),
                entry(4, &weights_buf),
                entry(5, &cells_buf),
                entry(6, &keys_buf),
                entry(7, &goff_buf),
                entry(8, &gmem_buf),
                entry(9, &dp_buf),
                entry(10, &voff_buf),
                entry(11, &vent_buf),
                entry(12, &out_buf),
            ],
        });

        let sample_groups =
            u32::try_from((p.sample_count as usize).div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let vertex_groups =
            u32::try_from((p.real_count as usize).div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_phase1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.phase1);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(sample_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_phase2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.phase2);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(vertex_groups, 1, 1);
        }
        buffer::copy(&mut encoder, &out_buf, &out_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let packed = buffer::read_back::<[f32; 4]>(ctx, &out_stage);
        packed
            .iter()
            .map(|q| Vec3::new(q[0], q[1], q[2]))
            .collect()
    }
}
