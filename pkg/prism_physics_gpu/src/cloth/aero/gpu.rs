//! Real-device `wgpu` compute implementation of the two-pass cloth aero kernel.
//!
//! [`GpuClothAero`] compiles the two aero shaders once
//! (`shaders/cloth_aero_force.wgsl` and `shaders/cloth_aero_gather.wgsl`) and
//! exposes [`GpuClothAero::solve`], which runs the race-free Jacobi
//! reformulation of `prism_physics_core`'s sequential `apply_aero_forces`:
//!
//! 1. **Phase 1 (per triangle).** Every retained face computes its force from
//!    the frozen position/velocity snapshot and the host-baked per-face wind,
//!    writing the evenly-shared `force / 3` to a scratch `tri_forces` buffer.
//! 2. **Phase 2 (per vertex).** Every free particle gathers the forces of its
//!    incident faces (ascending by triangle, the golden's reduction order) and
//!    applies the sum as `velocity += gathered * (inverse_mass * dt)`.
//!
//! Each phase is its own compute pass (WebGPU offers no intra-pass storage
//! barrier and phase 2 must observe phase 1's writes), and both passes are
//! encoded into one command buffer and submitted once. The deterministic
//! integer work — triangle filtering, the per-triangle wind, and the per-vertex
//! incidence `CSR` — is done on the host in [`super::prep`], so the only
//! floating-point divergence from the [`cpu_cloth_aero`](super::cpu::cpu_cloth_aero)
//! golden is the per-face drag/lift arithmetic; parity is checked within a
//! tight tolerance rather than bit-for-bit, the same model the rest of the
//! solver uses.
//!
//! Provenance: the per-triangle drag/lift decomposition and the optional
//! quadratic dynamic-pressure term are the standard, publicly documented cloth
//! aerodynamics model. No Unreal Engine source or derived code.

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
use super::{ClothAeroParams, ClothAeroTriangle, Real};

/// Lanes per workgroup; must match `@workgroup_size` in both aero kernels.
const WORKGROUP: usize = 64;

/// Uniform parameters shared with `Params` in the two aero shaders.
///
/// Padded to 32 bytes so the layout is identical in both kernels; the drag/lift
/// coefficients and air density drive phase 1, `dt` drives phase 2, and the two
/// counts gate the per-triangle and per-vertex invocations respectively.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Normal-direction (drag) coefficient.
    drag: f32,
    /// In-plane (lift) coefficient.
    lift: f32,
    /// Air density (`<= 0` selects the linear model).
    air_density: f32,
    /// Substep time (seconds).
    dt: f32,
    /// Number of retained (in-range) triangles.
    triangle_count: u32,
    /// Number of addressable particles.
    vertex_count: u32,
    /// Padding to a 32-byte stride.
    _pad0: u32,
    /// Padding to a 32-byte stride.
    _pad1: u32,
}

/// A compiled, reusable two-pass `GPU` cloth aerodynamics pipeline.
pub struct GpuClothAero {
    /// Kept alive so the force pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    force_module: ShaderModule,
    /// Kept alive so the gather pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    gather_module: ShaderModule,
    /// Bind-group layout for the per-triangle force pass.
    force_layout: BindGroupLayout,
    /// Bind-group layout for the per-vertex gather pass.
    gather_layout: BindGroupLayout,
    /// One invocation per triangle computes its evenly-shared face force.
    force_pipeline: ComputePipeline,
    /// One invocation per vertex gathers its incident faces' forces.
    gather_pipeline: ComputePipeline,
}

impl GpuClothAero {
    /// Compiles both aero kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothAero {
        let device = ctx.device();
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };

        let force_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_aero_force"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_aero_force.wgsl").into()),
        });
        let gather_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_aero_gather"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_aero_gather.wgsl").into()),
        });

        let force_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_aero_force_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, read),
                buffer_entry(2, read),
                buffer_entry(3, read),
                buffer_entry(4, read),
                buffer_entry(5, write),
            ],
        });
        let gather_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_aero_gather_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, write),
                buffer_entry(2, read),
                buffer_entry(3, read),
                buffer_entry(4, read),
                buffer_entry(5, read),
            ],
        });

        let force_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_aero_force_pipeline_layout"),
            bind_group_layouts: &[Some(&force_layout)],
            immediate_size: 0,
        });
        let gather_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_aero_gather_pipeline_layout"),
            bind_group_layouts: &[Some(&gather_layout)],
            immediate_size: 0,
        });

        let force_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cloth_aero_force_pipeline"),
            layout: Some(&force_pipeline_layout),
            module: &force_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let gather_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cloth_aero_gather_pipeline"),
            layout: Some(&gather_pipeline_layout),
            module: &gather_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuClothAero {
            force_module,
            gather_module,
            force_layout,
            gather_layout,
            force_pipeline,
            gather_pipeline,
        }
    }

    /// Runs one two-pass Jacobi aero velocity pre-pass on the `GPU`, returning
    /// the updated velocities.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_aero`](super::cpu::cpu_cloth_aero). The pass is a no-op
    /// (returns `velocities` unchanged) when `dt` is non-positive or non-finite,
    /// when the particle set is empty, when there are no triangles, when every
    /// triangle is out of range, or when the `velocities`/`inverse_masses`
    /// columns are not index-aligned with `positions` — the same guards the
    /// golden applies.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        velocities: &[Vec3],
        inverse_masses: &[Real],
        triangles: &[ClothAeroTriangle],
        params: ClothAeroParams,
        dt: Real,
    ) -> Vec<Vec3> {
        let Some(prep) = prep::build(positions, velocities, inverse_masses, triangles, params, dt)
        else {
            return velocities.to_vec();
        };
        self.dispatch(ctx, &prep, dt)
    }

    /// Uploads a prepared scene, runs both passes, and reads velocities back.
    fn dispatch(&self, ctx: &GpuContext, prep: &prep::ClothAeroPrep, dt: Real) -> Vec<Vec3> {
        let device = ctx.device();

        let params = Params {
            drag: prep.drag,
            lift: prep.lift,
            air_density: prep.air_density,
            dt,
            triangle_count: prep.triangle_count,
            vertex_count: prep.vertex_count,
            _pad0: 0,
            _pad1: 0,
        };

        let vel_bytes = (prep.velocities.len() as u64) * 16;
        let tri_force_bytes = (prep.triangle_count as u64) * 16;

        let params_buf = buffer::uniform(device, "prism_cloth_aero_params", &params);
        let positions_buf = buffer::storage_read(device, "prism_cloth_aero_pos", &prep.positions);
        let velocities_buf =
            buffer::storage_rw_init(device, "prism_cloth_aero_vel", &prep.velocities);
        let triangles_buf = buffer::storage_read(device, "prism_cloth_aero_tris", &prep.triangles);
        let winds_buf =
            buffer::storage_read(device, "prism_cloth_aero_winds", &prep.triangle_winds);
        let tri_forces_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_aero_tri_forces", tri_force_bytes);
        let inv_mass_buf =
            buffer::storage_read(device, "prism_cloth_aero_invmass", &prep.inverse_masses);
        let offsets_buf =
            buffer::storage_read(device, "prism_cloth_aero_vert_offsets", &prep.vert_offsets);
        let vert_tris_buf =
            buffer::storage_read(device, "prism_cloth_aero_vert_tris", &prep.vert_tris);
        let vel_stage = buffer::staging(device, "prism_cloth_aero_stage", vel_bytes);

        let force_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_aero_force_bind"),
            layout: &self.force_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &velocities_buf),
                entry(3, &triangles_buf),
                entry(4, &winds_buf),
                entry(5, &tri_forces_buf),
            ],
        });
        let gather_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_aero_gather_bind"),
            layout: &self.gather_layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &velocities_buf),
                entry(2, &inv_mass_buf),
                entry(3, &tri_forces_buf),
                entry(4, &offsets_buf),
                entry(5, &vert_tris_buf),
            ],
        });

        let force_groups =
            u32::try_from((prep.triangle_count as usize).div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let gather_groups =
            u32::try_from((prep.vertex_count as usize).div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_aero_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_aero_force_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.force_pipeline);
            pass.set_bind_group(0, &force_bind, &[]);
            pass.dispatch_workgroups(force_groups.max(1), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_cloth_aero_gather_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &gather_bind, &[]);
            pass.dispatch_workgroups(gather_groups.max(1), 1, 1);
        }
        buffer::copy(&mut encoder, &velocities_buf, &vel_stage, vel_bytes);
        ctx.queue().submit([encoder.finish()]);

        let read = buffer::read_back::<[f32; 4]>(ctx, &vel_stage);
        read.iter().map(|q| Vec3::new(q[0], q[1], q[2])).collect()
    }
}
