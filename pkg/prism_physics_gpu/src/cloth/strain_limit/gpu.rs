//! Real-device `wgpu` compute implementation of the cloth strain-limit sweep.
//!
//! [`GpuClothStrainLimit`] compiles `shaders/cloth_strain_limit.wgsl` once and
//! exposes [`GpuClothStrainLimit::solve`], which graph-colours the edges,
//! uploads them in colour order, and runs `iterations` colour-batched sweeps on
//! the `GPU` before reading the applied positions back. Each colour class is
//! dispatched as its own compute pass (WebGPU offers no intra-pass storage
//! barrier, and the classes must observe each other's writes), and every pass
//! of every iteration is encoded into a single command buffer and submitted
//! once.
//!
//! Within a colour class the edges touch disjoint particles, so the parallel
//! writes match the sequential golden exactly; the result therefore agrees with
//! the [`cpu_cloth_strain_limit`](super::cpu::cpu_cloth_strain_limit) twin
//! within a tight tolerance (`GPU` fused multiply-add and division/square-root
//! rounding perturb the low bits of each projection). The clamp is stateless,
//! so unlike the compliant kernels there is no Lagrange-multiplier buffer.
//!
//! Provenance: biphasic strain limiting is a standard, publicly documented
//! cloth technique (Provot 1995; Thomaszewski et al. 2009); greedy graph
//! colouring of the constraint conflict graph is standard batched-PBD practice.
//! No Unreal Engine source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::coloring::{colour_strain_limit, StrainLimitColoring};
use super::ClothStrainLimitConstraint;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Uniform parameters shared with `Params` in `shaders/cloth_strain_limit.wgsl`.
///
/// One instance is uploaded per colour class; `start`/`count` select the class
/// in the reordered edge array while `particle_count` is constant across the
/// solve. The clamp needs no `dt` (it is a stateless geometric projection), so
/// the fourth word is explicit padding to keep the uniform 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// First edge index (into the reordered array) of this colour class.
    start: u32,
    /// Number of edges in this colour class.
    count: u32,
    /// Number of addressable particles (min of positions / inverse-mass lengths).
    particle_count: u32,
    /// Padding to a 16-byte uniform; unused by the kernel.
    _pad: u32,
}

/// A compiled, reusable `GPU` cloth strain-limit pipeline.
pub struct GpuClothStrainLimit {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the sweep shares.
    layout: BindGroupLayout,
    /// One invocation per edge applies its biphasic clamp.
    pipeline: ComputePipeline,
}

impl GpuClothStrainLimit {
    /// Compiles the cloth strain-limit kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothStrainLimit {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_strain_limit"),
            source: ShaderSource::Wgsl(
                include_str!("../../shaders/cloth_strain_limit.wgsl").into(),
            ),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_strain_limit_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, write),
                buffer_entry(2, read),
                buffer_entry(3, read),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_strain_limit_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cloth_strain_limit_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothStrainLimit {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs `iterations` colour-batched strain-limit sweeps on the `GPU`,
    /// returning the applied positions.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_strain_limit`](super::cpu::cpu_cloth_strain_limit). An empty
    /// constraint set or zero iterations returns `positions` unchanged;
    /// individual out-of-range edges are skipped on the device exactly as the
    /// golden skips them.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        constraints: &[ClothStrainLimitConstraint],
        iterations: u32,
    ) -> Vec<Vec3> {
        if constraints.is_empty() || iterations == 0 || positions.is_empty() {
            return positions.to_vec();
        }
        let coloring = colour_strain_limit(constraints);
        self.dispatch(
            ctx,
            positions,
            inverse_masses,
            constraints,
            &coloring,
            iterations,
        )
    }

    /// Uploads a coloured scene, runs every pass, and reads back positions.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        constraints: &[ClothStrainLimitConstraint],
        coloring: &StrainLimitColoring,
        iterations: u32,
    ) -> Vec<Vec3> {
        let device = ctx.device();

        let particle_count =
            u32::try_from(positions.len().min(inverse_masses.len())).unwrap_or(u32::MAX);

        // Reorder the edges into colour order so each class occupies the
        // contiguous `[start, start + count)` span the uniform selects.
        let reordered: Vec<ClothStrainLimitConstraint> = coloring
            .order
            .iter()
            .map(|&i| constraints[i as usize])
            .collect();

        let packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let pos_bytes = (packed.len() as u64) * 16;

        let positions_buf =
            buffer::storage_rw_init(device, "prism_cloth_strain_limit_pos", &packed);
        let inv_mass_buf =
            buffer::storage_read(device, "prism_cloth_strain_limit_invmass", inverse_masses);
        let constraints_buf =
            buffer::storage_read(device, "prism_cloth_strain_limit_constraints", &reordered);
        let pos_stage = buffer::staging(device, "prism_cloth_strain_limit_stage", pos_bytes);

        // One uniform buffer and bind group per colour class, reused across all
        // iterations.
        let mut binds: Vec<BindGroup> = Vec::with_capacity(coloring.ranges.len());
        let mut uniforms: Vec<Buffer> = Vec::with_capacity(coloring.ranges.len());
        for &(start, count) in &coloring.ranges {
            let params = Params {
                start,
                count,
                particle_count,
                _pad: 0,
            };
            let params_buf = buffer::uniform(device, "prism_cloth_strain_limit_params", &params);
            let bind = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_cloth_strain_limit_bind"),
                layout: &self.layout,
                entries: &[
                    entry(0, &params_buf),
                    entry(1, &positions_buf),
                    entry(2, &inv_mass_buf),
                    entry(3, &constraints_buf),
                ],
            });
            uniforms.push(params_buf);
            binds.push(bind);
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_strain_limit_encoder"),
        });
        for _ in 0..iterations {
            for (color_idx, &(_, count)) in coloring.ranges.iter().enumerate() {
                let groups =
                    u32::try_from((count as usize).div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
                if groups == 0 {
                    continue;
                }
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism_cloth_strain_limit_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &binds[color_idx], &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
        buffer::copy(&mut encoder, &positions_buf, &pos_stage, pos_bytes);
        ctx.queue().submit([encoder.finish()]);

        let read = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        read.iter().map(|q| Vec3::new(q[0], q[1], q[2])).collect()
    }
}
