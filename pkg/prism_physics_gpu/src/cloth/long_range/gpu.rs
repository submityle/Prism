//! Real-device `wgpu` compute implementation of the cloth long-range sweep.
//!
//! [`GpuClothLongRange`] compiles `shaders/cloth_long_range.wgsl` once and
//! exposes [`GpuClothLongRange::solve`], which graph-colours the leashes,
//! uploads them in colour order, and runs `iterations` colour-batched sweeps on
//! the `GPU` before reading the applied positions back. Each colour class is
//! dispatched as its own compute pass (WebGPU offers no intra-pass storage
//! barrier, and the classes must observe each other's writes), and every pass
//! of every iteration is encoded into a single command buffer and submitted
//! once.
//!
//! Within a colour class the leashes touch disjoint particles, so the parallel
//! writes match the sequential golden exactly; the result therefore agrees with
//! the [`cpu_cloth_long_range`](super::cpu::cpu_cloth_long_range) twin within a
//! tight tolerance (`GPU` fused multiply-add and division/square-root rounding
//! perturb the low bits of each projection).
//!
//! Provenance: the one-sided long-range-attachment leash is a published
//! position-based dynamics technique (Kim et al., "Long Range Attachments");
//! greedy graph colouring of the constraint conflict graph is standard
//! batched-PBD practice. No Unreal Engine source or derived code.

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

use super::coloring::{colour_long_range, LongRangeColoring};
use super::ClothLongRangeConstraint;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size` in the kernel.
const WORKGROUP: usize = 64;

/// Uniform parameters shared with `Params` in `shaders/cloth_long_range.wgsl`.
///
/// One instance is uploaded per colour class; `start`/`count` select the class
/// in the reordered leash array while `particle_count`/`dt` are constant across
/// the solve.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// First leash index (into the reordered array) of this colour class.
    start: u32,
    /// Number of leashes in this colour class.
    count: u32,
    /// Number of addressable particles (min of positions / inverse-mass lengths).
    particle_count: u32,
    /// Substep time (seconds).
    dt: f32,
}

/// A compiled, reusable `GPU` cloth long-range pipeline.
pub struct GpuClothLongRange {
    /// Kept alive so the pipeline it produced stays valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the sweep shares.
    layout: BindGroupLayout,
    /// One invocation per leash projects its one-sided compliant correction.
    pipeline: ComputePipeline,
}

impl GpuClothLongRange {
    /// Compiles the cloth long-range kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothLongRange {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_long_range"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_long_range.wgsl").into()),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_long_range_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, write),
                buffer_entry(2, read),
                buffer_entry(3, read),
                buffer_entry(4, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_long_range_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_cloth_long_range_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuClothLongRange {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs `iterations` colour-batched long-range sweeps on the `GPU`,
    /// returning the applied positions.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_long_range`](super::cpu::cpu_cloth_long_range). An empty
    /// constraint set or zero iterations returns `positions` unchanged;
    /// individual out-of-range leashes are skipped on the device exactly as the
    /// golden skips them.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        constraints: &[ClothLongRangeConstraint],
        dt: Real,
        iterations: u32,
    ) -> Vec<Vec3> {
        if constraints.is_empty() || iterations == 0 || positions.is_empty() {
            return positions.to_vec();
        }
        let coloring = colour_long_range(constraints);
        self.dispatch(
            ctx,
            positions,
            inverse_masses,
            constraints,
            &coloring,
            dt,
            iterations,
        )
    }

    /// Uploads a coloured scene, runs every pass, and reads back positions.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        constraints: &[ClothLongRangeConstraint],
        coloring: &LongRangeColoring,
        dt: Real,
        iterations: u32,
    ) -> Vec<Vec3> {
        let device = ctx.device();

        let particle_count =
            u32::try_from(positions.len().min(inverse_masses.len())).unwrap_or(u32::MAX);

        // Reorder the leashes into colour order so each class occupies the
        // contiguous `[start, start + count)` span the uniform selects.
        let reordered: Vec<ClothLongRangeConstraint> = coloring
            .order
            .iter()
            .map(|&i| constraints[i as usize])
            .collect();

        let packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let pos_bytes = (packed.len() as u64) * 16;

        let positions_buf = buffer::storage_rw_init(device, "prism_cloth_long_range_pos", &packed);
        let inv_mass_buf =
            buffer::storage_read(device, "prism_cloth_long_range_invmass", inverse_masses);
        let constraints_buf =
            buffer::storage_read(device, "prism_cloth_long_range_constraints", &reordered);
        let lambda_bytes = (reordered.len() as u64) * 4;
        let lambda_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_long_range_lambda", lambda_bytes);
        let pos_stage = buffer::staging(device, "prism_cloth_long_range_stage", pos_bytes);

        // One uniform buffer and bind group per colour class, reused across all
        // iterations.
        let mut binds: Vec<BindGroup> = Vec::with_capacity(coloring.ranges.len());
        let mut uniforms: Vec<Buffer> = Vec::with_capacity(coloring.ranges.len());
        for &(start, count) in &coloring.ranges {
            let params = Params {
                start,
                count,
                particle_count,
                dt,
            };
            let params_buf = buffer::uniform(device, "prism_cloth_long_range_params", &params);
            let bind = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_cloth_long_range_bind"),
                layout: &self.layout,
                entries: &[
                    entry(0, &params_buf),
                    entry(1, &positions_buf),
                    entry(2, &inv_mass_buf),
                    entry(3, &constraints_buf),
                    entry(4, &lambda_buf),
                ],
            });
            uniforms.push(params_buf);
            binds.push(bind);
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_long_range_encoder"),
        });
        for _ in 0..iterations {
            for (color_idx, &(_, count)) in coloring.ranges.iter().enumerate() {
                let groups =
                    u32::try_from((count as usize).div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
                if groups == 0 {
                    continue;
                }
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism_cloth_long_range_pass"),
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
