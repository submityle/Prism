//! Real-device `wgpu` compute implementation of the Vertex Block Descent (VBD)
//! solver, the faithful twin of [`prism_physics_core`]'s
//! [`VbdSolver::step_colored`](prism_physics_core::VbdSolver::step_colored).
//!
//! [`GpuVbd`] compiles `shaders/vbd_sweep.wgsl` once and exposes a single
//! [`GpuVbd::solve`] that advances a coloured spring system one full step with
//! all state resident on the `GPU`, returning the advanced positions and
//! velocities.
//!
//! # Correctness model
//!
//! The topology — the per-vertex incident-spring `CSR` list and the colour-major
//! sweep order — is built on the host by [`prep::build`](super::prep::build) so
//! it is integer-identical to the golden. The host records, per substep, one
//! `predict` dispatch, then `iterations x color_count` `sweep_color` dispatches
//! (one per colour, in colour order), then one `recover` dispatch. Because a
//! colour's vertices share no spring, dispatching a colour in parallel
//! reproduces the golden's Gauss-Seidel-across-colours / Jacobi-within-colour
//! schedule. The only divergence from the `CPU` is a few `ULP` in the per-vertex
//! `3x3` solve's division; parity is verified within a tight tolerance.
//!
//! # Provenance
//!
//! VBD follows Chen et al., "Vertex Block Descent" (SIGGRAPH 2024); greedy graph
//! colouring is a standard, publicly documented technique. No Unreal Engine
//! source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use prism_physics_core::vbd::{SpringSet, VbdColoring, VbdConfig};
use prism_physics_core::ParticleStorage;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, Buffer,
    BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::prep;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per workgroup; must match `@workgroup_size(64)` in the kernel.
const WORKGROUP: usize = 64;

/// Global uniform parameters mirroring `Params` in `shaders/vbd_sweep.wgsl`
/// (32 bytes / 8 words). Reordering silently corrupts the solve.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    particle_count: u32,
    color_count: u32,
    h: f32,
    inv_h: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    velocity_scale: f32,
}

/// Per-colour uniform mirroring `ColorParams` in `shaders/vbd_sweep.wgsl`
/// (16 bytes / 4 words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColorParams {
    color_start: u32,
    color_vertex_count: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable `GPU` Vertex Block Descent pipeline set.
pub struct GpuVbd {
    /// Kept alive so the pipelines it produced stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// The shared (group 0) bind-group layout wiring every state buffer.
    layout: BindGroupLayout,
    /// The per-colour (group 1) bind-group layout wiring the colour range.
    color_layout: BindGroupLayout,
    /// Predict pass: snapshot prev, inertial target, warm-start.
    predict: ComputePipeline,
    /// Per-colour relaxation pass.
    sweep: ComputePipeline,
    /// Velocity recovery pass.
    recover: ComputePipeline,
}

impl GpuVbd {
    /// Compiles the three VBD kernels on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVbd {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_vbd_sweep"),
            source: ShaderSource::Wgsl(include_str!("../shaders/vbd_sweep.wgsl").into()),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_vbd_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, write),
                buffer_entry(2, write),
                buffer_entry(3, write),
                buffer_entry(4, write),
                buffer_entry(5, read),
                buffer_entry(6, read),
                buffer_entry(7, read),
                buffer_entry(8, read),
                buffer_entry(9, read),
            ],
        });
        let color_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_vbd_color_layout"),
            entries: &[buffer_entry(0, BufferBindingType::Uniform)],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_vbd_pipeline_layout"),
            bind_group_layouts: &[Some(&layout), Some(&color_layout)],
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
        let predict = make("predict", "prism_vbd_predict_pipeline");
        let sweep = make("sweep_color", "prism_vbd_sweep_pipeline");
        let recover = make("recover", "prism_vbd_recover_pipeline");
        GpuVbd {
            module,
            layout,
            color_layout,
            predict,
            sweep,
            recover,
        }
    }

    /// Advances the coloured spring system one full VBD step and returns the
    /// advanced `(positions, velocities)`.
    ///
    /// This is the real-device twin of [`cpu_vbd`](super::cpu::cpu_vbd): both
    /// reproduce [`prism_physics_core`]'s `VbdSolver::step_colored`. An empty
    /// particle set or a non-positive `dt` returns the caller's input unchanged,
    /// matching the golden's early-out.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        particles: &ParticleStorage,
        springs: &SpringSet,
        config: &VbdConfig,
        coloring: &VbdColoring,
        dt: Real,
    ) -> (Vec<Vec3>, Vec<Vec3>) {
        let base_positions = particles.positions().to_vec();
        let base_velocities = particles.velocities().to_vec();

        let substeps = config.substeps.max(1);
        let h = dt / substeps as Real;
        if particles.is_empty() || dt <= 0.0 || h <= 0.0 {
            return (base_positions, base_velocities);
        }
        let Some(prep) = prep::build(particles, springs, coloring) else {
            return (base_positions, base_velocities);
        };

        let device = ctx.device();
        let particle_count = prep.particle_count as usize;
        let out_bytes = (particle_count as u64) * 16;
        let iterations = config.iterations.max(1);
        let velocity_scale = (1.0 - config.damping * h).max(0.0);

        let params = Params {
            particle_count: prep.particle_count,
            color_count: prep.color_count,
            h,
            inv_h: 1.0 / h,
            gravity_x: config.gravity.x,
            gravity_y: config.gravity.y,
            gravity_z: config.gravity.z,
            velocity_scale,
        };
        let params_buf = buffer::uniform(device, "prism_vbd_params", &params);

        // State buffers resident on the GPU across every substep and sweep.
        let positions_buf = buffer::storage_rw_init(device, "prism_vbd_pos", &prep.positions);
        let velocities_buf = buffer::storage_rw_init(device, "prism_vbd_vel", &prep.velocities);
        let prev_buf = buffer::storage_rw_zeroed(device, "prism_vbd_prev", out_bytes);
        let targets_buf = buffer::storage_rw_zeroed(device, "prism_vbd_targets", out_bytes);

        let inv_mass_buf =
            buffer::storage_read(device, "prism_vbd_invmass", &prep.inverse_masses);
        // Empty read-only storage slices are illegal; pad to one dummy element
        // the kernel's CSR offsets guarantee it never reads.
        let springs_pod = pad_springs(&prep.springs);
        let springs_buf = buffer::storage_read(device, "prism_vbd_springs", &springs_pod);
        let voff_buf = buffer::storage_read(device, "prism_vbd_vert_off", &prep.vert_offsets);
        let vent_pod = pad_u32(&prep.vert_entries);
        let vent_buf = buffer::storage_read(device, "prism_vbd_vert_ent", &vent_pod);
        let order_buf = buffer::storage_read(device, "prism_vbd_order", &prep.order);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_vbd_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &velocities_buf),
                entry(3, &prev_buf),
                entry(4, &targets_buf),
                entry(5, &inv_mass_buf),
                entry(6, &springs_buf),
                entry(7, &voff_buf),
                entry(8, &vent_buf),
                entry(9, &order_buf),
            ],
        });

        // One ColorParams uniform + bind group per colour, reused across every
        // iteration and substep. `predict` / `recover` ignore the colour range
        // but still need group 1 bound, so colour 0's group doubles for them.
        let color_count = prep.color_count as usize;
        let mut color_groups: Vec<BindGroup> = Vec::with_capacity(color_count);
        let mut color_bufs: Vec<Buffer> = Vec::with_capacity(color_count);
        let mut color_vertex_counts: Vec<u32> = Vec::with_capacity(color_count);
        for c in 0..color_count {
            let start = prep.color_offsets[c];
            let end = prep.color_offsets[c + 1];
            let count = end - start;
            color_vertex_counts.push(count);
            let cp = ColorParams {
                color_start: start,
                color_vertex_count: count,
                pad0: 0,
                pad1: 0,
            };
            let buf = buffer::uniform(device, "prism_vbd_color_params", &cp);
            let group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_vbd_color_bind"),
                layout: &self.color_layout,
                entries: &[entry(0, &buf)],
            });
            color_bufs.push(buf);
            color_groups.push(group);
        }

        let particle_groups = u32::try_from(particle_count.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let pos_stage = buffer::staging(device, "prism_vbd_pos_stage", out_bytes);
        let vel_stage = buffer::staging(device, "prism_vbd_vel_stage", out_bytes);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_vbd_encoder"),
        });
        for _ in 0..substeps {
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism_vbd_predict"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.predict);
                pass.set_bind_group(0, &bind, &[]);
                pass.set_bind_group(1, &color_groups[0], &[]);
                pass.dispatch_workgroups(particle_groups.max(1), 1, 1);
            }
            for _ in 0..iterations {
                for c in 0..color_count {
                    let count = color_vertex_counts[c] as usize;
                    if count == 0 {
                        continue;
                    }
                    let groups = u32::try_from(count.div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("prism_vbd_sweep_color"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.sweep);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.set_bind_group(1, &color_groups[c], &[]);
                    pass.dispatch_workgroups(groups.max(1), 1, 1);
                }
            }
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("prism_vbd_recover"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.recover);
                pass.set_bind_group(0, &bind, &[]);
                pass.set_bind_group(1, &color_groups[0], &[]);
                pass.dispatch_workgroups(particle_groups.max(1), 1, 1);
            }
        }
        buffer::copy(&mut encoder, &positions_buf, &pos_stage, out_bytes);
        buffer::copy(&mut encoder, &velocities_buf, &vel_stage, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        let pos_read = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        let vel_read = buffer::read_back::<[f32; 4]>(ctx, &vel_stage);

        let mut out_positions = base_positions;
        let mut out_velocities = base_velocities;
        for i in 0..particle_count {
            let q = pos_read[i];
            out_positions[i] = Vec3::new(q[0], q[1], q[2]);
            let w = vel_read[i];
            out_velocities[i] = Vec3::new(w[0], w[1], w[2]);
        }

        (out_positions, out_velocities)
    }
}

/// Returns `springs` as-is, or a single zeroed dummy when empty, so the
/// read-only storage buffer is never zero-length (which `wgpu` rejects). The
/// kernel's CSR offsets guarantee a dummy is never indexed.
fn pad_springs(springs: &[prep::GpuSpring]) -> Vec<prep::GpuSpring> {
    if springs.is_empty() {
        alloc::vec![prep::GpuSpring {
            a: 0,
            b: 0,
            rest_length: 0.0,
            stiffness: 0.0,
        }]
    } else {
        springs.to_vec()
    }
}

/// Returns `values` as-is, or a single zero when empty, with the same
/// never-indexed guarantee as [`pad_springs`].
fn pad_u32(values: &[u32]) -> Vec<u32> {
    if values.is_empty() {
        alloc::vec![0u32]
    } else {
        values.to_vec()
    }
}
