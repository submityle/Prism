//! Real-device `wgpu` compute implementation of the closed-mesh cloth pressure
//! (volume-preservation / inflation) constraint.
//!
//! [`GpuClothPressure`] compiles `shaders/cloth_pressure.wgsl` once and exposes
//! [`GpuClothPressure::solve`], which runs the exact compliant-`XPBD` pressure
//! projection of [`prism_physics_core`]'s `project_pressure` on the `GPU`.
//!
//! Pressure is a *single global* constraint coupling every vertex of the shell
//! through the divergence-theorem signed-volume functional, so (unlike the
//! per-edge or per-pair sibling kernels) there is no graph colouring: one solver
//! iteration is six coordinated passes over the whole mesh with one accumulated
//! Lagrange multiplier.
//!
//!   1. `tri_pass`     — per triangle: raw volume term + three corner gradients
//!   2. `vol_reduce`   — tree-reduce the triangle volume terms (× 1/6)
//!   3. `gather_pass`  — per vertex: gather incident corner gradients (`CSR`)
//!   4. `denom_reduce` — tree-reduce the per-vertex denominator terms
//!   5. `lambda_pass`  — single invocation: `delta_lambda` + `lambda` update
//!   6. `apply_pass`   — per vertex: `x_i += w_i * delta_lambda * grad_i`
//!
//! Each pass is its own compute pass so it observes the previous one's writes
//! (`WebGPU` offers no intra-pass storage barrier). All passes of all iterations
//! are encoded into a single command buffer and submitted once. The per-triangle
//! gradient scatter to shared vertices is reformulated as a race-free per-vertex
//! gather through the host-built [`VertexTriangleAdjacency`], so no atomic
//! floating-point accumulation is needed and the result matches the sequential
//! [`cpu_cloth_pressure`](super::cpu::cpu_cloth_pressure) golden within a tight
//! tolerance (`GPU` fused multiply-add and division rounding perturb low bits).
//!
//! # Provenance
//!
//! Signed-volume-via-divergence-theorem pressure, its compliant `XPBD`
//! projection, the `CSR` scatter→gather reformulation, and workgroup tree
//! reduction are all standard, publicly documented techniques. No Unreal Engine
//! source or derived code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor,
    BufferBindingType, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource,
};

use crate::buffer;
use crate::cloth::layout::{buffer_entry, entry};
use crate::context::GpuContext;

use super::adjacency::build_vertex_triangle_adjacency;

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Lanes per element-parallel workgroup; must match `@workgroup_size(64)` in
/// the `tri_pass`, `gather_pass`, and `apply_pass` kernels.
const WORKGROUP: usize = 64;

/// Uniform parameters shared with `Params` in `shaders/cloth_pressure.wgsl`.
///
/// The field order and the two trailing pads mirror the `WGSL` struct exactly
/// (32 bytes / 8 words); reordering any field silently corrupts the solve.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Target enclosed volume `overpressure * rest_volume`.
    target_volume: f32,
    /// `XPBD` compliance (inverse stiffness); 0 is perfectly rigid.
    compliance: f32,
    /// Substep time (seconds).
    dt: f32,
    /// Number of addressable vertices (positions length).
    vertex_count: u32,
    /// Number of triangles in the closed shell.
    triangle_count: u32,
    /// Length of the inverse-mass array (indices at or past it are pinned).
    inv_mass_count: u32,
    /// Padding to the 32-byte `WGSL` struct size.
    _pad0: u32,
    /// Padding to the 32-byte `WGSL` struct size.
    _pad1: u32,
}

/// A compiled, reusable `GPU` cloth-pressure pipeline.
pub struct GpuClothPressure {
    /// Kept alive so the pipelines it produced stay valid.
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    /// The bind-group layout wiring every buffer the six passes share.
    layout: BindGroupLayout,
    /// Pass 1: per-triangle volume term and the three corner gradients.
    tri_pass: ComputePipeline,
    /// Pass 2: tree-reduce the triangle volume terms into `accum[0]`.
    vol_reduce: ComputePipeline,
    /// Pass 3: per-vertex `CSR` gradient gather and denominator term.
    gather_pass: ComputePipeline,
    /// Pass 4: tree-reduce the per-vertex denominator terms into `accum[1]`.
    denom_reduce: ComputePipeline,
    /// Pass 5: single-invocation compliant scalar update.
    lambda_pass: ComputePipeline,
    /// Pass 6: per-vertex position correction.
    apply_pass: ComputePipeline,
}

impl GpuClothPressure {
    /// Compiles the cloth-pressure kernel on `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuClothPressure {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_cloth_pressure"),
            source: ShaderSource::Wgsl(include_str!("../../shaders/cloth_pressure.wgsl").into()),
        });
        let read = BufferBindingType::Storage { read_only: true };
        let write = BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_cloth_pressure_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, write),
                buffer_entry(2, read),
                buffer_entry(3, read),
                buffer_entry(4, read),
                buffer_entry(5, read),
                buffer_entry(6, write),
                buffer_entry(7, write),
                buffer_entry(8, write),
                buffer_entry(9, write),
                buffer_entry(10, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_cloth_pressure_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry_point: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("prism_cloth_pressure_pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let tri_pass = make("tri_pass");
        let vol_reduce = make("vol_reduce");
        let gather_pass = make("gather_pass");
        let denom_reduce = make("denom_reduce");
        let lambda_pass = make("lambda_pass");
        let apply_pass = make("apply_pass");
        GpuClothPressure {
            module,
            layout,
            tri_pass,
            vol_reduce,
            gather_pass,
            denom_reduce,
            lambda_pass,
            apply_pass,
        }
    }

    /// Runs `iterations` compliant pressure projections over the closed shell on
    /// the `GPU`, returning the applied positions.
    ///
    /// This is the real-device twin of
    /// [`cpu_cloth_pressure`](super::cpu::cpu_cloth_pressure): both start the
    /// substep with a zero Lagrange multiplier (the device buffer is
    /// zero-initialised) and carry it across every iteration, driving the
    /// enclosed volume toward `target_volume = overpressure * rest_volume`. An
    /// empty triangle set, zero iterations, an empty position array, or a
    /// non-positive `dt` returns `positions` unchanged; pinned particles
    /// (`inverse_mass <= 0`) never move and out-of-range triangles are skipped,
    /// exactly as in `project_pressure`.
    #[must_use]
    pub fn solve(
        &self,
        ctx: &GpuContext,
        positions: &[Vec3],
        inverse_masses: &[Real],
        triangles: &[[u32; 3]],
        target_volume: Real,
        compliance: Real,
        dt: Real,
        iterations: u32,
    ) -> Vec<Vec3> {
        if triangles.is_empty() || iterations == 0 || positions.is_empty() || dt <= 0.0 {
            return positions.to_vec();
        }

        let device = ctx.device();

        let vertex_count = u32::try_from(positions.len()).unwrap_or(u32::MAX);
        let triangle_count = u32::try_from(triangles.len()).unwrap_or(u32::MAX);
        let inv_mass_count = u32::try_from(inverse_masses.len()).unwrap_or(u32::MAX);

        let params = Params {
            target_volume,
            compliance,
            dt,
            vertex_count,
            triangle_count,
            inv_mass_count,
            _pad0: 0,
            _pad1: 0,
        };

        // Positions as padded `vec4` so the device matches the `vec4` storage
        // array; the `w` lane is preserved on write-back.
        let packed: Vec<[f32; 4]> = positions.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let pos_bytes = (packed.len() as u64) * 16;

        // Triangles as padded `vec4<u32>` (a, b, c, 0).
        let tri_packed: Vec<[u32; 4]> =
            triangles.iter().map(|t| [t[0], t[1], t[2], 0]).collect();

        // Host-built CSR vertex->corner adjacency turns the per-triangle
        // gradient scatter into a race-free per-vertex gather.
        let adjacency = build_vertex_triangle_adjacency(triangles, positions.len());

        // A zero-length storage buffer is invalid; feed a one-element dummy when
        // every corner was dropped (e.g. all triangles out of range) or the
        // inverse-mass array is empty (every vertex pinned via `weight_of`).
        let dummy = [0u32; 1];
        let corners_src: &[u32] = if adjacency.corners.is_empty() {
            &dummy
        } else {
            &adjacency.corners
        };
        let inv_mass_src: &[Real] = if inverse_masses.is_empty() {
            // One zero weight keeps every vertex pinned, which `inv_mass_count`
            // (0) already enforces in `weight_of`.
            const PINNED: [Real; 1] = [0.0];
            &PINNED
        } else {
            inverse_masses
        };

        let params_buf = buffer::uniform(device, "prism_cloth_pressure_params", &params);
        let positions_buf = buffer::storage_rw_init(device, "prism_cloth_pressure_pos", &packed);
        let inv_mass_buf = buffer::storage_read(device, "prism_cloth_pressure_invmass", inv_mass_src);
        let triangles_buf =
            buffer::storage_read(device, "prism_cloth_pressure_tris", &tri_packed);
        let offsets_buf =
            buffer::storage_read(device, "prism_cloth_pressure_adj_offsets", &adjacency.offsets);
        let corners_buf =
            buffer::storage_read(device, "prism_cloth_pressure_adj_corners", corners_src);

        let tri_grads_bytes = (triangles.len() as u64) * 3 * 16;
        let tri_grads_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_pressure_tri_grads", tri_grads_bytes);
        let tri_vol_bytes = (triangles.len() as u64) * 4;
        let tri_vol_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_pressure_tri_vol", tri_vol_bytes);
        let vert_grad_bytes = (positions.len() as u64) * 16;
        let vert_grad_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_pressure_vert_grad", vert_grad_bytes);
        let denom_bytes = (positions.len() as u64) * 4;
        let denom_buf =
            buffer::storage_rw_zeroed(device, "prism_cloth_pressure_denom", denom_bytes);
        // accum: [0]=volume, [1]=denominator, [2]=lambda (persists), [3]=delta.
        let accum_buf = buffer::storage_rw_zeroed(device, "prism_cloth_pressure_accum", 16);

        let pos_stage = buffer::staging(device, "prism_cloth_pressure_stage", pos_bytes);

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_cloth_pressure_bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &params_buf),
                entry(1, &positions_buf),
                entry(2, &inv_mass_buf),
                entry(3, &triangles_buf),
                entry(4, &offsets_buf),
                entry(5, &corners_buf),
                entry(6, &tri_grads_buf),
                entry(7, &tri_vol_buf),
                entry(8, &vert_grad_buf),
                entry(9, &denom_buf),
                entry(10, &accum_buf),
            ],
        });

        let tri_groups = u32::try_from(triangles.len().div_ceil(WORKGROUP)).unwrap_or(u32::MAX);
        let vert_groups = u32::try_from(positions.len().div_ceil(WORKGROUP)).unwrap_or(u32::MAX);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_cloth_pressure_encoder"),
        });
        for _ in 0..iterations {
            self.run_pass(&mut encoder, &self.tri_pass, &bind, tri_groups, "tri");
            self.run_pass(&mut encoder, &self.vol_reduce, &bind, 1, "vol_reduce");
            self.run_pass(&mut encoder, &self.gather_pass, &bind, vert_groups, "gather");
            self.run_pass(&mut encoder, &self.denom_reduce, &bind, 1, "denom_reduce");
            self.run_pass(&mut encoder, &self.lambda_pass, &bind, 1, "lambda");
            self.run_pass(&mut encoder, &self.apply_pass, &bind, vert_groups, "apply");
        }
        buffer::copy(&mut encoder, &positions_buf, &pos_stage, pos_bytes);
        ctx.queue().submit([encoder.finish()]);

        let read = buffer::read_back::<[f32; 4]>(ctx, &pos_stage);
        read.iter().map(|q| Vec3::new(q[0], q[1], q[2])).collect()
    }

    /// Encodes one compute pass dispatching `groups` workgroups of `pipeline`
    /// against the shared bind group.
    fn run_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &ComputePipeline,
        bind: &BindGroup,
        groups: u32,
        label: &str,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups.max(1), 1, 1);
    }
}
