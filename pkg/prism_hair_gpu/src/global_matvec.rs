//! `wgpu` compute twin of Prism's projective-dynamics global dense
//! matrix-vector product
//! ([`GlobalSystem::mul`](prism_render_architecture::hair::projective_global::GlobalSystem::mul)).
//!
//! Projective Dynamics (`Bouaziz` 2014, the family behind `TressFX` / UE5
//! Groom-class global hair solves) couples every per-constraint local
//! projection through a single global symmetric positive-definite (`SPD`)
//! system `(M/h^2 + sum_i w_i S_i^T S_i) x = rhs`. Because the matrix `A` is
//! constant for a fixed constraint graph, it is assembled and factored once; the
//! hot per-iteration primitive of the conjugate-gradient / global-solve path is
//! then the dense product `A x`. The `CPU` golden
//! [`GlobalSystem::mul`](prism_render_architecture::hair::projective_global::GlobalSystem::mul)
//! computes, for each row `i`, `out[i] = sum_j A[i*n+j] * x[j]` independently
//! for the three coordinate components. This crate is the on-device twin of
//! exactly that product, one thread per matrix row, so a passing real-device
//! parity test is direct evidence the ported kernel computes the same matvec as
//! the reference - not merely that its shader compiles.
//!
//! # Relationship to the sibling edge-projection twin
//!
//! This is deliberately a different dispatch from
//! [`GpuProjectiveEdge`](crate::projective_edge::GpuProjectiveEdge), which
//! reproduces the *local* edge projection
//! ([`local_project_edge`](prism_render_architecture::hair::projective_global::local_project_edge)) -
//! a sparse per-edge position projection with one thread per edge. This twin is
//! the complementary *global* step: a dense `n x n` matrix-vector product with
//! one thread per row. The two are the disjoint halves of the projective
//! dynamics local/global split; no other twin computes a dense global matvec.
//! The companion direct and conjugate-gradient solvers
//! ([`GlobalSystem::solve`](prism_render_architecture::hair::projective_global::GlobalSystem::solve),
//! [`GlobalSystem::solve_cg`](prism_render_architecture::hair::projective_global::GlobalSystem::solve_cg))
//! are intrinsically serial reductions and are deliberately *not* mirrored by a
//! one-thread-per-element twin.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairGlobalMatvec::eval`] takes the row-major `n * n` matrix and an input
//! vector of per-particle `[f32; 3]` and returns the `n`-entry product `A x`.
//! The row index is the invocation id (`@compute @workgroup_size(64)`, a
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past
//! `n` early-return. Each row loops over all `n` columns accumulating the three
//! coordinate components in ascending column order, matching the golden's fixed
//! accumulation order. The golden pads a short `x` with zeros; the host mirrors
//! that by uploading `x` padded to `n` particles.
//!
//! # Correctness model
//!
//! The product is a sum of exact products in a fixed order, so the only `CPU`
//! vs `GPU` divergence is legal fused-multiply-add contraction in the row
//! accumulation. Each component is matched against a tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`), not bit-for-bit.
//!
//! # Portability
//!
//! The kernel uses only multiply and add in the portable core-`WGSL` subset -
//! no `exp`, `pow` or optional device feature - so the twin runs unmodified on
//! Metal, Vulkan and DX12.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard dense matrix-vector product plus a `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one matvec dispatch. Layout matches `Params` in
/// `shaders/global_matvec.wesl`: the dimension `n`, padded to one `16`-byte
/// uniform slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable dense global matrix-vector pipeline.
pub struct GpuHairGlobalMatvec {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairGlobalMatvec {
    /// Compiles the matvec shader and builds its pipeline against `ctx`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_global_matvec"),
            source: ShaderSource::Wgsl(include_str!("../shaders/global_matvec.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_global_matvec_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_global_matvec_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_global_matvec_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairGlobalMatvec {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes the dense product `A x` for a row-major `n * n` matrix and an
    /// input vector of per-particle `[f32; 3]`, returning the `n`-entry result.
    ///
    /// The result equals
    /// [`GlobalSystem::mul`](prism_render_architecture::hair::projective_global::GlobalSystem::mul)
    /// on the same matrix and vector, to within the fused-multiply-add tolerance
    /// documented on this module (`abs_diff < 1e-4` or `rel_diff < 1e-3`). An
    /// `x` shorter than `n` is padded with zeros, mirroring the golden; an `x`
    /// longer than `n` is truncated to `n`. A zero-dimension system (`n == 0`)
    /// returns an empty vector without a dispatch, since storage buffers cannot
    /// be zero-sized.
    ///
    /// # Panics
    ///
    /// Panics if `matrix.len() < n * n`, matching the golden's row-major
    /// indexing contract.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        matrix: &[f32],
        n: usize,
        x: &[[f32; 3]],
    ) -> Vec<[f32; 3]> {
        if n == 0 {
            return Vec::new();
        }
        assert!(
            matrix.len() >= n * n,
            "matrix must hold at least n*n entries (n = {n}, got {})",
            matrix.len()
        );

        // Upload exactly the n*n matrix the golden indexes, and x padded (or
        // truncated) to n particles so the device sees the same zero padding the
        // golden applies to a short slice.
        let matrix_slice = &matrix[..n * n];
        let mut x_flat: Vec<f32> = Vec::with_capacity(n * 3);
        for p in x.iter().take(n) {
            x_flat.extend_from_slice(p);
        }
        x_flat.resize(n * 3, 0.0);

        let device = ctx.device();
        let params = Params {
            n: n as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // n groups of three f32 out.
        let out_len = n * 3;
        let out_bytes = (out_len as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_global_matvec_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let matrix_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_global_matvec_matrix"),
            contents: bytemuck::cast_slice(matrix_slice),
            usage: BufferUsages::STORAGE,
        });
        let x_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_global_matvec_x"),
            contents: bytemuck::cast_slice(&x_flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_global_matvec_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_global_matvec_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_global_matvec_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: matrix_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: x_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_global_matvec_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_global_matvec_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (n as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();
        debug_assert_eq!(flat.len(), out_len);

        // Re-group the flat stream into per-particle [f32; 3].
        (0..n)
            .map(|i| {
                let b = i * 3;
                [flat[b], flat[b + 1], flat[b + 2]]
            })
            .collect()
    }
}

/// The `CPU` golden dense global matrix-vector product, provided so the parity
/// test can assert the device twin against an identical same-order reference in
/// addition to the architecture-side
/// [`GlobalSystem::mul`](prism_render_architecture::hair::projective_global::GlobalSystem::mul).
///
/// Computes, for each row `i`, `out[i] = sum_j matrix[i*n+j] * x[j]`
/// independently for the three coordinate components, accumulating in ascending
/// column order. An `x` shorter than `n` is treated as zero-padded, and an `x`
/// longer than `n` is truncated to `n`, exactly as the golden does.
///
/// # Panics
///
/// Panics if `matrix.len() < n * n`.
#[must_use]
pub fn reference_global_matvec(matrix: &[f32], n: usize, x: &[[f32; 3]]) -> Vec<[f32; 3]> {
    assert!(
        matrix.len() >= n * n,
        "matrix must hold at least n*n entries (n = {n}, got {})",
        matrix.len()
    );
    (0..n)
        .map(|i| {
            let row = i * n;
            let mut s = [0.0f32; 3];
            for j in 0..n {
                let a = matrix[row + j];
                let xj = x.get(j).copied().unwrap_or([0.0, 0.0, 0.0]);
                s[0] += a * xj[0];
                s[1] += a * xj[1];
                s[2] += a * xj[2];
            }
            s
        })
        .collect()
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
