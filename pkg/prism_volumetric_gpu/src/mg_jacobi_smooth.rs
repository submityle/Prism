//! `wgpu` compute twin of the single-level weighted-`Jacobi` smoother and its
//! `L2` residual, extracted from the `CPU` golden geometric-multigrid
//! pressure-projection solver
//! ([`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure),
//! design §10, §37).
//!
//! The golden module runs a full `V`-cycle; this twin reproduces only the two
//! per-level primitives that dominate its cost and are embarrassingly parallel:
//! the weighted-`Jacobi` relaxation sweep and the level residual norm. No
//! restriction, prolongation, mean removal, coarse solve, or multi-level
//! recursion is ported — just a fixed number of smoothing sweeps on one
//! [`GridResolution`] followed by the residual of the smoothed field.
//!
//! # Algorithm
//!
//! The level operator is the `7`-point `Laplacian` scaled by `inv_h2`:
//! `A·p = inv_h2·(Σ live neighbors − diagonal·p)`. Solving the center cell of
//! `A·p = rhs` and damping by `omega` gives the relaxation the golden
//! `jacobi_smooth` performs:
//! `p ← (1 − ω)·p + ω·(Σ neighbors − rhs / inv_h2) / diagonal`.
//!
//! One thread owns one voxel. Each smoothing dispatch is one sweep that reads
//! the previous iterate and the constant `rhs` and writes the next iterate; the
//! host ping-pongs two storage buffers so `sweeps` sweeps run back to back,
//! mirroring the golden host loop. The six axis-aligned face neighbors are
//! accumulated in the identical order the scalar reference uses (`+x`, `−x`,
//! `+y`, `−y`, `+z`, `−z`), so the two evaluate the same arithmetic in the same
//! order.
//!
//! After the final sweep a second kernel evaluates the per-cell squared
//! residual `r² = (rhs − inv_h2·(Σ neighbors − diagonal·p))²` of the smoothed
//! field. The host then sums those per-cell squares in ascending linear-index
//! order — exactly the `z`-`y`-`x` nested order the golden `level_residual_l2`
//! visits, since the row-major index increases monotonically through that
//! nest — divides by the voxel count and takes one `sqrt`. Keeping the final
//! order-sensitive reduction on the host (the standard two-level
//! split used by the sibling scan and compaction twins) pins the summation
//! order against the non-associative `f32` add, while the heavy per-cell
//! stencil stays on device.
//!
//! # Boundaries
//!
//! Two wall models are carried as a `u32` classification code in the shared
//! uniform. [`PRESSURE_BOUNDARY_DIRICHLET`] pins the outside pressure to zero so
//! the diagonal keeps all six faces; [`PRESSURE_BOUNDARY_NEUMANN`] mirrors a
//! missing neighbor so the diagonal drops to the live-neighbor count, clamped to
//! one for a lone cell so the sweep is a safe no-op rather than a divide by
//! zero. The kernel encodes this exactly as the reference does: it counts live
//! neighbors and picks the diagonal accordingly.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+ − × ÷` on `f32`, with one `sqrt` reached only on the
//! host — and no transcendental call and no optional device feature, so they
//! run unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! No sweep or residual term contains a transcendental call, so `CPU` and `GPU`
//! evaluate the same closed-form algebra. They are not bit-exact: a `GPU` may
//! fuse a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`, and that perturbation compounds across the
//! iterated sweeps. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a swapped neighbor, a missing diagonal term, a dropped boundary
//! case) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Briggs` multigrid weighted-`Jacobi` smoother and
//! residual norm plus `wgpu` compute dispatch; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::fluid::GridResolution;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Classification code for a homogeneous-`Dirichlet` wall: pressure outside the
/// grid is zero, so the stencil diagonal keeps all six faces.
///
/// Mirrors `PressureBoundary::Dirichlet` of the golden module without
/// re-exporting that enum.
///
/// Provenance: `multigrid_pressure` `Dirichlet` wall convention; no Unreal
/// Engine source or derived code.
pub const PRESSURE_BOUNDARY_DIRICHLET: u32 = 0;

/// Classification code for a homogeneous-`Neumann` wall: a missing neighbor
/// mirrors the center cell, so the stencil diagonal drops to the live-neighbor
/// count (clamped to one for a lone cell).
///
/// Mirrors `PressureBoundary::Neumann` of the golden module without
/// re-exporting that enum.
///
/// Provenance: `multigrid_pressure` `Neumann` wall convention; no Unreal Engine
/// source or derived code.
pub const PRESSURE_BOUNDARY_NEUMANN: u32 = 1;

/// One-dimensional workgroup width; one invocation owns one voxel.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` smoother and residual kernels, embedded inline so
/// the twin ships as a single source file. Both mirror the `CPU` reference
/// exactly; see the module documentation for the algorithm.
const MG_JACOBI_SMOOTH_WGSL: &str = r#"
// Single-level weighted-Jacobi smoother and residual twin. One thread per
// voxel. The `jacobi_sweep` entry performs one damped Jacobi sweep of
// `A*p = rhs`; the `residual_sq` entry writes the per-cell squared residual of
// the smoothed field. Both use only the portable core-WGSL subset (integer
// index math plus + - * / on f32) and take no optional feature, so they run
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Briggs multigrid weighted-Jacobi smoother and residual
// norm; no Unreal Engine source or derived code.

struct Params {
    // Grid extents in voxels along each axis.
    nx: u32,
    ny: u32,
    nz: u32,
    // Wall model: 0 = Dirichlet (diagonal is always six faces),
    // 1 = Neumann (diagonal is the live-neighbor count, clamped to one).
    boundary: u32,
    // Operator scale 1/h^2 of this level.
    inv_h2: f32,
    // Reciprocal 1/inv_h2, precomputed on the host so the GPU consumes the
    // identical f32 the CPU reference formed.
    inv_scale: f32,
    // Weighted-Jacobi damping factor omega.
    omega: f32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The constant right-hand side of the level system.
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
// The previous Jacobi iterate (or, for the residual kernel, the smoothed
// field whose residual is measured).
@group(0) @binding(2) var<storage, read> current: array<f32>;
// The next iterate (jacobi_sweep) or the per-cell squared residual
// (residual_sq).
@group(0) @binding(3) var<storage, read_write> out_field: array<f32>;

// Row-major linear index, matching `GridResolution::linear_index`:
// (z * ny + y) * nx + x.
fn lin(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.ny + y) * params.nx + x;
}

@compute @workgroup_size(64)
fn jacobi_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = params.nx * params.ny * params.nz;
    if (idx >= count) {
        return;
    }
    // Decode the row-major index back into (x, y, z); the inverse of `lin`.
    let x = idx % params.nx;
    let plane = idx / params.nx;
    let y = plane % params.ny;
    let z = plane / params.ny;

    // Accumulate the in-grid face neighbors in the exact order the scalar
    // reference uses, counting how many actually exist.
    var sum = 0.0;
    var live_count = 0u;
    if (x + 1u < params.nx) {
        sum = sum + current[lin(x + 1u, y, z)];
        live_count = live_count + 1u;
    }
    if (x > 0u) {
        sum = sum + current[lin(x - 1u, y, z)];
        live_count = live_count + 1u;
    }
    if (y + 1u < params.ny) {
        sum = sum + current[lin(x, y + 1u, z)];
        live_count = live_count + 1u;
    }
    if (y > 0u) {
        sum = sum + current[lin(x, y - 1u, z)];
        live_count = live_count + 1u;
    }
    if (z + 1u < params.nz) {
        sum = sum + current[lin(x, y, z + 1u)];
        live_count = live_count + 1u;
    }
    if (z > 0u) {
        sum = sum + current[lin(x, y, z - 1u)];
        live_count = live_count + 1u;
    }

    // Dirichlet keeps all six faces; Neumann uses the live-neighbor count,
    // clamped to one for a lone cell so the relaxation never divides by zero.
    var diag = 6.0;
    if (params.boundary == 1u) {
        if (live_count == 0u) {
            diag = 1.0;
        } else {
            diag = f32(live_count);
        }
    }

    let relaxed = (sum - rhs[idx] * params.inv_scale) / diag;
    out_field[idx] = (1.0 - params.omega) * current[idx] + params.omega * relaxed;
}

@compute @workgroup_size(64)
fn residual_sq(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = params.nx * params.ny * params.nz;
    if (idx >= count) {
        return;
    }
    let x = idx % params.nx;
    let plane = idx / params.nx;
    let y = plane % params.ny;
    let z = plane / params.ny;

    var sum = 0.0;
    var live_count = 0u;
    if (x + 1u < params.nx) {
        sum = sum + current[lin(x + 1u, y, z)];
        live_count = live_count + 1u;
    }
    if (x > 0u) {
        sum = sum + current[lin(x - 1u, y, z)];
        live_count = live_count + 1u;
    }
    if (y + 1u < params.ny) {
        sum = sum + current[lin(x, y + 1u, z)];
        live_count = live_count + 1u;
    }
    if (y > 0u) {
        sum = sum + current[lin(x, y - 1u, z)];
        live_count = live_count + 1u;
    }
    if (z + 1u < params.nz) {
        sum = sum + current[lin(x, y, z + 1u)];
        live_count = live_count + 1u;
    }
    if (z > 0u) {
        sum = sum + current[lin(x, y, z - 1u)];
        live_count = live_count + 1u;
    }

    var diag = 6.0;
    if (params.boundary == 1u) {
        if (live_count == 0u) {
            diag = 1.0;
        } else {
            diag = f32(live_count);
        }
    }

    let laplacian = params.inv_h2 * (sum - diag * current[idx]);
    let r = rhs[idx] - laplacian;
    out_field[idx] = r * r;
}
"#;

/// Uniform parameters for one solve. `repr(C)` `std430` layout matching
/// `Params` in [`MG_JACOBI_SMOOTH_WGSL`]: the three grid extents and the
/// boundary code, the operator scale `inv_h2`, its reciprocal `inv_scale`, the
/// damping factor `omega`, then one pad word — `32` bytes with no interior
/// padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Grid extent in voxels along `x`.
    nx: u32,
    /// Grid extent in voxels along `y`.
    ny: u32,
    /// Grid extent in voxels along `z`.
    nz: u32,
    /// Wall-model code ([`PRESSURE_BOUNDARY_DIRICHLET`] or
    /// [`PRESSURE_BOUNDARY_NEUMANN`]).
    boundary: u32,
    /// Operator scale `inv_h2 = 1/h^2` of this level.
    inv_h2: f32,
    /// Reciprocal `1/inv_h2`, precomputed on the host.
    inv_scale: f32,
    /// Weighted-`Jacobi` damping factor `omega`.
    omega: f32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad0: u32,
}

/// One single-level weighted-`Jacobi` smoothing-plus-residual query.
///
/// The field layout is row-major with
/// [`GridResolution::linear_index`](prism_render_architecture::particle::fluid::GridResolution::linear_index);
/// `pressure` and `rhs` must each carry at least `resolution.voxel_count()`
/// samples, and only that prefix is consumed.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgJacobiSmoothQuery {
    /// The initial pressure iterate, one `f32` per voxel.
    pub pressure: Vec<f32>,
    /// The constant right-hand side of the level system, one `f32` per voxel.
    pub rhs: Vec<f32>,
    /// The grid this level is discretized on.
    pub resolution: GridResolution,
    /// Operator scale `inv_h2 = 1/h^2` of this level.
    pub inv_h2: f32,
    /// Weighted-`Jacobi` damping factor `omega` (the golden default is `2/3`).
    pub omega: f32,
    /// Wall-model code ([`PRESSURE_BOUNDARY_DIRICHLET`] or
    /// [`PRESSURE_BOUNDARY_NEUMANN`]).
    pub boundary: u32,
    /// Number of weighted-`Jacobi` sweeps to apply before measuring the
    /// residual.
    pub sweeps: u32,
}

/// The outcome of a [`GpuMgJacobiSmooth::smooth`] solve.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgJacobiSmoothResult {
    /// The smoothed pressure field, row-major, one `f32` per voxel.
    pub pressure: Vec<f32>,
    /// The `L2` residual `‖rhs − A·p‖` of the smoothed field.
    pub residual: f32,
}

/// A compiled, reusable single-level weighted-`Jacobi` smoother pipeline pair.
pub struct GpuMgJacobiSmooth {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    smooth_pipeline: ComputePipeline,
    residual_pipeline: ComputePipeline,
}

impl GpuMgJacobiSmooth {
    /// Compiles the smoother and residual kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMgJacobiSmooth {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth"),
            source: ShaderSource::Wgsl(MG_JACOBI_SMOOTH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let smooth_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("jacobi_sweep"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let residual_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_jacobi_residual_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("residual_sq"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMgJacobiSmooth {
            module,
            layout,
            smooth_pipeline,
            residual_pipeline,
        }
    }

    /// Applies `query.sweeps` weighted-`Jacobi` sweeps to `query.pressure` and
    /// returns the smoothed field together with its `L2` residual.
    ///
    /// The returned field and residual equal the golden single-level
    /// `jacobi_smooth` followed by `level_residual_l2` to within the tolerance
    /// documented on this module. An empty grid, or a `pressure`/`rhs` shorter
    /// than `resolution.voxel_count()`, returns the input pressure unchanged and
    /// a zero residual (no dispatch), matching the reference's degenerate-input
    /// guards. A zero sweep count returns the clamped input field and the
    /// residual of that field.
    #[must_use]
    pub fn smooth(
        &self,
        ctx: &GpuContext,
        query: &GpuMgJacobiSmoothQuery,
    ) -> GpuMgJacobiSmoothResult {
        let count = query.resolution.voxel_count() as usize;
        if count == 0 || query.pressure.len() < count || query.rhs.len() < count {
            return GpuMgJacobiSmoothResult {
                pressure: query.pressure.clone(),
                residual: 0.0,
            };
        }

        let device = ctx.device();

        let inv_scale = 1.0 / query.inv_h2;
        let gpu_params = Params {
            nx: query.resolution.nx,
            ny: query.resolution.ny,
            nz: query.resolution.nz,
            boundary: query.boundary,
            inv_h2: query.inv_h2,
            inv_scale,
            omega: query.omega,
            pad0: 0,
        };

        let pressure = &query.pressure[..count];
        let rhs = &query.rhs[..count];
        let field_bytes = (count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let rhs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_rhs"),
            contents: bytemuck::cast_slice(rhs),
            usage: BufferUsages::STORAGE,
        });
        // Ping-pong iterate buffers. Buffer A starts holding the input pressure
        // so the first sweep reads the same initial iterate the reference seeds.
        let buf_a = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_iterate_a"),
            contents: bytemuck::cast_slice(pressure),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let buf_b = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_iterate_b"),
            size: field_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Per-cell squared residual of the smoothed field.
        let sq_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_residual_sq"),
            size: field_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pressure_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_pressure_stage"),
            size: field_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let residual_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_residual_stage"),
            size: field_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Smoothing bind groups: AB reads A and writes B; BA reads B and writes
        // A. The residual bind group reads the final smoothed buffer and writes
        // the per-cell squares.
        let bind_ab = self.bind_group(device, &params_buf, &rhs_buf, &buf_a, &buf_b);
        let bind_ba = self.bind_group(device, &params_buf, &rhs_buf, &buf_b, &buf_a);

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_encoder"),
        });
        for sweep in 0..query.sweeps {
            let bind = if sweep % 2 == 0 { &bind_ab } else { &bind_ba };
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_jacobi_smooth_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.smooth_pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // Sweep `i` (zero-based) writes B when `i` is even and A when odd, so an
        // odd sweep count leaves the smoothed field in B and an even count (or
        // zero) leaves it in A.
        let final_is_b = query.sweeps % 2 == 1;
        let final_buf = if final_is_b { &buf_b } else { &buf_a };

        // Residual pass: read the smoothed field, write per-cell squares.
        let bind_residual = self.bind_group(device, &params_buf, &rhs_buf, final_buf, &sq_buf);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_jacobi_residual_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.residual_pipeline);
            pass.set_bind_group(0, &bind_residual, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(final_buf, 0, &pressure_stage, 0, field_bytes);
        encoder.copy_buffer_to_buffer(&sq_buf, 0, &residual_stage, 0, field_bytes);
        ctx.queue().submit([encoder.finish()]);

        pressure_stage.slice(..).map_async(MapMode::Read, |_| {});
        residual_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let pressure_view = pressure_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped pressure readback range should be available after poll");
        let smoothed = bytemuck::cast_slice::<u8, f32>(&pressure_view).to_vec();
        drop(pressure_view);
        pressure_stage.unmap();

        let residual_view = residual_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped residual readback range should be available after poll");
        let squares = bytemuck::cast_slice::<u8, f32>(&residual_view).to_vec();
        drop(residual_view);
        residual_stage.unmap();

        debug_assert_eq!(smoothed.len(), count);
        debug_assert_eq!(squares.len(), count);

        // Sum the per-cell squares in ascending linear-index order — identical
        // to the golden z-y-x nest, which visits strictly increasing row-major
        // indices — then divide by the voxel count and take one sqrt. The
        // order-sensitive reduction stays on the host to pin the non-associative
        // f32 add against the reference.
        let mut sum_sq = 0.0f32;
        for &s in &squares {
            sum_sq += s;
        }
        let residual = (sum_sq / count as f32).sqrt();

        GpuMgJacobiSmoothResult {
            pressure: smoothed,
            residual,
        }
    }

    /// Builds a bind group binding the shared uniform and `rhs` alongside the
    /// chosen `current` (read) and `out` (write) buffers.
    fn bind_group(
        &self,
        device: &wgpu::Device,
        params_buf: &wgpu::Buffer,
        rhs_buf: &wgpu::Buffer,
        current: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_jacobi_smooth_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: rhs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: current.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
            ],
        })
    }
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
