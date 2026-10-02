//! `wgpu` compute twin of the single-level `L2` residual norm, extracted from
//! the `CPU` golden geometric-multigrid pressure-projection solver
//! ([`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure),
//! design §10, §37).
//!
//! The golden module runs a full `V`-cycle; this twin reproduces only the
//! order-sensitive residual-norm primitive its convergence test reads every
//! cycle. No smoothing sweep, restriction, prolongation, mean removal, coarse
//! solve, or multi-level recursion is ported — just the level residual
//! `‖b − A·p‖` of one pressure field on one [`GridResolution`].
//!
//! # Algorithm
//!
//! The level operator is the `7`-point `Laplacian` scaled by `inv_h2`:
//! `A·p = inv_h2·(Σ live neighbors − diagonal·p)`. For each voxel the twin
//! evaluates the per-cell squared residual `r² = (b − A·p)²` of the field,
//! mirroring the golden `level_residual_l2`:
//! `r = rhs − inv_h2·(Σ neighbors − diagonal·p)`.
//!
//! One thread owns one voxel: a single compute dispatch reads the pressure and
//! the constant `rhs` and writes one squared residual per cell. The six
//! axis-aligned face neighbors are accumulated in the identical order the
//! scalar reference uses (`+x`, `−x`, `+y`, `−y`, `+z`, `−z`), so the two
//! evaluate the same arithmetic in the same order.
//!
//! The host then sums those per-cell squares in ascending linear-index order —
//! exactly the `z`-`y`-`x` nested order the golden `level_residual_l2` visits,
//! since the row-major index increases monotonically through that nest —
//! divides by the voxel count and takes one `sqrt`. Keeping the final
//! order-sensitive reduction on the host (the standard two-level split used by
//! the sibling scan, compaction and smoother twins) pins the summation order
//! against the non-associative `f32` add, while the heavy per-cell stencil
//! stays on device.
//!
//! # Boundaries
//!
//! Two wall models are carried as a `u32` classification code in the shared
//! uniform. [`PRESSURE_BOUNDARY_DIRICHLET`] pins the outside pressure to zero so
//! the diagonal keeps all six faces; [`PRESSURE_BOUNDARY_NEUMANN`] mirrors a
//! missing neighbor so the diagonal drops to the live-neighbor count, clamped to
//! one for a lone cell so the operator is a safe no-op rather than a divide by
//! zero. The kernel encodes this exactly as the reference does: it counts live
//! neighbors and picks the diagonal accordingly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+ − × ÷` on `f32`, with one `sqrt` reached only on the
//! host — and no transcendental call and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! No residual term contains a transcendental call, so `CPU` and `GPU` evaluate
//! the same closed-form algebra. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a swapped neighbor, a missing diagonal term, a dropped boundary
//! case) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Briggs` multigrid level residual norm plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

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
/// re-exporting that enum, and matches the sibling
/// [`mg_jacobi_smooth`](crate::mg_jacobi_smooth) code of the same name.
///
/// Provenance: `multigrid_pressure` `Dirichlet` wall convention; no Unreal
/// Engine source or derived code.
pub const PRESSURE_BOUNDARY_DIRICHLET: u32 = 0;

/// Classification code for a homogeneous-`Neumann` wall: a missing neighbor
/// mirrors the center cell, so the stencil diagonal drops to the live-neighbor
/// count (clamped to one for a lone cell).
///
/// Mirrors `PressureBoundary::Neumann` of the golden module without
/// re-exporting that enum, and matches the sibling
/// [`mg_jacobi_smooth`](crate::mg_jacobi_smooth) code of the same name.
///
/// Provenance: `multigrid_pressure` `Neumann` wall convention; no Unreal Engine
/// source or derived code.
pub const PRESSURE_BOUNDARY_NEUMANN: u32 = 1;

/// One-dimensional workgroup width; one invocation owns one voxel.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` residual kernel, embedded inline so the twin ships
/// as a single source file. It mirrors the `CPU` reference exactly; see the
/// module documentation for the algorithm.
const MG_LEVEL_RESIDUAL_WGSL: &str = r#"
// Single-level L2 residual twin. One thread per voxel. The `residual_sq` entry
// writes the per-cell squared residual `(rhs - A*p)^2` of the pressure field,
// where `A*p = inv_h2*(sum live neighbors - diagonal*p)`. It uses only the
// portable core-WGSL subset (integer index math plus + - * / on f32); the one
// sqrt of the norm is taken on the host. No optional feature is used, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Briggs multigrid level residual norm; no Unreal Engine
// source or derived code.

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
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The constant right-hand side of the level system.
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
// The pressure field whose residual is measured.
@group(0) @binding(2) var<storage, read> pressure: array<f32>;
// The per-cell squared residual.
@group(0) @binding(3) var<storage, read_write> out_sq: array<f32>;

// Row-major linear index, matching `GridResolution::linear_index`:
// (z * ny + y) * nx + x.
fn lin(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.ny + y) * params.nx + x;
}

@compute @workgroup_size(64)
fn residual_sq(@builtin(global_invocation_id) gid: vec3<u32>) {
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
        sum = sum + pressure[lin(x + 1u, y, z)];
        live_count = live_count + 1u;
    }
    if (x > 0u) {
        sum = sum + pressure[lin(x - 1u, y, z)];
        live_count = live_count + 1u;
    }
    if (y + 1u < params.ny) {
        sum = sum + pressure[lin(x, y + 1u, z)];
        live_count = live_count + 1u;
    }
    if (y > 0u) {
        sum = sum + pressure[lin(x, y - 1u, z)];
        live_count = live_count + 1u;
    }
    if (z + 1u < params.nz) {
        sum = sum + pressure[lin(x, y, z + 1u)];
        live_count = live_count + 1u;
    }
    if (z > 0u) {
        sum = sum + pressure[lin(x, y, z - 1u)];
        live_count = live_count + 1u;
    }

    // Dirichlet keeps all six faces; Neumann uses the live-neighbor count,
    // clamped to one for a lone cell so the operator never divides by zero.
    var diag = 6.0;
    if (params.boundary == 1u) {
        if (live_count == 0u) {
            diag = 1.0;
        } else {
            diag = f32(live_count);
        }
    }

    let laplacian = params.inv_h2 * (sum - diag * pressure[idx]);
    let r = rhs[idx] - laplacian;
    out_sq[idx] = r * r;
}
"#;

/// Uniform parameters for one solve. `repr(C)` `std430` layout matching
/// `Params` in [`MG_LEVEL_RESIDUAL_WGSL`]: the three grid extents and the
/// boundary code, the operator scale `inv_h2`, then three pad words — `32`
/// bytes with no interior padding.
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
    /// Padding to keep the struct a multiple of `16` bytes.
    pad0: u32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad1: u32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad2: u32,
}

/// One single-level `L2` residual query.
///
/// The field layout is row-major with
/// [`GridResolution::linear_index`](prism_render_architecture::particle::fluid::GridResolution::linear_index);
/// `pressure` and `rhs` must each carry at least `resolution.voxel_count()`
/// samples, and only that prefix is consumed.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgLevelResidualQuery {
    /// The pressure field whose residual is measured, one `f32` per voxel.
    pub pressure: Vec<f32>,
    /// The constant right-hand side of the level system, one `f32` per voxel.
    pub rhs: Vec<f32>,
    /// The grid this level is discretized on.
    pub resolution: GridResolution,
    /// Operator scale `inv_h2 = 1/h^2` of this level.
    pub inv_h2: f32,
    /// Wall-model code ([`PRESSURE_BOUNDARY_DIRICHLET`] or
    /// [`PRESSURE_BOUNDARY_NEUMANN`]).
    pub boundary: u32,
}

/// The outcome of a [`GpuMgLevelResidual::residual`] solve.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMgLevelResidualResult {
    /// The `L2` residual `‖rhs − A·p‖` of the field.
    pub residual: f32,
}

/// A compiled, reusable single-level residual pipeline.
pub struct GpuMgLevelResidual {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    residual_pipeline: ComputePipeline,
}

impl GpuMgLevelResidual {
    /// Compiles the residual kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMgLevelResidual {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mg_level_residual"),
            source: ShaderSource::Wgsl(MG_LEVEL_RESIDUAL_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mg_level_residual_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mg_level_residual_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let residual_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_level_residual_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("residual_sq"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMgLevelResidual {
            module,
            layout,
            residual_pipeline,
        }
    }

    /// Measures the `L2` residual `‖rhs − A·p‖` of `query.pressure`.
    ///
    /// The returned residual equals the golden single-level `level_residual_l2`
    /// to within the tolerance documented on this module. An empty grid, or a
    /// `pressure`/`rhs` shorter than `resolution.voxel_count()`, returns a zero
    /// residual (no dispatch), matching the reference's degenerate-input guards.
    #[must_use]
    pub fn residual(
        &self,
        ctx: &GpuContext,
        query: &GpuMgLevelResidualQuery,
    ) -> GpuMgLevelResidualResult {
        let count = query.resolution.voxel_count() as usize;
        if count == 0 || query.pressure.len() < count || query.rhs.len() < count {
            return GpuMgLevelResidualResult { residual: 0.0 };
        }

        let device = ctx.device();

        let gpu_params = Params {
            nx: query.resolution.nx,
            ny: query.resolution.ny,
            nz: query.resolution.nz,
            boundary: query.boundary,
            inv_h2: query.inv_h2,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let pressure = &query.pressure[..count];
        let rhs = &query.rhs[..count];
        let field_bytes = (count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_level_residual_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let rhs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_level_residual_rhs"),
            contents: bytemuck::cast_slice(rhs),
            usage: BufferUsages::STORAGE,
        });
        let pressure_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_level_residual_pressure"),
            contents: bytemuck::cast_slice(pressure),
            usage: BufferUsages::STORAGE,
        });
        // Per-cell squared residual of the field.
        let sq_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_level_residual_sq"),
            size: field_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let residual_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_level_residual_stage"),
            size: field_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_level_residual_bind_group"),
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
                    resource: pressure_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: sq_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_level_residual_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_level_residual_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.residual_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&sq_buf, 0, &residual_stage, 0, field_bytes);
        ctx.queue().submit([encoder.finish()]);

        residual_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let residual_view = residual_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped residual readback range should be available after poll");
        let squares = bytemuck::cast_slice::<u8, f32>(&residual_view).to_vec();
        drop(residual_view);
        residual_stage.unmap();

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

        GpuMgLevelResidualResult { residual }
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
