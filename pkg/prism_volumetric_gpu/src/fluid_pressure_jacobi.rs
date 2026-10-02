//! `wgpu` compute twin of the `CPU` golden fixed-iteration `Jacobi` pressure
//! solver and its gradient-projection write-back
//! ([`jacobi_pressure_solve`](prism_render_architecture::particle::fluid::jacobi_pressure_solve),
//! design §10).
//!
//! Step three of the stable-fluids pipeline makes the velocity field (near)
//! divergence-free. It solves the pressure Poisson equation `∇²p = div` on a
//! unit-spaced grid with homogeneous (`p = 0`) Dirichlet boundaries, then
//! subtracts the pressure gradient from the velocity. The `CPU` golden
//! [`fluid`](prism_render_architecture::particle::fluid) module owns that math;
//! [`GpuFluidPressureJacobi`] is the on-device twin, validated against that
//! reference so a passing real-device parity test is direct evidence the ported
//! kernel relaxes the same field, not merely that its shader compiles.
//!
//! # Algorithm
//!
//! The pressure solve reproduces the reference sweep for sweep. One thread owns
//! one voxel. Each dispatch is one `Jacobi` sweep that reads the previous
//! iterate (`current`) and the constant right-hand side (`divergence`) and
//! writes the next iterate as `(Σ₆ neighbors − div) / 6`; the host ping-pongs
//! two storage buffers so `iterations` sweeps run back to back, exactly
//! mirroring the host loop in
//! [`jacobi_pressure_solve`](prism_render_architecture::particle::fluid::jacobi_pressure_solve).
//! The six axis-aligned face neighbors are summed in the identical order the
//! scalar reference uses (`+x`, `−x`, `+y`, `−y`, `+z`, `−z`) with homogeneous
//! Dirichlet walls (an out-of-grid neighbor contributes nothing). The pressure
//! field is seeded to all zeros exactly as the reference seeds it.
//!
//! The gradient-projection kernel twins the
//! [`central_gradient`](prism_render_architecture::particle::fluid::central_gradient)
//! and
//! [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient)
//! cluster: one thread owns one query and computes
//! `velocity − ((x_plus − x_minus)·inv_2h, (y_plus − y_minus)·inv_2h, (z_plus − z_minus)·inv_2h)`,
//! the same multiply-add-then-subtract the reference evaluates.
//!
//! # Fixed-iteration contract
//!
//! The reference stops at `plan.max_iterations` sweeps or when the `L2`
//! residual falls to `plan.residual_tolerance`, whichever comes first. The twin
//! takes a plain host `iterations` count and always runs exactly that many
//! sweeps, matching the reference whenever the reference is driven to its full
//! iteration budget (the parity test forces this with a non-positive
//! tolerance). The final `L2` residual is a host reduction over the
//! device-produced pressure field, summed in the reference's `z→y→x` order so
//! the twin and the reference accumulate the same algebra; a `GPU` reduction
//! would reorder that sum and be a weaker match.
//!
//! # Boundaries
//!
//! Both the sweep kernel and the host residual use homogeneous Dirichlet walls:
//! a neighbor outside the grid is skipped (its implicit value is the zero
//! wall), so a boundary voxel still divides by the full `6`, exactly as the
//! reference does.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+ − × ÷` on `f32` — with no `sin`, `cos`, `exp`, `log`,
//! `pow` or optional device feature, so they run unmodified on `Metal`,
//! `Vulkan` and `DX12`. A `Jacobi` sweep is multiply-add plus one division per
//! voxel, so it needs nothing beyond that subset.
//!
//! # Correctness model
//!
//! The sweep contains no transcendental call, so `CPU` and `GPU` evaluate the
//! same closed-form algebra in the same order. They are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits by a few `ULP`, and that perturbation compounds across the
//! iterated sweeps. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a swapped neighbor, a dropped boundary case, a missing divisor)
//! yet loose enough to admit legal fused multiply-add contraction summed over
//! the sweep count.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twins the `CPU` golden `jacobi_pressure_solve`,
//! `pressure_residual_l2`, `central_gradient` and `subtract_pressure_gradient`
//! in `prism_render_architecture::particle::fluid`; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::fluid::{GridResolution, NeighborScalars};
use prism_render_architecture::particle::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The number of threads per workgroup. `64` is a portable, warp-friendly size
/// used across this crate's kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The number of `f32` lanes one gradient-projection query occupies in the
/// flat upload buffer: the six neighbor samples, the three velocity components,
/// then `inv_2h`.
const PROJECT_STRIDE: usize = 10;

/// The portable core-`WGSL` pressure-relaxation kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` sweep exactly; see the
/// module documentation for the algorithm.
const SOLVE_WGSL: &str = r#"
// Pressure Jacobi-relaxation twin: one thread per voxel performs one sweep of
// the pressure Poisson system, updating its cell to
// `(neighbor_sum - divergence) / 6`. It mirrors the CPU golden
// `particle::fluid::jacobi_pressure_solve`, uses only the portable core-WGSL
// subset (integer index math plus + - * / on scalars), and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twins the CPU golden jacobi_pressure_solve; no Unreal Engine
// source or derived code.

struct Params {
    // Grid extents in voxels along each axis.
    nx: u32,
    ny: u32,
    nz: u32,
    // Padding to keep the uniform a multiple of 16 bytes.
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The constant right-hand side (the velocity divergence field).
@group(0) @binding(1) var<storage, read> divergence: array<f32>;
// The previous Jacobi iterate whose neighbors this sweep reads.
@group(0) @binding(2) var<storage, read> current: array<f32>;
// The next iterate this sweep writes.
@group(0) @binding(3) var<storage, read_write> out_field: array<f32>;

// Row-major linear index, matching `GridResolution::linear_index`:
// (z * ny + y) * nx + x.
fn lin(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.ny + y) * params.nx + x;
}

@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Sum the in-grid face neighbors in the exact order the scalar reference
    // uses. A missing neighbor is the zero Dirichlet wall, so it is skipped.
    var sum = 0.0;
    if (x + 1u < params.nx) {
        sum = sum + current[lin(x + 1u, y, z)];
    }
    if (x > 0u) {
        sum = sum + current[lin(x - 1u, y, z)];
    }
    if (y + 1u < params.ny) {
        sum = sum + current[lin(x, y + 1u, z)];
    }
    if (y > 0u) {
        sum = sum + current[lin(x, y - 1u, z)];
    }
    if (z + 1u < params.nz) {
        sum = sum + current[lin(x, y, z + 1u)];
    }
    if (z > 0u) {
        sum = sum + current[lin(x, y, z - 1u)];
    }

    // Divide by the full six faces, matching the reference's `/ 6.0`.
    out_field[idx] = (sum - divergence[idx]) / 6.0;
}
"#;

/// The portable core-`WGSL` gradient-projection kernel: one thread per query
/// forms the central-difference gradient and subtracts it from the velocity.
const PROJECT_WGSL: &str = r#"
// Gradient-projection twin: one thread per query computes
// `velocity - central_gradient(neighbors, inv_2h)`. It mirrors the CPU golden
// `central_gradient` + `subtract_pressure_gradient`, uses only the portable
// core-WGSL subset, and takes no optional feature.
//
// Provenance: twins the CPU golden central_gradient and
// subtract_pressure_gradient; no Unreal Engine source or derived code.

struct Params {
    // Number of queries.
    count: u32,
    // Padding to keep the uniform a multiple of 16 bytes.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Flat query lanes: 10 f32 per query (x_plus, x_minus, y_plus, y_minus,
// z_plus, z_minus, velocity.x, velocity.y, velocity.z, inv_2h).
@group(0) @binding(1) var<storage, read> src: array<f32>;
// The projected velocity, one padded vec4 per query.
@group(0) @binding(2) var<storage, read_write> out_field: array<vec4<f32>>;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let base = idx * 10u;
    let x_plus = src[base + 0u];
    let x_minus = src[base + 1u];
    let y_plus = src[base + 2u];
    let y_minus = src[base + 3u];
    let z_plus = src[base + 4u];
    let z_minus = src[base + 5u];
    let vx = src[base + 6u];
    let vy = src[base + 7u];
    let vz = src[base + 8u];
    let inv_2h = src[base + 9u];

    // Central-difference gradient (multiply-add only), then v - grad.
    let gx = (x_plus - x_minus) * inv_2h;
    let gy = (y_plus - y_minus) * inv_2h;
    let gz = (z_plus - z_minus) * inv_2h;
    out_field[idx] = vec4<f32>(vx - gx, vy - gy, vz - gz, 0.0);
}
"#;

/// Uniform parameters for one pressure sweep. `repr(C)` `std430` layout matching
/// `Params` in [`SOLVE_WGSL`]: the three grid extents then one pad word — `16`
/// bytes total with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SolveParams {
    /// Grid extent in voxels along `x`.
    nx: u32,
    /// Grid extent in voxels along `y`.
    ny: u32,
    /// Grid extent in voxels along `z`.
    nz: u32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad0: u32,
}

/// Uniform parameters for one gradient-projection dispatch. `repr(C)` `std430`
/// layout matching `Params` in [`PROJECT_WGSL`]: the query count then three pad
/// words — `16` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProjectParams {
    /// Number of queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One projected velocity as read back. `16`-byte `std430` stride matching
/// `array<vec4<f32>>` in the shader: the three components plus a zero pad lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVec3Pad {
    /// X component.
    x: f32,
    /// Y component.
    y: f32,
    /// Z component.
    z: f32,
    /// Padding lane, held at zero so it never perturbs the arithmetic.
    pad: f32,
}

impl GpuVec3Pad {
    /// Unpacks the device layout back into a [`Vec3`], dropping the pad lane.
    fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

/// A pressure-solve request: the divergence field, its grid, and the fixed
/// number of `Jacobi` sweeps to run.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuPressureSolveQuery {
    /// The velocity divergence field, row-major, one scalar per voxel.
    pub divergence: Vec<f32>,
    /// The grid resolution the field is sampled on.
    pub resolution: GridResolution,
    /// The fixed number of `Jacobi` sweeps to run.
    pub iterations: u32,
}

/// The outcome of a `GPU` pressure solve, mirroring the `CPU`
/// [`PressureSolveResult`](prism_render_architecture::particle::fluid::PressureSolveResult).
#[derive(Clone, Debug, PartialEq)]
pub struct GpuPressureSolveResult {
    /// The solved pressure field, row-major, one scalar per voxel.
    pub pressure: Vec<f32>,
    /// The final `L2` residual `‖∇²p − div‖`, a host reduction over the
    /// device-produced pressure field.
    pub residual: f32,
    /// How many `Jacobi` sweeps actually ran.
    pub iterations_run: u32,
}

/// A single gradient-projection query: the six neighbor pressure samples, the
/// velocity to project, and `inv_2h = 1 / (2·h)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuGradientProjectionQuery {
    /// The six axis-aligned neighbor samples around the voxel.
    pub neighbors: NeighborScalars,
    /// The velocity to project.
    pub velocity: Vec3,
    /// The reciprocal `1 / (2·h)` for cell size `h`.
    pub inv_2h: f32,
}

/// A compiled, reusable pressure-solve and gradient-projection pipeline pair.
pub struct GpuFluidPressureJacobi {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    solve_module: ShaderModule,
    solve_layout: BindGroupLayout,
    solve_pipeline: ComputePipeline,
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    project_module: ShaderModule,
    project_layout: BindGroupLayout,
    project_pipeline: ComputePipeline,
}

impl GpuFluidPressureJacobi {
    /// Compiles the pressure-solve and gradient-projection kernels on `ctx`.
    ///
    /// Both kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFluidPressureJacobi {
        let device = ctx.device();

        let solve_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_solve"),
            source: ShaderSource::Wgsl(SOLVE_WGSL.into()),
        });
        let solve_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_solve_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let solve_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_solve_pipeline_layout"),
            bind_group_layouts: &[Some(&solve_layout)],
            immediate_size: 0,
        });
        let solve_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_solve_pipeline"),
            layout: Some(&solve_pipeline_layout),
            module: &solve_module,
            entry_point: Some("sweep"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let project_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project"),
            source: ShaderSource::Wgsl(PROJECT_WGSL.into()),
        });
        let project_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let project_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project_pipeline_layout"),
            bind_group_layouts: &[Some(&project_layout)],
            immediate_size: 0,
        });
        let project_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project_pipeline"),
            layout: Some(&project_pipeline_layout),
            module: &project_module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuFluidPressureJacobi {
            solve_module,
            solve_layout,
            solve_pipeline,
            project_module,
            project_layout,
            project_pipeline,
        }
    }

    /// Solves the pressure Poisson equation with `query.iterations` `Jacobi`
    /// sweeps, returning the pressure field in input order plus the final `L2`
    /// residual.
    ///
    /// The returned field equals
    /// [`jacobi_pressure_solve`](prism_render_architecture::particle::fluid::jacobi_pressure_solve)
    /// driven to the same sweep count, to within the tolerance documented on
    /// this module. An empty grid, or a `query.divergence` carrying fewer than
    /// `query.resolution.voxel_count()` samples, yields an empty field, zero
    /// residual and zero iterations, matching the reference's degenerate-input
    /// guard. A zero iteration count returns the all-zero seed field and its
    /// residual without dispatching, matching the reference's initial state.
    #[must_use]
    pub fn solve(&self, ctx: &GpuContext, query: &GpuPressureSolveQuery) -> GpuPressureSolveResult {
        let res = query.resolution;
        let count = res.voxel_count() as usize;
        if count == 0 || query.divergence.len() < count {
            return GpuPressureSolveResult {
                pressure: Vec::new(),
                residual: 0.0,
                iterations_run: 0,
            };
        }

        let divergence = &query.divergence[..count];

        // The Jacobi seed is the all-zero field; a zero-sweep solve returns it
        // and its residual without touching the device, exactly as the
        // reference's pre-loop state.
        if query.iterations == 0 {
            let pressure = vec![0.0f32; count];
            let residual = host_residual_l2(&pressure, divergence, res);
            return GpuPressureSolveResult {
                pressure,
                residual,
                iterations_run: 0,
            };
        }

        let device = ctx.device();
        let gpu_params = SolveParams {
            nx: res.nx,
            ny: res.ny,
            nz: res.nz,
            pad0: 0,
        };

        let seed = vec![0.0f32; count];
        let field_bytes = (count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let divergence_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_divergence"),
            contents: bytemuck::cast_slice(divergence),
            usage: BufferUsages::STORAGE,
        });
        // Ping-pong iterate buffers. Buffer A starts holding the zero seed so
        // the first sweep reads the same initial iterate the reference seeds.
        let buf_a = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_iterate_a"),
            contents: bytemuck::cast_slice(&seed),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let buf_b = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_iterate_b"),
            size: field_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_stage"),
            size: field_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group AB reads A and writes B; BA reads B and writes A.
        let bind_ab = self.solve_bind_group(device, &params_buf, &divergence_buf, &buf_a, &buf_b);
        let bind_ba = self.solve_bind_group(device, &params_buf, &divergence_buf, &buf_b, &buf_a);

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_encoder"),
        });
        for sweep in 0..query.iterations {
            let bind = if sweep % 2 == 0 { &bind_ab } else { &bind_ba };
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fluid_pressure_jacobi_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.solve_pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // Sweep `i` (zero-based) writes B when `i` is even and A when odd, so the
        // final write lands in B for an odd iteration count and A for an even
        // one.
        let final_buf = if query.iterations % 2 == 1 {
            &buf_b
        } else {
            &buf_a
        };
        encoder.copy_buffer_to_buffer(final_buf, 0, &stage, 0, field_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let pressure = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(pressure.len(), count);

        // The reference reports the residual of the final iterate; recompute it
        // on the host in the reference's z->y->x order over the device field.
        let residual = host_residual_l2(&pressure, divergence, res);

        GpuPressureSolveResult {
            pressure,
            residual,
            iterations_run: query.iterations,
        }
    }

    /// Projects a batch of velocities by subtracting their central-difference
    /// pressure gradients, one thread per query.
    ///
    /// Each result equals
    /// [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient)`(velocity, `[`central_gradient`](prism_render_architecture::particle::fluid::central_gradient)`(neighbors, inv_2h))`
    /// to within the tolerance documented on this module. An empty query slice
    /// returns an empty vector without dispatching.
    #[must_use]
    pub fn project(&self, ctx: &GpuContext, queries: &[GpuGradientProjectionQuery]) -> Vec<Vec3> {
        let n = queries.len();
        if n == 0 {
            return Vec::new();
        }

        let device = ctx.device();
        let gpu_params = ProjectParams {
            count: n as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Pack each query into the flat 10-lane layout the kernel reads.
        let mut packed = Vec::with_capacity(n * PROJECT_STRIDE);
        for q in queries {
            packed.push(q.neighbors.x_plus);
            packed.push(q.neighbors.x_minus);
            packed.push(q.neighbors.y_plus);
            packed.push(q.neighbors.y_minus);
            packed.push(q.neighbors.z_plus);
            packed.push(q.neighbors.z_minus);
            packed.push(q.velocity.x);
            packed.push(q.velocity.y);
            packed.push(q.velocity.z);
            packed.push(q.inv_2h);
        }
        let out_bytes = (n as u64) * (size_of::<GpuVec3Pad>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let src_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project_src"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project_bind_group"),
            layout: &self.project_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (n as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_project_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fluid_pressure_jacobi_project_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_out = bytemuck::cast_slice::<u8, GpuVec3Pad>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(gpu_out.len(), n);

        gpu_out.into_iter().map(GpuVec3Pad::to_vec3).collect()
    }

    /// Builds a sweep bind group binding the shared uniform and divergence
    /// alongside the chosen `current` (read) and `out` (write) iterate buffers.
    fn solve_bind_group(
        &self,
        device: &wgpu::Device,
        params_buf: &wgpu::Buffer,
        divergence_buf: &wgpu::Buffer,
        current: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fluid_pressure_jacobi_bind_group"),
            layout: &self.solve_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: divergence_buf.as_entire_binding(),
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

/// Sum of the six face-neighbor scalars around `(x, y, z)` with homogeneous
/// (`0`) Dirichlet boundaries, in the reference's `+x, −x, +y, −y, +z, −z`
/// order.
fn host_neighbor_sum(field: &[f32], res: GridResolution, x: u32, y: u32, z: u32) -> f32 {
    let mut sum = 0.0;
    if x + 1 < res.nx {
        sum += field[res.linear_index(x + 1, y, z) as usize];
    }
    if x > 0 {
        sum += field[res.linear_index(x - 1, y, z) as usize];
    }
    if y + 1 < res.ny {
        sum += field[res.linear_index(x, y + 1, z) as usize];
    }
    if y > 0 {
        sum += field[res.linear_index(x, y - 1, z) as usize];
    }
    if z + 1 < res.nz {
        sum += field[res.linear_index(x, y, z + 1) as usize];
    }
    if z > 0 {
        sum += field[res.linear_index(x, y, z - 1) as usize];
    }
    sum
}

/// `L2` residual of a pressure field against a divergence field, summed in the
/// reference's `z→y→x` order so the host reduction matches the golden algebra.
fn host_residual_l2(pressure: &[f32], divergence: &[f32], res: GridResolution) -> f32 {
    let count = res.voxel_count() as usize;
    if count == 0 || pressure.len() < count || divergence.len() < count {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for z in 0..res.nz {
        for y in 0..res.ny {
            for x in 0..res.nx {
                let idx = res.linear_index(x, y, z) as usize;
                let center = pressure[idx];
                let laplacian = host_neighbor_sum(pressure, res, x, y, z) - 6.0 * center;
                let r = divergence[idx] - laplacian;
                sum_sq += r * r;
            }
        }
    }
    (sum_sq / count as f32).sqrt()
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
