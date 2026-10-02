//! `wgpu` compute twin of the `CPU` golden single-level multigrid difference
//! operators: the forward-difference divergence
//! [`divergence_forward`](prism_render_architecture::particle::multigrid_pressure::divergence_forward)
//! and the wall-aware backward-difference gradient projection that composes the
//! private `gradient_backward` of the golden multigrid module with the public
//! [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient)
//! (design §10).
//!
//! These are the two one-sided difference operators the pressure-projection
//! stage wires together: the forward divergence feeds the pressure solve and
//! the backward gradient is subtracted from the velocity on write-back. The
//! `CPU` golden [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
//! and [`fluid`](prism_render_architecture::particle::fluid) modules own that
//! math; [`GpuMgDivergenceGradient`] is the on-device twin, validated against
//! those references so a passing real-device parity test is direct evidence the
//! ported kernels evaluate the same stencils, not merely that their shaders
//! compile. This module does not run a `V`-cycle or a solve; it is only the two
//! per-cell gather operators.
//!
//! # Algorithm
//!
//! Both kernels are pure per-voxel gathers: one thread owns one voxel, reads
//! only its own cell and its axis-aligned neighbors, and never scatters.
//!
//! The divergence kernel twins
//! [`divergence_forward`](prism_render_architecture::particle::multigrid_pressure::divergence_forward).
//! For voxel `(x, y, z)` it reads `here = velocity[idx]`, the `+x` neighbor's
//! `x` component, the `+y` neighbor's `y` component and the `+z` neighbor's `z`
//! component, then writes `(vx − here.x) + (vy − here.y) + (vz − here.z)` in
//! that exact grouping and order. A high-face neighbor outside the grid
//! contributes `0.0` (a closed wall), matching the reference.
//!
//! The projection kernel twins the composition
//! [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient)`(velocity[idx], gradient_backward(pressure, …))`.
//! For voxel `(x, y, z)` it forms the backward difference
//! `gx = x > 0 ? here − pressure[−x] : 0.0` (and likewise `gy`, `gz` on the
//! low faces), then writes `velocity[idx] − (gx, gy, gz)`. A low-face neighbor
//! outside the grid drops that term (zero normal gradient at a wall), matching
//! the reference. The golden `gradient_backward` is a private function, so the
//! parity test transcribes it as a local mirror (see that test's provenance
//! comment) and composes it with the public `subtract_pressure_gradient`.
//!
//! Both kernels share the row-major linear index of
//! [`GridResolution::linear_index`](prism_render_architecture::particle::fluid::GridResolution::linear_index),
//! `(z · ny + y) · nx + x`, replicated as `lin` in each shader.
//!
//! # Boundaries
//!
//! The divergence kernel uses the forward (high-face) difference: a neighbor at
//! `x + 1`, `y + 1` or `z + 1` outside the grid is treated as zero velocity.
//! The projection kernel uses the backward (low-face) difference: a neighbor at
//! `x − 1`, `y − 1` or `z − 1` outside the grid drops that gradient component.
//! Composed, the two one-sided operators reproduce the compact `Neumann`
//! stencil the golden modules document.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+ − × ÷` on `f32` — with no `sin`, `cos`, `exp`, `log`,
//! `pow` or optional device feature, so they run unmodified on `Metal`,
//! `Vulkan` and `DX12`. Each is one gather plus a handful of subtractions per
//! voxel.
//!
//! # Correctness model
//!
//! Neither kernel contains a transcendental call, so `CPU` and `GPU` evaluate
//! the same closed-form algebra in the same order. They are not bit-exact: a
//! `GPU` may fuse a multiply-add the scalar reference leaves separate,
//! perturbing the low mantissa bits by a few `ULP`. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough
//! to catch a genuinely wrong port (a swapped neighbor, a dropped boundary
//! case, a flipped difference direction) yet loose enough to admit legal fused
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twins the `CPU` golden `divergence_forward` and the composition
//! of the private `gradient_backward` with the public
//! `subtract_pressure_gradient` in
//! `prism_render_architecture::particle::multigrid_pressure` and
//! `prism_render_architecture::particle::fluid`; no Unreal Engine source or
//! derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::fluid::GridResolution;
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

/// The portable core-`WGSL` forward-difference divergence kernel, embedded
/// inline so the twin ships as a single source file. Mirrors the `CPU` gather
/// exactly; see the module documentation for the algorithm.
const DIVERGENCE_WGSL: &str = r#"
// Forward-difference divergence twin: one thread per voxel reads its own
// velocity and the +x/+y/+z face neighbors, writing
// `(vx - here.x) + (vy - here.y) + (vz - here.z)`. It mirrors the CPU golden
// `particle::multigrid_pressure::divergence_forward`, uses only the portable
// core-WGSL subset (integer index math plus + - * / on scalars), and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twins the CPU golden divergence_forward; no Unreal Engine source
// or derived code.

struct Params {
    // Grid extents in voxels along each axis.
    nx: u32,
    ny: u32,
    nz: u32,
    // Padding to keep the uniform a multiple of 16 bytes.
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The input velocity field, one padded vec4 per voxel.
@group(0) @binding(1) var<storage, read> velocity: array<vec4<f32>>;
// The forward-difference divergence this kernel writes, one scalar per voxel.
@group(0) @binding(2) var<storage, read_write> out_field: array<f32>;

// Row-major linear index, matching `GridResolution::linear_index`:
// (z * ny + y) * nx + x.
fn lin(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.ny + y) * params.nx + x;
}

@compute @workgroup_size(64)
fn divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let here = velocity[idx];

    // High-face neighbors; an out-of-grid neighbor is the zero wall.
    var vx = 0.0;
    if (x + 1u < params.nx) {
        vx = velocity[lin(x + 1u, y, z)].x;
    }
    var vy = 0.0;
    if (y + 1u < params.ny) {
        vy = velocity[lin(x, y + 1u, z)].y;
    }
    var vz = 0.0;
    if (z + 1u < params.nz) {
        vz = velocity[lin(x, y, z + 1u)].z;
    }

    // Exact grouping/order of the scalar reference.
    out_field[idx] = (vx - here.x) + (vy - here.y) + (vz - here.z);
}
"#;

/// The portable core-`WGSL` backward-difference gradient-projection kernel: one
/// thread per voxel subtracts the wall-aware backward pressure gradient from
/// the velocity. Mirrors the `CPU` composition exactly; see the module
/// documentation for the algorithm.
const PROJECTION_WGSL: &str = r#"
// Gradient-projection twin: one thread per voxel forms the backward-difference
// pressure gradient `gradient_backward` and subtracts it from the velocity via
// `subtract_pressure_gradient`. It mirrors the CPU golden composition in
// `particle::multigrid_pressure` and `particle::fluid`, uses only the portable
// core-WGSL subset, and takes no optional feature.
//
// Provenance: twins the CPU golden gradient_backward (private) composed with
// subtract_pressure_gradient; no Unreal Engine source or derived code.

struct Params {
    // Grid extents in voxels along each axis.
    nx: u32,
    ny: u32,
    nz: u32,
    // Padding to keep the uniform a multiple of 16 bytes.
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The input velocity field, one padded vec4 per voxel.
@group(0) @binding(1) var<storage, read> velocity: array<vec4<f32>>;
// The input pressure field, one scalar per voxel.
@group(0) @binding(2) var<storage, read> pressure: array<f32>;
// The projected velocity this kernel writes, one padded vec4 per voxel.
@group(0) @binding(3) var<storage, read_write> out_field: array<vec4<f32>>;

// Row-major linear index, matching `GridResolution::linear_index`:
// (z * ny + y) * nx + x.
fn lin(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.ny + y) * params.nx + x;
}

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let here = pressure[idx];

    // Backward-difference gradient; a low-face neighbor outside the grid drops
    // its term (zero normal gradient at a wall).
    var gx = 0.0;
    if (x > 0u) {
        gx = here - pressure[lin(x - 1u, y, z)];
    }
    var gy = 0.0;
    if (y > 0u) {
        gy = here - pressure[lin(x, y - 1u, z)];
    }
    var gz = 0.0;
    if (z > 0u) {
        gz = here - pressure[lin(x, y, z - 1u)];
    }

    let v = velocity[idx];
    out_field[idx] = vec4<f32>(v.x - gx, v.y - gy, v.z - gz, 0.0);
}
"#;

/// Uniform parameters for both kernels. `repr(C)` `std430` layout matching
/// `Params` in [`DIVERGENCE_WGSL`] and [`PROJECTION_WGSL`]: the three grid
/// extents then one pad word — `16` bytes total with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridParams {
    /// Grid extent in voxels along `x`.
    nx: u32,
    /// Grid extent in voxels along `y`.
    ny: u32,
    /// Grid extent in voxels along `z`.
    nz: u32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad0: u32,
}

/// One velocity sample as uploaded or read back. `16`-byte `std430` stride
/// matching `array<vec4<f32>>` in the shaders: the three components plus a zero
/// pad lane.
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
    /// Packs a [`Vec3`] into the padded device layout.
    fn from_vec3(v: Vec3) -> GpuVec3Pad {
        GpuVec3Pad {
            x: v.x,
            y: v.y,
            z: v.z,
            pad: 0.0,
        }
    }

    /// Unpacks the device layout back into a [`Vec3`], dropping the pad lane.
    fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

/// A forward-difference divergence request: a velocity field and its grid.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuDivergenceQuery {
    /// The velocity field, row-major, one vector per voxel.
    pub velocity: Vec<Vec3>,
    /// The grid resolution the field is sampled on.
    pub resolution: GridResolution,
}

/// A gradient-projection request: a velocity field, the pressure field whose
/// backward gradient is subtracted, and their shared grid.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuProjectionQuery {
    /// The velocity field to project, row-major, one vector per voxel.
    pub velocity: Vec<Vec3>,
    /// The pressure field, row-major, one scalar per voxel.
    pub pressure: Vec<f32>,
    /// The grid resolution both fields are sampled on.
    pub resolution: GridResolution,
}

/// A compiled, reusable divergence and gradient-projection pipeline pair.
pub struct GpuMgDivergenceGradient {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    divergence_module: ShaderModule,
    divergence_layout: BindGroupLayout,
    divergence_pipeline: ComputePipeline,
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    projection_module: ShaderModule,
    projection_layout: BindGroupLayout,
    projection_pipeline: ComputePipeline,
}

impl GpuMgDivergenceGradient {
    /// Compiles the divergence and gradient-projection kernels on `ctx`.
    ///
    /// Both kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMgDivergenceGradient {
        let device = ctx.device();

        let divergence_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence"),
            source: ShaderSource::Wgsl(DIVERGENCE_WGSL.into()),
        });
        let divergence_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let divergence_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence_pipeline_layout"),
            bind_group_layouts: &[Some(&divergence_layout)],
            immediate_size: 0,
        });
        let divergence_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence_pipeline"),
            layout: Some(&divergence_pipeline_layout),
            module: &divergence_module,
            entry_point: Some("divergence"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let projection_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection"),
            source: ShaderSource::Wgsl(PROJECTION_WGSL.into()),
        });
        let projection_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let projection_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_pipeline_layout"),
            bind_group_layouts: &[Some(&projection_layout)],
            immediate_size: 0,
        });
        let projection_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_pipeline"),
            layout: Some(&projection_pipeline_layout),
            module: &projection_module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuMgDivergenceGradient {
            divergence_module,
            divergence_layout,
            divergence_pipeline,
            projection_module,
            projection_layout,
            projection_pipeline,
        }
    }

    /// Computes the forward-difference divergence of `query.velocity`, one
    /// thread per voxel.
    ///
    /// The returned field equals
    /// [`divergence_forward`](prism_render_architecture::particle::multigrid_pressure::divergence_forward)`(&query.velocity, query.resolution)`
    /// element for element, to within the tolerance documented on this module,
    /// whenever the input is well formed. An empty grid, or a `query.velocity`
    /// carrying fewer than `query.resolution.voxel_count()` samples, yields an
    /// empty vector without dispatching.
    #[must_use]
    pub fn divergence(&self, ctx: &GpuContext, query: &GpuDivergenceQuery) -> Vec<f32> {
        let res = query.resolution;
        let count = res.voxel_count() as usize;
        if count == 0 || query.velocity.len() < count {
            return Vec::new();
        }

        let device = ctx.device();
        let gpu_params = GridParams {
            nx: res.nx,
            ny: res.ny,
            nz: res.nz,
            pad0: 0,
        };

        let packed: Vec<GpuVec3Pad> = query.velocity[..count]
            .iter()
            .copied()
            .map(GpuVec3Pad::from_vec3)
            .collect();
        let out_bytes = (count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let velocity_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence_velocity"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence_bind_group"),
            layout: &self.divergence_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: velocity_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_divergence_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_divergence_gradient_divergence_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.divergence_pipeline);
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
        let gpu_out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(gpu_out.len(), count);

        gpu_out
    }

    /// Projects `query.velocity` by subtracting the wall-aware backward-
    /// difference gradient of `query.pressure`, one thread per voxel.
    ///
    /// Each result equals
    /// [`subtract_pressure_gradient`](prism_render_architecture::particle::fluid::subtract_pressure_gradient)`(velocity[idx], gradient_backward(pressure, resolution, x, y, z))`
    /// to within the tolerance documented on this module, where
    /// `gradient_backward` is the private golden backward difference the parity
    /// test mirrors. An empty grid, or a `query.velocity` or `query.pressure`
    /// carrying fewer than `query.resolution.voxel_count()` samples, yields an
    /// empty vector without dispatching.
    #[must_use]
    pub fn project(&self, ctx: &GpuContext, query: &GpuProjectionQuery) -> Vec<Vec3> {
        let res = query.resolution;
        let count = res.voxel_count() as usize;
        if count == 0 || query.velocity.len() < count || query.pressure.len() < count {
            return Vec::new();
        }

        let device = ctx.device();
        let gpu_params = GridParams {
            nx: res.nx,
            ny: res.ny,
            nz: res.nz,
            pad0: 0,
        };

        let packed_velocity: Vec<GpuVec3Pad> = query.velocity[..count]
            .iter()
            .copied()
            .map(GpuVec3Pad::from_vec3)
            .collect();
        let pressure = &query.pressure[..count];
        let out_bytes = (count as u64) * (size_of::<GpuVec3Pad>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let velocity_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_velocity"),
            contents: bytemuck::cast_slice(&packed_velocity),
            usage: BufferUsages::STORAGE,
        });
        let pressure_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_pressure"),
            contents: bytemuck::cast_slice(pressure),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_bind_group"),
            layout: &self.projection_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: velocity_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: pressure_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_mg_divergence_gradient_projection_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_mg_divergence_gradient_projection_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.projection_pipeline);
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
        debug_assert_eq!(gpu_out.len(), count);

        gpu_out.into_iter().map(GpuVec3Pad::to_vec3).collect()
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
