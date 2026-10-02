//! `wgpu` compute twin of the particle-subsystem *vorticity-confinement* force
//! ([`vorticity_confinement_force`](prism_render_architecture::particle::fluid::vorticity_confinement_force),
//! fluid design §10).
//!
//! Semi-Lagrangian advection numerically dissipates small-scale swirl; the
//! vorticity-confinement step restores it so smoke rolls stay crisp. For each
//! voxel the `CPU` golden
//! [`fluid`](prism_render_architecture::particle::fluid) module forms the local
//! curl `ω = ∇ × v` from six axis-aligned velocity neighbors
//! ([`NeighborVelocities::curl`](prism_render_architecture::particle::fluid::NeighborVelocities::curl)),
//! normalizes the gradient of `‖ω‖` into a direction `N` that points toward
//! higher vorticity, and injects the location force `ε·h·(N × ω)`
//! ([`vorticity_confinement_force`](prism_render_architecture::particle::fluid::vorticity_confinement_force)).
//! [`GpuFluidVorticity`] is the on-device twin, validated against that
//! reference so a passing real-device parity test is direct evidence the ported
//! kernel restores the same force field, not merely that its shader compiles.
//!
//! # Algorithm
//!
//! One thread owns one voxel. The kernel samples the six axis-aligned velocity
//! neighbors with clamp-to-edge boundaries (the replicate convention the golden
//! [`sample_velocity_field`](prism_render_architecture::particle::fluid::sample_velocity_field)
//! uses, where an out-of-grid coordinate is clamped onto the last live cell),
//! assembles the same
//! [`NeighborVelocities`](prism_render_architecture::particle::fluid::NeighborVelocities)
//! the reference folds, and evaluates the identical central-difference curl in
//! the identical term order. It then reads the host-supplied gradient of
//! `‖ω‖`, normalizes it with the same guarded rule the golden
//! [`normalize_or_zero`](prism_render_architecture::particle::Vec3::normalize_or_zero)
//! applies, and forms the cross product and scale exactly as
//! [`vorticity_confinement_force`](prism_render_architecture::particle::fluid::vorticity_confinement_force)
//! does. The curl and the confinement force are written out per voxel.
//!
//! # Why the gradient is an input
//!
//! The golden
//! [`vorticity_confinement_force`](prism_render_architecture::particle::fluid::vorticity_confinement_force)
//! takes the gradient of `‖ω‖` as a direct argument rather than deriving it
//! from a field, so the twin mirrors that contract: the per-voxel gradient is
//! uploaded alongside the velocity field and the kernel twins only the two pure
//! functions the reference actually owns (the curl and the confinement force).
//! This keeps every device output checkable against an existing golden function
//! with no invented field math.
//!
//! # Guarded normalize
//!
//! `WGSL` has no `normalize_or_zero`, so the twin re-implements the golden rule
//! by hand: it forms `len² = dot(g, g)` and, only when `len² > EPS_LEN_SQ`
//! (the golden
//! [`EPS_LEN_SQ`](prism_render_architecture::particle::EPS_LEN_SQ) of `1e-12`),
//! scales by `1 / sqrt(len²)`; otherwise it returns the zero vector. A flat
//! vorticity region therefore contributes no force instead of a `NaN`, matching
//! the reference branch for branch. The fixtures keep every non-flat gradient's
//! `len²` far above the threshold so the device and the reference always take
//! the same branch.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: integer index
//! arithmetic, `f32` `+ − × ÷`, the `cross` / `dot` built-ins and a single
//! `sqrt` for the normalize. There is no `sin`, `cos`, `exp`, `log`, `pow`,
//! inverse trigonometric call, `f64` or `u64`, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12` with no optional device feature.
//!
//! # Correctness model
//!
//! The curl, cross and scale are closed-form multiply-add algebra evaluated in
//! the golden's term order, so `CPU` and `GPU` compute the same expression; they
//! are not bit-exact because a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate and the normalize `sqrt` / reciprocal may differ by
//! a few `ULP`. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a wrong
//! port (a swapped neighbor, a transposed cross product, a dropped boundary
//! clamp) yet loose enough to admit a legal fused multiply-add.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::fluid::{GridResolution, VorticityParams};
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` vorticity-confinement kernel, embedded inline so the
/// twin ships as a single source file. Mirrors the `CPU` golden curl and
/// confinement force exactly; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
const FLUID_VORTICITY_WGSL: &str = r#"
// Vorticity-confinement twin: one thread per voxel samples its six axis
// neighbors with clamp-to-edge boundaries, forms the central-difference curl in
// the golden `NeighborVelocities::curl` term order, normalizes the host-supplied
// gradient of |curl| with the golden guarded rule, and injects the force
// epsilon*cell_size*(N x curl) exactly as `vorticity_confinement_force`. It
// uses only the portable core-WGSL subset (integer index math, f32 + - * /,
// cross / dot and one sqrt) and takes no optional feature, so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::fluid;
// 无第三方引擎源码或衍生代码。

// Dispatch parameters. 32-byte uniform block: the three grid extents and one
// pad word, then the finite-difference scale, the confinement coefficient, the
// cell size and one pad word, matching the host `Params`.
struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    pad0: u32,
    inv_2h: f32,
    epsilon: f32,
    cell_size: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Velocity field, row-major, one padded vec4 per voxel.
@group(0) @binding(1) var<storage, read> velocity: array<vec4<f32>>;
// Host-supplied gradient of |curl|, row-major, one padded vec4 per voxel.
@group(0) @binding(2) var<storage, read> magnitude_gradient: array<vec4<f32>>;
// Per-voxel curl output.
@group(0) @binding(3) var<storage, read_write> out_curl: array<vec4<f32>>;
// Per-voxel confinement-force output.
@group(0) @binding(4) var<storage, read_write> out_force: array<vec4<f32>>;

// The golden `EPS_LEN_SQ`: the squared-length floor below which a direction
// vector is treated as zero so the normalize never yields a NaN.
const EPS_LEN_SQ: f32 = 1e-12;

// Row-major linear index, matching `GridResolution::linear_index`:
// (z * ny + y) * nx + x.
fn lin(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.ny + y) * params.nx + x;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Clamp-to-edge neighbor coordinates: an out-of-grid face replicates the
    // last live cell, the convention the golden `sample_velocity_field` clamp
    // uses. The host guarantees nx, ny, nz >= 1, so `n - 1u` never underflows.
    var xp = x + 1u;
    if (xp > params.nx - 1u) {
        xp = params.nx - 1u;
    }
    var xm = x;
    if (x > 0u) {
        xm = x - 1u;
    }
    var yp = y + 1u;
    if (yp > params.ny - 1u) {
        yp = params.ny - 1u;
    }
    var ym = y;
    if (y > 0u) {
        ym = y - 1u;
    }
    var zp = z + 1u;
    if (zp > params.nz - 1u) {
        zp = params.nz - 1u;
    }
    var zm = z;
    if (z > 0u) {
        zm = z - 1u;
    }

    let v_x_plus = velocity[lin(xp, y, z)].xyz;
    let v_x_minus = velocity[lin(xm, y, z)].xyz;
    let v_y_plus = velocity[lin(x, yp, z)].xyz;
    let v_y_minus = velocity[lin(x, ym, z)].xyz;
    let v_z_plus = velocity[lin(x, y, zp)].xyz;
    let v_z_minus = velocity[lin(x, y, zm)].xyz;

    // Central-difference curl, in the exact term order of the golden
    // `NeighborVelocities::curl`.
    let dwz_dy = v_y_plus.z - v_y_minus.z;
    let dvy_dz = v_z_plus.y - v_z_minus.y;
    let dux_dz = v_z_plus.x - v_z_minus.x;
    let dwz_dx = v_x_plus.z - v_x_minus.z;
    let dvy_dx = v_x_plus.y - v_x_minus.y;
    let dux_dy = v_y_plus.x - v_y_minus.x;
    let curl = vec3<f32>(
        (dwz_dy - dvy_dz) * params.inv_2h,
        (dux_dz - dwz_dx) * params.inv_2h,
        (dvy_dx - dux_dy) * params.inv_2h,
    );

    // Guarded normalize of the gradient, matching `Vec3::normalize_or_zero`:
    // scale by 1 / sqrt(len^2) only above the floor, else the zero vector.
    let g = magnitude_gradient[idx].xyz;
    let len_sq = dot(g, g);
    var n = vec3<f32>(0.0, 0.0, 0.0);
    if (len_sq > EPS_LEN_SQ) {
        n = g * (1.0 / sqrt(len_sq));
    }

    // Confinement force epsilon*cell_size*(N x curl), matching
    // `vorticity_confinement_force`: cross in the N-then-curl order, then scale.
    let force = cross(n, curl) * (params.epsilon * params.cell_size);

    out_curl[idx] = vec4<f32>(curl, 0.0);
    out_force[idx] = vec4<f32>(force, 0.0);
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`FLUID_VORTICITY_WGSL`]: the three grid extents and one pad
/// word, the finite-difference scale `inv_2h`, the confinement coefficient
/// `epsilon`, the `cell_size` and one pad word — `32` bytes with no interior
/// padding.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Grid extent in voxels along `x`.
    nx: u32,
    /// Grid extent in voxels along `y`.
    ny: u32,
    /// Grid extent in voxels along `z`.
    nz: u32,
    /// Padding word keeping the next field `16`-byte aligned.
    pad0: u32,
    /// The finite-difference scale `1 / (2·h)` used by the curl.
    inv_2h: f32,
    /// The confinement coefficient `epsilon`.
    epsilon: f32,
    /// The cell size `h` the force is scaled by.
    cell_size: f32,
    /// Padding lane keeping the struct a multiple of `16` bytes.
    pad1: f32,
}

/// One `Vec3` as uploaded and read back. `16`-byte `std430` stride matching
/// `array<vec4<f32>>` in the shader: the three components plus one pad lane held
/// at zero so it never perturbs the arithmetic.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVec3 {
    /// X component.
    x: f32,
    /// Y component.
    y: f32,
    /// Z component.
    z: f32,
    /// Padding lane, held at zero.
    pad: f32,
}

impl GpuVec3 {
    /// Packs a [`Vec3`] into the padded device layout.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
    fn from_vec3(v: Vec3) -> Self {
        GpuVec3 {
            x: v.x,
            y: v.y,
            z: v.z,
            pad: 0.0,
        }
    }

    /// Unpacks the device layout back into a [`Vec3`], dropping the pad lane.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
    fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

/// One vorticity-confinement request over a velocity grid.
///
/// The `velocity` and `magnitude_gradient` arrays are row-major, one [`Vec3`]
/// per voxel in [`GridResolution::linear_index`](prism_render_architecture::particle::fluid::GridResolution::linear_index)
/// order, and must each carry at least
/// [`GridResolution::voxel_count`](prism_render_architecture::particle::fluid::GridResolution::voxel_count)
/// entries. `inv_2h` is the `1 / (2·h)` finite-difference scale the golden curl
/// consumes, and `cell_size` is the `h` the force is scaled by.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuVorticityQuery {
    /// Grid extents in voxels.
    pub resolution: GridResolution,
    /// Row-major velocity field, one [`Vec3`] per voxel.
    pub velocity: Vec<Vec3>,
    /// Row-major gradient of `‖curl‖`, one [`Vec3`] per voxel (the golden
    /// confinement force takes this gradient as a direct input).
    pub magnitude_gradient: Vec<Vec3>,
    /// The confinement coefficient `epsilon`.
    pub params: VorticityParams,
    /// The finite-difference scale `1 / (2·h)`.
    pub inv_2h: f32,
    /// The cell size `h` the force is scaled by.
    pub cell_size: f32,
}

/// The device-computed answer of one vorticity-confinement solve: the per-voxel
/// curl and the per-voxel confinement force, both row-major in input order.
///
/// Both vectors are empty when the grid is empty or an input array is too short
/// (the host short-circuits without a dispatch).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
#[derive(Clone, Debug, PartialEq)]
pub struct GpuVorticityResult {
    /// Per-voxel curl `ω = ∇ × v`.
    pub curl: Vec<Vec3>,
    /// Per-voxel confinement force `ε·h·(N × ω)`.
    pub force: Vec<Vec3>,
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

/// A compiled, reusable vorticity-confinement compute pipeline, twinning the
/// `CPU` golden
/// [`fluid`](prism_render_architecture::particle::fluid) curl and confinement
/// force.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
pub struct GpuFluidVorticity {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFluidVorticity {
    /// Compiles the vorticity-confinement kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFluidVorticity {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_module"),
            source: ShaderSource::Wgsl(FLUID_VORTICITY_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
                buffer_entry(4, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFluidVorticity {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the vorticity-confinement solve on the device and returns the
    /// per-voxel curl and confinement force in input order.
    ///
    /// An empty grid, or a `velocity` / `magnitude_gradient` array shorter than
    /// [`GridResolution::voxel_count`](prism_render_architecture::particle::fluid::GridResolution::voxel_count),
    /// issues **no dispatch** — a storage buffer may not be zero-sized — and
    /// returns empty curl and force vectors, matching the reference's
    /// degenerate-input guard.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, query: &GpuVorticityQuery) -> GpuVorticityResult {
        let count = query.resolution.voxel_count() as usize;
        if count == 0 || query.velocity.len() < count || query.magnitude_gradient.len() < count {
            return GpuVorticityResult {
                curl: Vec::new(),
                force: Vec::new(),
            };
        }

        let device = ctx.device();

        let gpu_params = Params {
            nx: query.resolution.nx,
            ny: query.resolution.ny,
            nz: query.resolution.nz,
            pad0: 0,
            inv_2h: query.inv_2h,
            epsilon: query.params.epsilon,
            cell_size: query.cell_size,
            pad1: 0.0,
        };

        let packed_velocity: Vec<GpuVec3> = query.velocity[..count]
            .iter()
            .map(|&v| GpuVec3::from_vec3(v))
            .collect();
        let packed_gradient: Vec<GpuVec3> = query.magnitude_gradient[..count]
            .iter()
            .map(|&v| GpuVec3::from_vec3(v))
            .collect();
        let field_bytes = (count as u64) * (size_of::<GpuVec3>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let velocity_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_velocity"),
            contents: bytemuck::cast_slice(&packed_velocity),
            usage: BufferUsages::STORAGE,
        });
        let gradient_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_gradient"),
            contents: bytemuck::cast_slice(&packed_gradient),
            usage: BufferUsages::STORAGE,
        });

        let curl_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_curl"),
            size: field_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let force_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_force"),
            size: field_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let curl_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_curl_stage"),
            size: field_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let force_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_force_stage"),
            size: field_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_bind_group"),
            layout: &self.layout,
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
                    resource: gradient_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: curl_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: force_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fluid_vorticity_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fluid_vorticity_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per voxel, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&curl_buf, 0, &curl_stage, 0, field_bytes);
        encoder.copy_buffer_to_buffer(&force_buf, 0, &force_stage, 0, field_bytes);
        ctx.queue().submit([encoder.finish()]);

        curl_stage.slice(..).map_async(MapMode::Read, |_| {});
        force_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        let curl = read_field(&curl_stage, count);
        let force = read_field(&force_stage, count);

        GpuVorticityResult { curl, force }
    }
}

/// Reads back a mapped staging buffer of [`GpuVec3`] and unpacks it into a
/// `Vec<Vec3>` of exactly `count` entries.
fn read_field(stage: &wgpu::Buffer, count: usize) -> Vec<Vec3> {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let packed = bytemuck::cast_slice::<u8, GpuVec3>(&view).to_vec();
    drop(view);
    stage.unmap();
    debug_assert_eq!(packed.len(), count);
    packed.into_iter().map(GpuVec3::to_vec3).collect()
}
