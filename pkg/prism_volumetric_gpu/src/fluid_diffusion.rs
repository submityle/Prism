//! `wgpu` compute twin of the `CPU` golden viscous-diffusion relaxation solver
//! ([`viscous_diffuse`](prism_render_architecture::particle::fluid_diffusion::viscous_diffuse),
//! design §10).
//!
//! Step two of the stable-fluids pipeline diffuses momentum implicitly: it
//! solves `(I − α·L)·v' = v` for the post-advection velocity field `v`, where
//! `L` is the `7`-point discrete Laplacian and `α = ν·dt/h²` is the
//! dimensionless diffusion number
//! ([`diffusion_alpha`](prism_render_architecture::particle::fluid_diffusion::diffusion_alpha)).
//! The symmetric, diagonally dominant system is relaxed with a fixed number of
//! damped `Jacobi` sweeps, each of which updates every voxel to
//! `(v + α·Σ₆ neighbors) / (1 + diagonal·α)`. The `CPU` golden
//! [`fluid_diffusion`](prism_render_architecture::particle::fluid_diffusion)
//! module owns that math; [`GpuFluidDiffusion`] is the on-device twin, validated
//! against that reference so a passing real-device parity test is direct
//! evidence the ported kernel relaxes the same field, not merely that its shader
//! compiles.
//!
//! # Algorithm
//!
//! The twin reproduces the reference sweep for sweep. One thread owns one voxel.
//! Each dispatch is one `Jacobi` sweep that reads the previous iterate
//! (`current`) and the constant right-hand side (`source`) and writes the next
//! iterate; the host ping-pongs two storage buffers so `iterations` sweeps run
//! back to back, exactly mirroring the host loop in
//! [`viscous_diffuse`](prism_render_architecture::particle::fluid_diffusion::viscous_diffuse).
//! The six axis-aligned face neighbors are summed in the identical order the
//! scalar reference uses (`+x`, `−x`, `+y`, `−y`, `+z`, `−z`), and the final
//! update multiplies by the reciprocal `1 / (1 + diagonal·α)` just as the
//! reference does, so the two evaluate the same arithmetic in the same order.
//!
//! # Boundaries
//!
//! Both wall models the reference supports are carried in the shared uniform.
//! Under [`DiffusionBoundary::Fixed`](prism_render_architecture::particle::fluid_diffusion::DiffusionBoundary::Fixed)
//! an out-of-grid neighbor is the zero wall velocity (homogeneous Dirichlet), so
//! the implicit diagonal keeps all six faces. Under
//! [`DiffusionBoundary::Free`](prism_render_architecture::particle::fluid_diffusion::DiffusionBoundary::Free)
//! a missing neighbor mirrors the center cell (homogeneous Neumann), so the
//! diagonal drops to the count of neighbors that actually exist. The kernel
//! encodes this exactly as the reference does: it counts live neighbors and
//! picks the diagonal accordingly.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer index
//! arithmetic plus `+ − × ÷` on `f32` vectors — with no `sin`, `cos`, `exp`,
//! `log`, `pow` or optional device feature, so it runs unmodified on Metal,
//! Vulkan and `DX12`. `Jacobi` relaxation is multiply-add plus one reciprocal
//! per voxel, so it needs nothing beyond that subset.
//!
//! # Correctness model
//!
//! The sweep contains no transcendental call, so `CPU` and `GPU` evaluate the
//! same closed-form algebra in the same order. They are not bit-exact: a `GPU`
//! may fuse a multiply-add the scalar reference leaves separate, perturbing the
//! low mantissa bits by a few `ULP`, and that perturbation compounds across the
//! iterated sweeps. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a swapped neighbor, a missing diagonal term, a dropped boundary
//! case) yet loose enough to admit legal fused multiply-add contraction summed
//! over the sweep count.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Stam` stable-fluids implicit viscous diffusion by
//! damped `Jacobi` relaxation plus `wgpu` compute dispatch; no Unreal Engine
//! source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::Vec3;
use prism_render_architecture::particle::fluid::GridResolution;
use prism_render_architecture::particle::fluid_diffusion::{
    DiffusionBoundary, DiffusionParams, diffusion_alpha,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The uniform tag selecting the no-slip (`Dirichlet`) wall model, matching
/// [`DiffusionBoundary::Fixed`](prism_render_architecture::particle::fluid_diffusion::DiffusionBoundary::Fixed).
const BOUNDARY_FIXED: u32 = 0;

/// The uniform tag selecting the zero-gradient (`Neumann`) wall model, matching
/// [`DiffusionBoundary::Free`](prism_render_architecture::particle::fluid_diffusion::DiffusionBoundary::Free).
const BOUNDARY_FREE: u32 = 1;

/// The number of threads per workgroup. `64` is a portable, warp-friendly size
/// used across this crate's kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` relaxation kernel, embedded inline so the twin ships
/// as a single source file. Mirrors the `CPU` sweep exactly; see the module
/// documentation for the algorithm.
const FLUID_DIFFUSION_WGSL: &str = r#"
// Viscous-diffusion relaxation twin: one thread per voxel performs one damped
// Jacobi sweep of the implicit system `(I - alpha*L)*v' = source`, updating its
// cell to `(source + alpha * neighbor_sum) * (1 / (1 + diagonal * alpha))`. It
// mirrors the CPU golden `particle::fluid_diffusion::diffuse_relax_sweep`, uses
// only the portable core-WGSL subset (integer index math plus + - * / on
// vectors), and takes no optional feature, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: standard Stam stable-fluids implicit viscous diffusion; no Unreal
// Engine source or derived code.

struct Params {
    // Grid extents in voxels along each axis.
    nx: u32,
    ny: u32,
    nz: u32,
    // Wall model: 0 = Fixed (Dirichlet, diagonal is always six faces),
    // 1 = Free (Neumann, diagonal is the live-neighbor count).
    boundary: u32,
    // The diffusion number alpha = nu*dt/h^2, precomputed on the host so the
    // GPU consumes the identical f32 the CPU reference formed.
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The constant right-hand side (the pre-diffusion velocity field).
@group(0) @binding(1) var<storage, read> source: array<vec4<f32>>;
// The previous Jacobi iterate whose neighbors this sweep reads.
@group(0) @binding(2) var<storage, read> current: array<vec4<f32>>;
// The next iterate this sweep writes.
@group(0) @binding(3) var<storage, read_write> out_field: array<vec4<f32>>;

// Row-major linear index, matching `GridResolution::linear_index`:
// (z * ny + y) * nx + x.
fn lin(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.ny + y) * params.nx + x;
}

@compute @workgroup_size(64)
fn relax(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
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

    // Fixed walls keep all six faces (the missing ones are the zero Dirichlet
    // wall); free walls use only the live neighbors (zero-gradient Neumann).
    var diagonal = 6.0;
    if (params.boundary == 1u) {
        diagonal = f32(live_count);
    }

    let denom = 1.0 + diagonal * params.alpha;
    // Multiply by the reciprocal, matching the reference's `scale(1.0 / denom)`.
    let inv = 1.0 / denom;
    out_field[idx] = (source[idx] + sum * params.alpha) * inv;
}
"#;

/// Uniform parameters for one relaxation dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`FLUID_DIFFUSION_WGSL`]: the three grid extents and the
/// boundary tag, the precomputed diffusion number `alpha`, then three pad words
/// — `32` bytes total with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Grid extent in voxels along `x`.
    nx: u32,
    /// Grid extent in voxels along `y`.
    ny: u32,
    /// Grid extent in voxels along `z`.
    nz: u32,
    /// Wall-model tag ([`BOUNDARY_FIXED`] or [`BOUNDARY_FREE`]).
    boundary: u32,
    /// The diffusion number `alpha = nu*dt/h^2`, precomputed on the host.
    alpha: f32,
    /// Padding to keep the struct a multiple of `16` bytes.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One velocity sample as uploaded and read back. `16`-byte `std430` stride
/// matching `array<vec4<f32>>` in the shader: the three velocity components plus
/// one pad lane that stays zero.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVelocity {
    /// X velocity component.
    x: f32,
    /// Y velocity component.
    y: f32,
    /// Z velocity component.
    z: f32,
    /// Padding lane, held at zero so it never perturbs the arithmetic.
    pad: f32,
}

impl GpuVelocity {
    /// Packs a [`Vec3`] into the padded device layout.
    fn from_vec3(v: Vec3) -> Self {
        GpuVelocity {
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

/// The outcome of a `GPU` viscous-diffusion solve, mirroring the `CPU`
/// [`DiffusionResult`](prism_render_architecture::particle::fluid_diffusion::DiffusionResult).
#[derive(Clone, Debug, PartialEq)]
pub struct GpuDiffusionResult {
    /// The diffused velocity field, row-major, one [`Vec3`] per voxel.
    pub velocity: Vec<Vec3>,
    /// How many `Jacobi` sweeps actually ran.
    pub iterations_run: u32,
}

/// A compiled, reusable viscous-diffusion relaxation pipeline.
pub struct GpuFluidDiffusion {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFluidDiffusion {
    /// Compiles the relaxation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFluidDiffusion {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fluid_diffusion"),
            source: ShaderSource::Wgsl(FLUID_DIFFUSION_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("relax"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFluidDiffusion {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves the implicit viscous-diffusion system with `params.iterations`
    /// damped `Jacobi` sweeps, returning the diffused field in input order.
    ///
    /// The returned field equals
    /// [`viscous_diffuse`](prism_render_architecture::particle::fluid_diffusion::viscous_diffuse)`(source, res, params)`
    /// to within the tolerance documented on this module. The grid is empty, or
    /// `source` carries fewer than `res.voxel_count()` samples, yields an empty
    /// field and zero iterations, matching the reference's degenerate-input
    /// guard. A zero iteration count returns an unmodified copy of the field
    /// (no dispatch), also matching the reference.
    #[must_use]
    pub fn diffuse(
        &self,
        ctx: &GpuContext,
        source: &[Vec3],
        res: GridResolution,
        params: DiffusionParams,
    ) -> GpuDiffusionResult {
        let count = res.voxel_count() as usize;
        if count == 0 || source.len() < count {
            return GpuDiffusionResult {
                velocity: Vec::new(),
                iterations_run: 0,
            };
        }
        // A zero-sweep solve is the identity: hand back the clamped source copy
        // without touching the device, exactly as the reference does.
        if params.iterations == 0 {
            return GpuDiffusionResult {
                velocity: source[..count].to_vec(),
                iterations_run: 0,
            };
        }

        let device = ctx.device();

        let boundary = match params.boundary {
            DiffusionBoundary::Fixed => BOUNDARY_FIXED,
            DiffusionBoundary::Free => BOUNDARY_FREE,
        };
        let gpu_params = Params {
            nx: res.nx,
            ny: res.ny,
            nz: res.nz,
            boundary,
            // The host forms alpha with the reference helper so the GPU reads
            // the identical f32 value, not a re-derived one.
            alpha: diffusion_alpha(params),
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let packed: Vec<GpuVelocity> = source[..count]
            .iter()
            .map(|&v| GpuVelocity::from_vec3(v))
            .collect();
        let field_bytes = (count as u64) * (size_of::<GpuVelocity>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let source_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_source"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        // Ping-pong iterate buffers. Buffer A starts holding the source copy so
        // the first sweep reads the same initial iterate the reference seeds.
        let buf_a = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_iterate_a"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let buf_b = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_iterate_b"),
            size: field_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_stage"),
            size: field_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group AB reads A and writes B; BA reads B and writes A.
        let bind_ab = self.bind_group(device, &params_buf, &source_buf, &buf_a, &buf_b);
        let bind_ba = self.bind_group(device, &params_buf, &source_buf, &buf_b, &buf_a);

        let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_encoder"),
        });
        for sweep in 0..params.iterations {
            let bind = if sweep % 2 == 0 { &bind_ab } else { &bind_ba };
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fluid_diffusion_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // Sweep `i` (zero-based) writes B when `i` is even and A when odd, so the
        // final write lands in B for an odd iteration count and A for an even
        // one.
        let final_buf = if params.iterations % 2 == 1 {
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
        let gpu_field = bytemuck::cast_slice::<u8, GpuVelocity>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(gpu_field.len(), count);

        GpuDiffusionResult {
            velocity: gpu_field.into_iter().map(GpuVelocity::to_vec3).collect(),
            iterations_run: params.iterations,
        }
    }

    /// Builds a relaxation bind group binding the shared uniform and source
    /// alongside the chosen `current` (read) and `out` (write) iterate buffers.
    fn bind_group(
        &self,
        device: &wgpu::Device,
        params_buf: &wgpu::Buffer,
        source_buf: &wgpu::Buffer,
        current: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fluid_diffusion_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: source_buf.as_entire_binding(),
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
