//! `wgpu` compute twin of the Fedkiw vorticity-confinement turbulence force
//! ([`confinement_force`](prism_render_architecture::particle::vorticity_confinement::confinement_force),
//! design section 10, "涡量约束").
//!
//! Large-scale numerical advection dissipates the small rotational structure
//! that reads as turbulent detail in fire, smoke and dust. Fedkiw's
//! vorticity-confinement force reinjects it by locating where the vorticity
//! magnitude peaks and pushing the flow back toward those cores. Over a sampled
//! velocity grid the recipe is: vorticity `omega = curl(v)` by central
//! difference, its scalar magnitude field `|omega|`, the normalized gradient
//! `N = grad(|omega|) / |grad(|omega|)|`, and the force
//! `f = epsilon * dx * (N x omega)`.
//!
//! The `CPU` golden
//! [`particle::vorticity_confinement`](prism_render_architecture::particle::vorticity_confinement)
//! owns that math over a [`VelocityGrid`]; [`GpuVorticityConfinement`] is the
//! on-device twin that runs one thread per grid cell and returns, for every
//! cell, both the intermediate vorticity `omega` and the final confinement
//! force. A passing real-device parity test is therefore direct evidence the
//! ported kernel differentiates the same grid the reference does, not merely
//! that its shader compiles.
//!
//! # Step-for-step parity
//!
//! The kernel mirrors the reference stencil exactly: the same clamped
//! central-difference neighbor selection (`hi = min(c + 1, dim - 1)`,
//! `lo = c.saturating_sub(1)`), the same `span * dx` denominator with the same
//! `MIN_LENGTH` divide-by-zero guard, the same `omega` component order
//! `(dvdy.z - dvdz.y, dvdz.x - dvdx.z, dvdx.y - dvdy.x)`, the same
//! magnitude-gradient stencil evaluated on neighbor curls, the same
//! normalize-or-zero rule, and the same right-handed `N x omega` scaled by
//! `epsilon * dx`. Row-major (`x`-fastest) indexing matches
//! [`VelocityGrid::index`](prism_render_architecture::particle::vorticity_confinement::VelocityGrid::index).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `sqrt`, `+ - * /` and unsigned integer compares — with no `sin`, `cos`,
//! `exp`, `log`, `pow` or optional device feature, so it runs unmodified on
//! Metal, Vulkan and DX12. The only non-integer primitive is `sqrt` (vector
//! length and gradient normalization), exactly as in the reference, and every
//! reciprocal is guarded by a `MIN_LENGTH` clamp so no divide ever hits a
//! vanishing denominator.
//!
//! # Correctness model
//!
//! The whole pipeline is finite-difference algebra with no transcendental call
//! and no reorderable reduction, so `CPU` and `GPU` evaluate the same closed
//! form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test asserts a
//! tolerance (`abs_diff <= 1e-5` or `rel_diff <= 1e-5`) tight enough to catch a
//! genuinely wrong port yet loose enough to admit legal fused multiply-add
//! contraction across the six neighbor curls that feed each gradient.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard Fedkiw-Stam-Jensen vorticity confinement plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::vorticity_confinement::{Vec3, VelocityGrid};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` vorticity-confinement kernel, embedded inline so
/// the twin ships as a single source file. Mirrors the `CPU` stencil exactly;
/// see the module documentation for the algorithm.
const VORTICITY_CONFINEMENT_WGSL: &str = r#"
// Vorticity-confinement twin: one thread per grid cell computes the vorticity
// `omega = curl(v)` by clamped central difference, the normalized gradient of
// `|omega|`, and the Fedkiw confinement force `epsilon * dx * (N x omega)`,
// writing both `omega` and the force per cell. It mirrors the CPU golden
// `particle::vorticity_confinement`, uses only the portable core-WGSL subset
// (min/max/sqrt and + - * /), and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Fedkiw-Stam-Jensen vorticity confinement; no Unreal
// Engine source or derived code.

struct Params {
    // Grid cell counts along x, y, z.
    nx: u32,
    ny: u32,
    nz: u32,
    // Total cell count `nx * ny * nz` (one thread each).
    cell_count: u32,
    // Uniform grid spacing `dx`, shared by all three axes.
    cell_size: f32,
    // Confinement strength `epsilon`.
    epsilon: f32,
    // Padding to a 32-byte std430 struct.
    pad0: u32,
    pad1: u32,
}

// One cell result. 32-byte std430 stride: the vorticity triple plus a pad word,
// then the force triple plus a pad word, matching the host `GpuVorticity`.
struct Vorticity {
    curl_x: f32,
    curl_y: f32,
    curl_z: f32,
    pad0: f32,
    force_x: f32,
    force_y: f32,
    force_z: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> results: array<Vorticity>;

// Vectors shorter than this are treated as zero-length, so normalization and
// finite-difference denominators never divide by a vanishing quantity. Matches
// the reference `MIN_LENGTH`.
const MIN_LENGTH: f32 = 1.0e-12;

// Clamps a coordinate into [0, dim), guarding an empty axis, like the reference
// `clamp_index`.
fn clamp_index(v: u32, dim: u32) -> u32 {
    if (dim == 0u) {
        return 0u;
    }
    return min(v, dim - 1u);
}

// Samples the velocity at (i, j, k), clamping to the grid boundary, like the
// reference `VelocityGrid::sample`.
fn sample_vel(i: u32, j: u32, k: u32) -> vec3<f32> {
    let ci = clamp_index(i, params.nx);
    let cj = clamp_index(j, params.ny);
    let ck = clamp_index(k, params.nz);
    let idx = ci + params.nx * (cj + params.ny * ck);
    let v = velocities[idx];
    return vec3<f32>(v.x, v.y, v.z);
}

// Euclidean length, the only place sqrt is used.
fn vlen(v: vec3<f32>) -> f32 {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// The axis span (hi - lo) of the clamped central-difference stencil at coord
// component `c` on an axis of extent `dim`, returned as both the stencil's
// hi/lo indices (packed) and the finite-difference denominator. WGSL lacks
// multiple return values, so this is split into the small helpers below.
fn stencil_hi(c: u32, dim: u32) -> u32 {
    var dim_m1 = 0u;
    if (dim != 0u) {
        dim_m1 = dim - 1u;
    }
    return min(c + 1u, dim_m1);
}

fn stencil_lo(c: u32) -> u32 {
    if (c == 0u) {
        return 0u;
    }
    return c - 1u;
}

// The `span * dx` denominator for a stencil whose hi/lo indices are `hi`/`lo`.
// The span is clamped to zero beyond 255 to mirror the reference
// `u8::try_from(span).unwrap_or(0)`; a real central-difference span is at most
// two, so the clamp only guards pathological inputs.
fn stencil_denom(hi: u32, lo: u32) -> f32 {
    var span = 0u;
    if (hi > lo) {
        span = hi - lo;
    }
    var span_scalar = f32(span);
    if (span > 255u) {
        span_scalar = 0.0;
    }
    return span_scalar * params.cell_size;
}

// The partial derivative d v / d(axis) at (i, j, k) by clamped central
// difference, like the reference `velocity_partial`.
fn velocity_partial(i: u32, j: u32, k: u32, axis: u32) -> vec3<f32> {
    var dim = params.nx;
    var c = i;
    if (axis == 1u) {
        dim = params.ny;
        c = j;
    }
    if (axis == 2u) {
        dim = params.nz;
        c = k;
    }
    let hi_c = stencil_hi(c, dim);
    let lo_c = stencil_lo(c);
    let denom = stencil_denom(hi_c, lo_c);

    var hi_i = i;
    var hi_j = j;
    var hi_k = k;
    var lo_i = i;
    var lo_j = j;
    var lo_k = k;
    if (axis == 0u) {
        hi_i = hi_c;
        lo_i = lo_c;
    }
    if (axis == 1u) {
        hi_j = hi_c;
        lo_j = lo_c;
    }
    if (axis == 2u) {
        hi_k = hi_c;
        lo_k = lo_c;
    }

    let diff = sample_vel(hi_i, hi_j, hi_k) - sample_vel(lo_i, lo_j, lo_k);
    if (denom > MIN_LENGTH) {
        return diff * (1.0 / denom);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Vorticity omega = curl(v) at (i, j, k) via central differences, like the
// reference `curl_at`.
fn curl_at(i: u32, j: u32, k: u32) -> vec3<f32> {
    let dvdx = velocity_partial(i, j, k, 0u);
    let dvdy = velocity_partial(i, j, k, 1u);
    let dvdz = velocity_partial(i, j, k, 2u);
    return vec3<f32>(
        dvdy.z - dvdz.y,
        dvdz.x - dvdx.z,
        dvdx.y - dvdy.x
    );
}

// The partial derivative d|omega| / d(axis) at (i, j, k), matching the clamped
// central-difference stencil used for the velocity but evaluated on neighbor
// curl magnitudes, like the reference `magnitude_partial`.
fn magnitude_partial(i: u32, j: u32, k: u32, axis: u32) -> f32 {
    var dim = params.nx;
    var c = i;
    if (axis == 1u) {
        dim = params.ny;
        c = j;
    }
    if (axis == 2u) {
        dim = params.nz;
        c = k;
    }
    let hi_c = stencil_hi(c, dim);
    let lo_c = stencil_lo(c);
    let denom = stencil_denom(hi_c, lo_c);

    var hi_i = i;
    var hi_j = j;
    var hi_k = k;
    var lo_i = i;
    var lo_j = j;
    var lo_k = k;
    if (axis == 0u) {
        hi_i = hi_c;
        lo_i = lo_c;
    }
    if (axis == 1u) {
        hi_j = hi_c;
        lo_j = lo_c;
    }
    if (axis == 2u) {
        hi_k = hi_c;
        lo_k = lo_c;
    }

    let mag_hi = vlen(curl_at(hi_i, hi_j, hi_k));
    let mag_lo = vlen(curl_at(lo_i, lo_j, lo_k));
    if (denom > MIN_LENGTH) {
        return (mag_hi - mag_lo) / denom;
    }
    return 0.0;
}

// Returns the unit vector, or zero for a (near-)zero vector, like the reference
// `normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = vlen(v);
    if (len > MIN_LENGTH) {
        return v * (1.0 / len);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Right-handed cross product a x b, component order matching the reference
// `Vec3::cross`.
fn cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

@compute @workgroup_size(64)
fn confine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.cell_count) {
        return;
    }
    // Recover the row-major (x-fastest) coordinate from the flat index.
    let i = idx % params.nx;
    let rem = idx / params.nx;
    let j = rem % params.ny;
    let k = rem / params.ny;

    let omega = curl_at(i, j, k);
    let grad = vec3<f32>(
        magnitude_partial(i, j, k, 0u),
        magnitude_partial(i, j, k, 1u),
        magnitude_partial(i, j, k, 2u)
    );
    let normal = normalize_or_zero(grad);
    let force = cross3(normal, omega) * (params.epsilon * params.cell_size);

    var out: Vorticity;
    out.curl_x = omega.x;
    out.curl_y = omega.y;
    out.curl_z = omega.z;
    out.pad0 = 0.0;
    out.force_x = force.x;
    out.force_y = force.y;
    out.force_z = force.z;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// One cell's vorticity-confinement outputs: the intermediate vorticity `omega`
/// and the final confinement force at that cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VorticityResult {
    /// Vorticity `omega = curl(v)` at the cell.
    pub curl: Vec3,
    /// Confinement force `epsilon * dx * (N x omega)` at the cell.
    pub force: Vec3,
}

/// Uniform parameters for one confinement dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`VORTICITY_CONFINEMENT_WGSL`]: four index words then
/// two `f32` words and two pad words — `32` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Cell count along `x`.
    nx: u32,
    /// Cell count along `y`.
    ny: u32,
    /// Cell count along `z`.
    nz: u32,
    /// Total cell count `nx * ny * nz`.
    cell_count: u32,
    /// Uniform grid spacing `dx`.
    cell_size: f32,
    /// Confinement strength `epsilon`.
    epsilon: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One velocity sample as uploaded. `16`-byte `std430` stride: the `xyz`
/// components plus one pad word, matching the one-padded-`vec4`-per-cell layout
/// the reference documents through
/// [`VelocityGrid::velocity_buffer_bytes`](prism_render_architecture::particle::vorticity_confinement::VelocityGrid::velocity_buffer_bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVelocity {
    /// The `x` component.
    x: f32,
    /// The `y` component.
    y: f32,
    /// The `z` component.
    z: f32,
    /// Padding word.
    pad: f32,
}

/// One cell result as read back. `32`-byte `std430` stride matching `Vorticity`
/// in the shader: the vorticity triple plus a pad word, then the force triple
/// plus a pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVorticity {
    /// Vorticity `x`.
    curl_x: f32,
    /// Vorticity `y`.
    curl_y: f32,
    /// Vorticity `z`.
    curl_z: f32,
    /// Padding word.
    pad0: f32,
    /// Force `x`.
    force_x: f32,
    /// Force `y`.
    force_y: f32,
    /// Force `z`.
    force_z: f32,
    /// Padding word.
    pad1: f32,
}

/// A compiled, reusable vorticity-confinement pipeline.
pub struct GpuVorticityConfinement {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuVorticityConfinement {
    /// Compiles the vorticity-confinement kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuVorticityConfinement {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_vorticity_confinement"),
            source: ShaderSource::Wgsl(VORTICITY_CONFINEMENT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_vorticity_confinement_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_vorticity_confinement_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_vorticity_confinement_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("confine"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuVorticityConfinement {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the vorticity and confinement force for every cell of `grid`
    /// at confinement strength `epsilon`, returning one [`VorticityResult`] per
    /// cell in row-major (`x`-fastest) order.
    ///
    /// The returned `curl` for cell `(i, j, k)` equals
    /// [`VelocityGrid::curl_at`](prism_render_architecture::particle::vorticity_confinement::VelocityGrid::curl_at)
    /// and the returned `force` equals
    /// [`confinement_force`](prism_render_architecture::particle::vorticity_confinement::confinement_force)
    /// to within the tolerance documented on this module. An empty grid yields
    /// an empty result — a storage buffer cannot be zero-sized, so it is handled
    /// by an early return. Missing samples (a `data` slice shorter than the cell
    /// count) read as zero, matching the reference `sample`.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        grid: &VelocityGrid,
        epsilon: f32,
    ) -> Vec<VorticityResult> {
        let count = grid.cell_count();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();
        let [nx, ny, nz] = grid.dims;

        // Build a dense, cell-count-length velocity buffer. Missing samples stay
        // zero and extra samples are ignored, matching the clamped,
        // `get`-with-default reads of the reference.
        let mut velocities = vec![GpuVelocity::zeroed(); count];
        for (slot, v) in velocities.iter_mut().zip(grid.data.iter()) {
            slot.x = v.x;
            slot.y = v.y;
            slot.z = v.z;
        }

        let gpu_params = Params {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            cell_count: count as u32,
            cell_size: grid.cell_size,
            epsilon,
            pad0: 0,
            pad1: 0,
        };

        let out_bytes = (count as u64) * (size_of::<GpuVorticity>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vorticity_confinement_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let velocities_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_vorticity_confinement_velocities"),
            contents: bytemuck::cast_slice(&velocities),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_vorticity_confinement_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_vorticity_confinement_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_vorticity_confinement_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: velocities_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_vorticity_confinement_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_vorticity_confinement_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per cell, in workgroups of 64 (the kernel's size).
            let groups = (count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_results = bytemuck::cast_slice::<u8, GpuVorticity>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), count);

        gpu_results
            .into_iter()
            .map(|r| VorticityResult {
                curl: Vec3::new(r.curl_x, r.curl_y, r.curl_z),
                force: Vec3::new(r.force_x, r.force_y, r.force_z),
            })
            .collect()
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
