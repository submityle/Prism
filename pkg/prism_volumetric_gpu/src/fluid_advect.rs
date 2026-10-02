//! `wgpu` compute twin of the `CPU` golden semi-Lagrangian advection
//! pure-function cluster
//! ([`fluid`](prism_render_architecture::particle::fluid), design §10).
//!
//! Step one of the stable-fluids pipeline moves fields along the flow. The `CPU`
//! golden
//! [`fluid`](prism_render_architecture::particle::fluid) module owns the math for
//! the five pure functions this twin reproduces, and [`GpuFluidAdvect`] is the
//! on-device twin validated against that reference so a passing real-device
//! parity test is direct evidence the ported kernel traces, samples and corrects
//! the same field, not merely that its shader compiles.
//!
//! # What is twinned
//!
//! Each thread owns one query against a shared velocity field and produces, in a
//! single [`GpuAdvectResult`], the output of five golden functions evaluated at
//! that query:
//!
//! - [`backtrace`](GpuAdvectResult::backtrace) is the semi-Lagrangian back-trace
//!   `pos − velocity·dt`, twinning
//!   [`semi_lagrangian_backtrace`](prism_render_architecture::particle::fluid::semi_lagrangian_backtrace).
//! - [`sampled`](GpuAdvectResult::sampled) is the trilinearly interpolated
//!   velocity at `pos`, twinning
//!   [`sample_velocity_field`](prism_render_architecture::particle::fluid::sample_velocity_field)
//!   (which itself exercises
//!   [`trilinear_weights`](prism_render_architecture::particle::fluid::trilinear_weights)
//!   and
//!   [`trilinear_sample`](prism_render_architecture::particle::fluid::trilinear_sample)).
//! - [`advected`](GpuAdvectResult::advected) is `pos + sampled·dt`, twinning
//!   [`advect_particle_in_field`](prism_render_architecture::particle::fluid::advect_particle_in_field).
//! - [`maccormack`](GpuAdvectResult::maccormack) is the `MacCormack` correction
//!   `forward + 0.5·(original − back)`, twinning
//!   [`maccormack_corrected`](prism_render_architecture::particle::fluid::maccormack_corrected).
//! - [`weights`](GpuAdvectResult::weights) are the eight trilinear corner weights
//!   at the in-cell fraction of `pos`, a direct twin of
//!   [`trilinear_weights`](prism_render_architecture::particle::fluid::trilinear_weights).
//!
//! The eight-corner fetch order, the per-axis `clamp-to-edge` addressing, the
//! `floor`-based base-voxel split and the weighted-sum accumulation order all
//! match the reference tap for tap, so the two evaluate the same closed-form
//! algebra in the same order.
//!
//! # Correctness model
//!
//! Every output is a multiply-add plus the sampler's `floor` and integer
//! `clamp`; there is no transcendental call, so `CPU` and `GPU` evaluate the same
//! closed form. They are not bit-exact: a `GPU` may fuse a multiply-add the
//! scalar reference leaves separate, perturbing the low mantissa bits by a few
//! `ULP`. The parity test asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`), tight enough to catch a genuinely wrong port (a swapped
//! corner, a dropped offset, a missing edge clamp) yet loose enough to admit
//! legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! An empty query batch short-circuits on the host with no dispatch (a storage
//! buffer cannot be zero-sized). An empty or short field (fewer than
//! `res.voxel_count()` samples, or a zero-voxel grid) marks the field invalid in
//! the shared uniform: [`sampled`](GpuAdvectResult::sampled) is
//! [`Vec3::ZERO`](prism_render_architecture::particle::Vec3::ZERO) and
//! [`advected`](GpuAdvectResult::advected) is the untouched position, exactly as
//! the reference guard returns. The field storage buffer still holds one
//! placeholder velocity the kernel never reads because the invalid flag fires
//! first.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `floor`,
//! `min`, `+ − × ÷` on `f32` vectors and unsigned index arithmetic — with no
//! `sin`, `cos`, `exp`, `pow`, optional device feature or `u64`, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid` 的半拉格朗日
//! 平流纯函数簇；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

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

/// The portable core-`WGSL` advection kernel, embedded inline so the twin ships
/// as a single source file. One thread owns one query; see the module
/// documentation for the algorithm.
const FLUID_ADVECT_WGSL: &str = r#"
// Semi-Lagrangian advection twin: one thread owns one query against a shared
// velocity field and reproduces five CPU golden functions in one result record
// (back-trace, field sample, in-field advect, MacCormack correction, and the
// eight trilinear corner weights). It uses only the portable core-WGSL subset
// (integer index math plus + - * / on f32 vectors, floor, min, clamp), takes no
// optional feature, and runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 particle::fluid 的半拉格朗日平流纯函数簇；无第三方引擎源码或衍生代码。

struct Params {
    // Grid extents in voxels along each axis.
    nx: u32,
    ny: u32,
    nz: u32,
    // Non-zero when the field carries at least voxel_count samples; zero marks a
    // degenerate field sampled as the zero vector.
    field_valid: u32,
    // Number of valid queries; threads past this short-circuit.
    query_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    // The advection time step, shared by every query.
    dt: f32,
    pad3: f32,
    pad4: f32,
    pad5: f32,
}

struct Query {
    // Sample / integration position (xyz; w unused).
    pos: vec4<f32>,
    // Velocity used by the semi-Lagrangian back-trace (xyz; w unused).
    velocity: vec4<f32>,
    // MacCormack forward-advected value (xyz; w unused).
    mac_forward: vec4<f32>,
    // MacCormack original value (xyz; w unused).
    mac_original: vec4<f32>,
    // MacCormack back-advected value (xyz; w unused).
    mac_back: vec4<f32>,
}

struct AdvectResult {
    // Semi-Lagrangian back-trace position (xyz; w zero).
    backtrace: vec4<f32>,
    // Trilinearly sampled velocity at pos (xyz; w zero).
    sampled: vec4<f32>,
    // In-field advected position pos + sampled*dt (xyz; w zero).
    advected: vec4<f32>,
    // MacCormack-corrected value (xyz; w zero).
    maccormack: vec4<f32>,
    // Trilinear corner weights 0..3.
    weights_lo: vec4<f32>,
    // Trilinear corner weights 4..7.
    weights_hi: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
// The row-major velocity field, one vec4 per voxel (xyz used, w pad).
@group(0) @binding(1) var<storage, read> field: array<vec4<f32>>;
// The per-thread query batch.
@group(0) @binding(2) var<storage, read> queries: array<Query>;
// The per-thread result batch.
@group(0) @binding(3) var<storage, read_write> results: array<AdvectResult>;

// Row-major linear index, matching `GridResolution::linear_index`:
// (z * ny + y) * nx + x.
fn lin(x: u32, y: u32, z: u32) -> u32 {
    return (z * params.ny + y) * params.nx + x;
}

@compute @workgroup_size(64)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.query_count) {
        return;
    }

    let q = queries[idx];
    let pos = q.pos.xyz;
    let velocity = q.velocity.xyz;

    // Semi-Lagrangian back-trace: pos - velocity*dt (multiply-add only).
    let backtrace = pos - velocity * params.dt;

    var sampled = vec3<f32>(0.0, 0.0, 0.0);
    var w0 = 0.0;
    var w1 = 0.0;
    var w2 = 0.0;
    var w3 = 0.0;
    var w4 = 0.0;
    var w5 = 0.0;
    var w6 = 0.0;
    var w7 = 0.0;

    if (params.field_valid != 0u) {
        let max_x = params.nx - 1u;
        let max_y = params.ny - 1u;
        let max_z = params.nz - 1u;
        // Clamp the position into the grid, exactly as the reference does.
        let cx = clamp(pos.x, 0.0, f32(max_x));
        let cy = clamp(pos.y, 0.0, f32(max_y));
        let cz = clamp(pos.z, 0.0, f32(max_z));
        let x0f = floor(cx);
        let y0f = floor(cy);
        let z0f = floor(cz);
        // In-cell fractions on [0, 1] per axis.
        let fx = cx - x0f;
        let fy = cy - y0f;
        let fz = cz - z0f;
        // Complementary fractions (1 - f), matching the reference's (gx, gy, gz).
        let gx = 1.0 - fx;
        let gy = 1.0 - fy;
        let gz = 1.0 - fz;
        // Trilinear corner weights in the reference bitmask order
        // (bit 0 = X, bit 1 = Y, bit 2 = Z).
        w0 = gx * gy * gz;
        w1 = fx * gy * gz;
        w2 = gx * fy * gz;
        w3 = fx * fy * gz;
        w4 = gx * gy * fz;
        w5 = fx * gy * fz;
        w6 = gx * fy * fz;
        w7 = fx * fy * fz;

        let x0 = u32(x0f);
        let y0 = u32(y0f);
        let z0 = u32(z0f);
        let x1 = min(x0 + 1u, max_x);
        let y1 = min(y0 + 1u, max_y);
        let z1 = min(z0 + 1u, max_z);

        // Eight corner velocities in the reference fetch order
        // [000, 100, 010, 110, 001, 101, 011, 111].
        let c0 = field[lin(x0, y0, z0)].xyz;
        let c1 = field[lin(x1, y0, z0)].xyz;
        let c2 = field[lin(x0, y1, z0)].xyz;
        let c3 = field[lin(x1, y1, z0)].xyz;
        let c4 = field[lin(x0, y0, z1)].xyz;
        let c5 = field[lin(x1, y0, z1)].xyz;
        let c6 = field[lin(x0, y1, z1)].xyz;
        let c7 = field[lin(x1, y1, z1)].xyz;

        // Weighted sum folded in corner order 0..7, matching `trilinear_sample`.
        var acc = vec3<f32>(0.0, 0.0, 0.0);
        acc = acc + c0 * w0;
        acc = acc + c1 * w1;
        acc = acc + c2 * w2;
        acc = acc + c3 * w3;
        acc = acc + c4 * w4;
        acc = acc + c5 * w5;
        acc = acc + c6 * w6;
        acc = acc + c7 * w7;
        sampled = acc;
    }

    // In-field advect: pos + sampled*dt (multiply-add only).
    let advected = pos + sampled * params.dt;
    // MacCormack correction: forward + 0.5*(original - back).
    let maccormack = q.mac_forward.xyz + (q.mac_original.xyz - q.mac_back.xyz) * 0.5;

    results[idx].backtrace = vec4<f32>(backtrace, 0.0);
    results[idx].sampled = vec4<f32>(sampled, 0.0);
    results[idx].advected = vec4<f32>(advected, 0.0);
    results[idx].maccormack = vec4<f32>(maccormack, 0.0);
    results[idx].weights_lo = vec4<f32>(w0, w1, w2, w3);
    results[idx].weights_hi = vec4<f32>(w4, w5, w6, w7);
}
"#;

/// Uniform parameters for one advection dispatch. `repr(C)` `std140` layout
/// matching `Params` in [`FLUID_ADVECT_WGSL`]: the three grid extents, the
/// field-valid flag, the query count, three pad words, the time step and three
/// pad words — `48` bytes total.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Grid extent in voxels along `x`.
    nx: u32,
    /// Grid extent in voxels along `y`.
    ny: u32,
    /// Grid extent in voxels along `z`.
    nz: u32,
    /// Non-zero when the field is large enough to sample.
    field_valid: u32,
    /// Number of valid queries in the batch.
    query_count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
    /// The advection time step shared by every query.
    dt: f32,
    /// Padding word.
    pad3: f32,
    /// Padding word.
    pad4: f32,
    /// Padding word.
    pad5: f32,
}

/// One velocity sample as uploaded. `16`-byte `std430` stride matching
/// `array<vec4<f32>>` in the shader: the three velocity components plus one pad
/// lane that stays zero.
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
    ///
    /// Provenance: local device-upload helper for `GpuFluidAdvect`; no third
    /// party engine source or derived code.
    fn from_vec3(v: Vec3) -> Self {
        GpuVelocity {
            x: v.x,
            y: v.y,
            z: v.z,
            pad: 0.0,
        }
    }
}

/// One query as uploaded to the device. `80`-byte `std430` stride matching
/// `Query` in [`FLUID_ADVECT_WGSL`]: five padded `vec4` lanes whose `xyz`
/// components carry the position, velocity and the three `MacCormack` inputs.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQueryRaw {
    /// Sample / integration position (`xyz`; `w` unused).
    pos: [f32; 4],
    /// Back-trace velocity (`xyz`; `w` unused).
    velocity: [f32; 4],
    /// `MacCormack` forward-advected value (`xyz`; `w` unused).
    mac_forward: [f32; 4],
    /// `MacCormack` original value (`xyz`; `w` unused).
    mac_original: [f32; 4],
    /// `MacCormack` back-advected value (`xyz`; `w` unused).
    mac_back: [f32; 4],
}

/// One result as read back from the device. `96`-byte `std430` stride matching
/// `AdvectResult` in [`FLUID_ADVECT_WGSL`]: four padded `vec4` outputs plus two
/// `vec4` lanes packing the eight trilinear weights.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResultRaw {
    /// Semi-Lagrangian back-trace position (`xyz`; `w` zero).
    backtrace: [f32; 4],
    /// Trilinearly sampled velocity (`xyz`; `w` zero).
    sampled: [f32; 4],
    /// In-field advected position (`xyz`; `w` zero).
    advected: [f32; 4],
    /// `MacCormack`-corrected value (`xyz`; `w` zero).
    maccormack: [f32; 4],
    /// Trilinear corner weights `0..3`.
    weights_lo: [f32; 4],
    /// Trilinear corner weights `4..7`.
    weights_hi: [f32; 4],
}

/// One advection query: a sample position, a back-trace velocity and the three
/// values the `MacCormack` correction blends.
///
/// Provenance: twin input record for the golden
/// [`fluid`](prism_render_architecture::particle::fluid) advection cluster; no
/// third party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuAdvectQuery {
    /// The sample and integration position in grid space.
    pub pos: Vec3,
    /// The velocity used by the semi-Lagrangian back-trace.
    pub velocity: Vec3,
    /// The forward-advected value fed to the `MacCormack` correction.
    pub mac_forward: Vec3,
    /// The original value fed to the `MacCormack` correction.
    pub mac_original: Vec3,
    /// The back-advected value fed to the `MacCormack` correction.
    pub mac_back: Vec3,
}

/// The five golden advection outputs evaluated at one [`GpuAdvectQuery`].
///
/// Provenance: twin output record for the golden
/// [`fluid`](prism_render_architecture::particle::fluid) advection cluster; no
/// third party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuAdvectResult {
    /// The semi-Lagrangian back-trace position `pos − velocity·dt`.
    pub backtrace: Vec3,
    /// The trilinearly sampled velocity at `pos`
    /// ([`Vec3::ZERO`](prism_render_architecture::particle::Vec3::ZERO) for a
    /// degenerate field).
    pub sampled: Vec3,
    /// The in-field advected position `pos + sampled·dt`.
    pub advected: Vec3,
    /// The `MacCormack`-corrected value `forward + 0.5·(original − back)`.
    pub maccormack: Vec3,
    /// The eight trilinear corner weights at `pos`'s in-cell fraction, in the
    /// reference bitmask order (all zero for a degenerate field).
    pub weights: [f32; 8],
}

/// A compiled, reusable semi-Lagrangian advection kernel, twinning the `CPU`
/// golden [`fluid`](prism_render_architecture::particle::fluid) advection
/// cluster.
///
/// Provenance: on-device twin of the golden
/// [`fluid`](prism_render_architecture::particle::fluid) advection pure
/// functions; no third party engine source or derived code.
pub struct GpuFluidAdvect {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFluidAdvect {
    /// Compiles the advection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    ///
    /// Provenance: pipeline construction for the golden
    /// [`fluid`](prism_render_architecture::particle::fluid) advection twin; no
    /// third party engine source or derived code.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFluidAdvect {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fluid_advect_module"),
            source: ShaderSource::Wgsl(FLUID_ADVECT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fluid_advect_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fluid_advect_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fluid_advect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("advect"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFluidAdvect {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query against the shared velocity `field` at resolution
    /// `res` and time step `dt`, returning one [`GpuAdvectResult`] per query in
    /// input order.
    ///
    /// Each result matches the golden functions to within the tolerance
    /// documented on this module. An empty query batch returns an empty vector
    /// with no dispatch issued (a storage buffer cannot be zero-sized). A
    /// degenerate field (fewer than `res.voxel_count()` samples, or a zero-voxel
    /// grid) yields a zero
    /// [`sampled`](GpuAdvectResult::sampled) velocity and an untouched
    /// [`advected`](GpuAdvectResult::advected) position, matching the reference
    /// guard; the back-trace and `MacCormack` outputs are still valid.
    ///
    /// Provenance: dispatch and read-back for the golden
    /// [`fluid`](prism_render_architecture::particle::fluid) advection twin; no
    /// third party engine source or derived code.
    #[must_use]
    pub fn advect(
        &self,
        ctx: &GpuContext,
        field: &[Vec3],
        res: GridResolution,
        dt: f32,
        queries: &[GpuAdvectQuery],
    ) -> Vec<GpuAdvectResult> {
        if queries.is_empty() {
            return Vec::new();
        }

        let device = ctx.device();

        let voxel_count = res.voxel_count() as usize;
        let field_valid = voxel_count > 0 && field.len() >= voxel_count;

        let gpu_params = Params {
            nx: res.nx,
            ny: res.ny,
            nz: res.nz,
            field_valid: u32::from(field_valid),
            query_count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
            dt,
            pad3: 0.0,
            pad4: 0.0,
            pad5: 0.0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_advect_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });

        // A storage buffer cannot be zero-sized; for a degenerate field upload a
        // single placeholder voxel the kernel never reads because `field_valid`
        // is zero.
        let placeholder = [GpuVelocity {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            pad: 0.0,
        }];
        let packed: Vec<GpuVelocity> = if field_valid {
            field[..voxel_count]
                .iter()
                .map(|&v| GpuVelocity::from_vec3(v))
                .collect()
        } else {
            Vec::new()
        };
        let field_contents: &[GpuVelocity] = if packed.is_empty() {
            &placeholder
        } else {
            &packed
        };
        let field_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_advect_field"),
            contents: bytemuck::cast_slice(field_contents),
            usage: BufferUsages::STORAGE,
        });

        let raw_queries: Vec<GpuQueryRaw> = queries
            .iter()
            .map(|q| GpuQueryRaw {
                pos: [q.pos.x, q.pos.y, q.pos.z, 0.0],
                velocity: [q.velocity.x, q.velocity.y, q.velocity.z, 0.0],
                mac_forward: [q.mac_forward.x, q.mac_forward.y, q.mac_forward.z, 0.0],
                mac_original: [q.mac_original.x, q.mac_original.y, q.mac_original.z, 0.0],
                mac_back: [q.mac_back.x, q.mac_back.y, q.mac_back.z, 0.0],
            })
            .collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_advect_queries"),
            contents: bytemuck::cast_slice(&raw_queries),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (queries.len() * size_of::<GpuResultRaw>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_advect_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fluid_advect_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: field_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_advect_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fluid_advect_encoder"),
        });
        {
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fluid_advect_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
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
        let raw = bytemuck::cast_slice::<u8, GpuResultRaw>(&view).to_vec();
        drop(view);
        stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.into_iter()
            .map(|r| GpuAdvectResult {
                backtrace: Vec3::new(r.backtrace[0], r.backtrace[1], r.backtrace[2]),
                sampled: Vec3::new(r.sampled[0], r.sampled[1], r.sampled[2]),
                advected: Vec3::new(r.advected[0], r.advected[1], r.advected[2]),
                maccormack: Vec3::new(r.maccormack[0], r.maccormack[1], r.maccormack[2]),
                weights: [
                    r.weights_lo[0],
                    r.weights_lo[1],
                    r.weights_lo[2],
                    r.weights_lo[3],
                    r.weights_hi[0],
                    r.weights_hi[1],
                    r.weights_hi[2],
                    r.weights_hi[3],
                ],
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
