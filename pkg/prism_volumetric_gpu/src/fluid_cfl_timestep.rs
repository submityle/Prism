//! `wgpu` compute twin of the particle-subsystem fluid *stability* and
//! *advection* scalars: the advection `CFL` number
//! ([`cfl_number`](prism_render_architecture::particle::fluid::cfl_number)), the
//! largest stable time step
//! ([`stable_timestep`](prism_render_architecture::particle::fluid::stable_timestep))
//! and the one-step particle integration
//! ([`advect_particle`](prism_render_architecture::particle::fluid::advect_particle),
//! fluid design §10).
//!
//! The fluid solve picks its step from the current peak speed and cell size,
//! reports the resulting `CFL` number, and integrates parcels forward under an
//! already-sampled velocity. The three pieces are tiny per-cell scalar kernels,
//! so [`GpuFluidCflTimestep`] packs one *query* per thread and emits all three
//! results in a single dispatch. A passing real-device parity test is therefore
//! direct evidence the ported kernel reproduces the same two guarded divisions
//! and the same multiply-add integration the reference does, not merely that its
//! shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel computes, guard-for-guard with the golden:
//! - the `CFL` number `|v|·dt / h`, which is `0` when `|h|` is at or below the
//!   golden `EPS_LEN_SQ` floor so a degenerate cell size never divides by zero;
//! - the stable step `cfl_target·h / |v|`, which is `0` when `|v|` is at or
//!   below that same floor so a still field yields a zero step rather than a
//!   division by zero;
//! - the advected position `pos + v·dt`, a pure multiply-add.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+ − × ÷` and
//! integer index math — with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt` or
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable pair of guarded divisions plus a
//! multiply-add, so `CPU` and `GPU` evaluate the same closed form. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`) on the continuous values, while the degenerate branch is
//! pinned by fixtures that keep every denominator either exactly zero or far
//! above the floor, so the device and the reference always take the same branch
//! and the zero-return cases match exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// shared by every kernel in this crate.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `CFL`/timestep/advection kernel, embedded inline so
/// the twin ships as a single source file. Mirrors the `CPU` golden
/// [`cfl_number`](prism_render_architecture::particle::fluid::cfl_number),
/// [`stable_timestep`](prism_render_architecture::particle::fluid::stable_timestep)
/// and [`advect_particle`](prism_render_architecture::particle::fluid::advect_particle)
/// guard-for-guard; see the module documentation for the algorithm.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
const FLUID_CFL_TIMESTEP_WGSL: &str = r#"
// Fluid CFL/timestep/advection twin: one thread per query computes the CFL
// number |v|*dt/h (zero when |h| <= EPS_LEN_SQ), the stable step
// cfl_target*h/|v| (zero when |v| <= EPS_LEN_SQ) and the advected position
// pos + v*dt, mirroring the CPU golden `particle::fluid` guard for guard. It
// uses only the portable core-WGSL subset (abs, + - * / and integer index
// math) and takes no optional feature, so it runs unmodified on Metal, Vulkan
// and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::fluid;
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 48-byte std430 stride matching the host `GpuQuery`: the parcel
// position and the sampled velocity, each padded to a vec4 so the storage array
// needs no manual vec3 alignment arithmetic, then the four scalars the three
// golden functions consume.
struct Query {
    pos: vec4<f32>,
    velocity: vec4<f32>,
    max_velocity: f32,
    dt: f32,
    cell_size: f32,
    cfl_target: f32,
}

// One result. 32-byte std430 stride matching the host `GpuResult`: the advected
// position padded to a vec4, then the CFL number, the stable step and two pad
// words.
struct Result {
    advected_pos: vec4<f32>,
    cfl: f32,
    stable_dt: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// The golden `EPS_LEN_SQ`: the magnitude floor below which a denominator is
// treated as zero so neither division ever yields a NaN. A direct f32 `==`/`!=`
// is forbidden, so the degenerate tests compare magnitudes against this floor.
const EPS_LEN_SQ: f32 = 1e-12;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // CFL number |v|*dt/h, guarded on the cell size exactly as the golden
    // `cfl_number`: only above the floor is the division taken, else zero.
    var cfl = 0.0;
    if (abs(q.cell_size) > EPS_LEN_SQ) {
        cfl = (q.max_velocity * q.dt) / q.cell_size;
    }

    // Stable step cfl_target*h/|v|, guarded on the max speed exactly as the
    // golden `stable_timestep`: a still field yields a zero step.
    var stable_dt = 0.0;
    if (abs(q.max_velocity) > EPS_LEN_SQ) {
        stable_dt = (q.cfl_target * q.cell_size) / q.max_velocity;
    }

    // Advected position pos + v*dt, the multiply-add of the golden
    // `advect_particle`.
    let advected = q.pos.xyz + q.velocity.xyz * q.dt;

    var r: Result;
    r.advected_pos = vec4<f32>(advected, 0.0);
    r.cfl = cfl;
    r.stable_dt = stable_dt;
    r.pad0 = 0.0;
    r.pad1 = 0.0;
    results[idx] = r;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`FLUID_CFL_TIMESTEP_WGSL`]: the query count and three pad words
/// — `16` bytes, each field at the uniform offset the shader expects.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `48`-byte `std430` stride matching `Query` in the
/// shader: the parcel position and the sampled velocity, each padded to a
/// `vec4` lane, then the four scalars the three golden functions consume.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Parcel position in `xyz`; the `w` lane is unused padding.
    pos: [f32; 4],
    /// Sampled velocity in `xyz`; the `w` lane is unused padding.
    velocity: [f32; 4],
    /// Peak speed `|v|` driving the `CFL` and the stable-step divisions.
    max_velocity: f32,
    /// Time step `dt` for the `CFL` number and the advection.
    dt: f32,
    /// Cell size `h` (the `CFL` denominator and the stable-step numerator).
    cell_size: f32,
    /// Target `CFL` number the stable step solves for.
    cfl_target: f32,
}

impl GpuQuery {
    /// Packs a [`GpuCflTimestepQuery`] into the `std430` upload layout.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
    fn from_query(query: &GpuCflTimestepQuery) -> GpuQuery {
        let p = query.pos;
        let v = query.velocity;
        GpuQuery {
            pos: [p.x, p.y, p.z, 0.0],
            velocity: [v.x, v.y, v.z, 0.0],
            max_velocity: query.max_velocity,
            dt: query.dt,
            cell_size: query.cell_size,
            cfl_target: query.cfl_target,
        }
    }
}

/// One result as read back. `32`-byte `std430` stride matching `Result` in the
/// shader: the advected position padded to a `vec4`, then the `CFL` number, the
/// stable step and two pad lanes held at zero.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Advected position in `xyz`; the `w` lane is unused padding.
    advected_pos: [f32; 4],
    /// The `CFL` number `|v|·dt / h`.
    cfl: f32,
    /// The stable step `cfl_target·h / |v|`.
    stable_dt: f32,
    /// Padding lane, held at zero.
    pad0: f32,
    /// Padding lane, held at zero.
    pad1: f32,
}

/// Maps one kernel `Result` lane back to the host [`GpuCflTimestepResult`].
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
fn decode_result(raw: &GpuResult) -> GpuCflTimestepResult {
    GpuCflTimestepResult {
        cfl: raw.cfl,
        stable_dt: raw.stable_dt,
        advected_pos: Vec3::new(
            raw.advected_pos[0],
            raw.advected_pos[1],
            raw.advected_pos[2],
        ),
    }
}

/// One fluid stability-and-advection request.
///
/// `max_velocity`, `dt` and `cell_size` feed the `CFL` number; `cfl_target`,
/// `cell_size` and `max_velocity` feed the stable step; `pos`, `velocity` and
/// `dt` feed the one-step advection. Derives only [`PartialEq`] (no
/// [`Eq`]/[`Hash`]) because it holds `f32` fields.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCflTimestepQuery {
    /// Peak speed `|v|` of the field.
    pub max_velocity: f32,
    /// Time step `dt`.
    pub dt: f32,
    /// Cell size `h`.
    pub cell_size: f32,
    /// Target `CFL` number for the stable step.
    pub cfl_target: f32,
    /// Parcel position being integrated.
    pub pos: Vec3,
    /// Already-sampled velocity at the parcel.
    pub velocity: Vec3,
}

/// The device-computed answer for one query: the `CFL` number, the stable step
/// and the advected position.
///
/// Derives only [`PartialEq`] (no [`Eq`]/[`Hash`]) because it holds `f32`
/// fields.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCflTimestepResult {
    /// The advection `CFL` number `|v|·dt / h` (zero on a degenerate cell size).
    pub cfl: f32,
    /// The largest stable step `cfl_target·h / |v|` (zero on a still field).
    pub stable_dt: f32,
    /// The advected position `pos + v·dt`.
    pub advected_pos: Vec3,
}

/// Builds a compute-visible buffer binding layout entry.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
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

/// A compiled, reusable `CFL`/timestep/advection compute pipeline, twinning the
/// `CPU` golden
/// [`fluid`](prism_render_architecture::particle::fluid) stability scalars.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
pub struct GpuFluidCflTimestep {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFluidCflTimestep {
    /// Compiles the `CFL`/timestep/advection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFluidCflTimestep {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_module"),
            source: ShaderSource::Wgsl(FLUID_CFL_TIMESTEP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFluidCflTimestep {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries`, returning one [`GpuCflTimestepResult`]
    /// per query in input order.
    ///
    /// Each result mirrors
    /// [`cfl_number`](prism_render_architecture::particle::fluid::cfl_number),
    /// [`stable_timestep`](prism_render_architecture::particle::fluid::stable_timestep)
    /// and [`advect_particle`](prism_render_architecture::particle::fluid::advect_particle)
    /// evaluated on that query. An empty `queries` slice yields an empty result
    /// — a storage buffer may not be zero-sized — handled by an early return
    /// before any dispatch.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuCflTimestepQuery],
    ) -> Vec<GpuCflTimestepResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(GpuQuery::from_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fluid_cfl_timestep_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fluid_cfl_timestep_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(raw.len(), queries.len());

        raw.iter().map(decode_result).collect()
    }
}
