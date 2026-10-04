//! `wgpu` compute twin of the Shirley-Chiu concentric square-to-disk map, from
//! the `CPU` golden `prism_render_architecture::reference_pt::concentric`'s
//! `concentric_disk`.
//!
//! Mapping a unit square onto the unit disk is the primitive that lets a
//! low-discrepancy (`QMC`) sequence drive aperture and hemisphere sampling
//! without the stratification loss of rejection sampling. Peter `Shirley` and
//! Kenneth `Chiu`'s concentric map folds the square into four wedges and sends
//! concentric square rings to concentric circles, so neighbouring square
//! samples stay neighbours on the disk; the map is area-preserving and keeps
//! the discrepancy of its input sequence. This module ports that single
//! stateless closed form onto the device: one thread resolves one query.
//!
//! # What is twinned
//!
//! For each `(u, v)` in `[0, 1]^2` the kernel reproduces `concentric_disk`:
//!
//! * Remap to `a = 2*u - 1`, `b = 2*v - 1` so the wedges are symmetric.
//! * The exact centre (`a*a + b*b <= 0`, i.e. `u = v = 0.5`) maps to `(0, 0)`.
//! * Horizontal wedge (`a*a > b*b`): `phi = (pi/4) * (b/a)`,
//!   `(x, y) = (a*cos(phi), a*sin(phi))`.
//! * Vertical wedge (otherwise): `t = (pi/4) * (a/b)`,
//!   `(x, y) = (b*sin(t), b*cos(t))`.
//!
//! The wedge angle never leaves `[-pi/4, pi/4]`. The golden evaluates `sin`
//! and `cos` with fixed `f64` truncated `Taylor` polynomials (residual below
//! `2e-9` over the wedge); on the device the argument is likewise confined to
//! `[-pi/4, pi/4]`, where the native `f32` `sin`/`cos` differ from the `f64`
//! polynomials by far less than the parity tolerance, so the kernel uses the
//! native intrinsics directly.
//!
//! # Correctness model
//!
//! The continuous `(disk_x, disk_y)` pair is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`) against an
//! independent host oracle that re-derives the golden `f64` `Taylor` series. The
//! map always produces a defined output, so the discrete `valid` flag is held
//! at `1` to keep the one-thread-per-query `std430` round-trip uniform with the
//! other twins in this crate; it is compared exactly.
//!
//! # Degenerate inputs
//!
//! The square centre is the only point with no well-defined wedge angle and is
//! sent to the origin. The kernel feeds each wedge divisor through a `select`
//! guard keyed on the taken branch, so the un-taken branch never divides by
//! zero. An empty query batch short-circuits on the host with no dispatch, since
//! a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `sin`, `cos`,
//! `+ - * /`, `select` and unsigned index arithmetic — with no `tan`, `exp`,
//! `log`, `pow`, no `round`, no `f32` remainder, and no `f64`/`u64`/`i64`. The
//! centre test and the wedge selection are ordered compares fed to `select`;
//! there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::concentric`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

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

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` concentric-disk kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `concentric_disk`; see the module documentation for the closed
/// form.
const CONCENTRIC_DISK_WGSL: &str = r#"
// Concentric square-to-disk twin: one thread per query reproduces the
// Shirley-Chiu map. It uses only the portable core-WGSL subset (abs, sin, cos,
// + - * /, select plus unsigned index math), takes no optional feature, and has
// no loop and no branch, so it provably terminates. The centre test and wedge
// selection are ordered compares fed to select; there is no f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Unit-square coordinate u in [0, 1].
    u: f32,
    // Unit-square coordinate v in [0, 1].
    v: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Disk coordinate x.
    disk_x: f32,
    // Disk coordinate y.
    disk_y: f32,
    // Always 1: the map is total, so every query is valid.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// A quarter of pi; the half-width of a single concentric wedge's angular sweep.
const FRAC_PI_4: f32 = 0.7853981633974483;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    // Remap the unit square into [-1, 1]^2 so the wedges are symmetric.
    let a = 2.0 * q.u - 1.0;
    let b = 2.0 * q.v - 1.0;

    // The centre is the only point with a <= 0 sum of squares; it has no wedge
    // angle and maps to the origin. The wedge choice is the ordered a*a > b*b.
    let is_centre = (a * a + b * b) <= 0.0;
    let horizontal = (a * a) > (b * b);
    let use_horizontal = horizontal && !is_centre;
    let use_vertical = (!horizontal) && (!is_centre);

    // Guard each divisor on its own taken branch so the un-taken branch (and the
    // centre) never divides by zero.
    let a_denom = select(1.0, a, use_horizontal);
    let b_denom = select(1.0, b, use_vertical);

    // Horizontal wedge: radius |a|, angle (pi/4)*(b/a) within [-pi/4, pi/4].
    let phi = FRAC_PI_4 * (b / a_denom);
    let hx = a * cos(phi);
    let hy = a * sin(phi);

    // Vertical wedge: fold through the diagonal so the argument stays in range.
    // With t = (pi/4)*(a/b) the exact angle is pi/2 - t, and cos(pi/2 - t) =
    // sin(t), sin(pi/2 - t) = cos(t).
    let t = FRAC_PI_4 * (a / b_denom);
    let vx = b * sin(t);
    let vy = b * cos(t);

    let wedge_x = select(vx, hx, horizontal);
    let wedge_y = select(vy, hy, horizontal);

    var out: Result;
    out.disk_x = select(wedge_x, 0.0, is_centre);
    out.disk_y = select(wedge_y, 0.0, is_centre);
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// `repr(C)` `std430` dispatch parameters: the query count plus padding to a
/// 16-byte uniform block.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The two coordinates are padded to `4` `f32` words (`16` bytes), aligned to
/// `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    u: f32,
    v: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the two disk coordinates and the validity flag — `3` words
/// (`12` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    disk_x: f32,
    disk_y: f32,
    valid: u32,
}

/// One concentric-disk query: a unit-square sample `(u, v)` in `[0, 1]^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConcentricDiskQuery {
    /// Unit-square coordinate `u`.
    pub u: f32,
    /// Unit-square coordinate `v`.
    pub v: f32,
}

impl ConcentricDiskQuery {
    /// Builds a query from the two unit-square coordinates.
    #[must_use]
    pub fn new(u: f32, v: f32) -> ConcentricDiskQuery {
        ConcentricDiskQuery { u, v }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `concentric_disk` output for that unit-square sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConcentricDiskResult {
    /// Disk coordinate `x`.
    pub disk_x: f32,
    /// Disk coordinate `y`.
    pub disk_y: f32,
    /// Always `1`: the concentric map is total, so every query is valid.
    pub valid: u32,
}

/// Encodes one [`ConcentricDiskQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ConcentricDiskQuery) -> GpuQuery {
    GpuQuery {
        u: q.u,
        v: q.v,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ConcentricDiskResult`].
fn decode_result(raw: &GpuResult) -> ConcentricDiskResult {
    ConcentricDiskResult {
        disk_x: raw.disk_x,
        disk_y: raw.disk_y,
        valid: raw.valid,
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

/// A compiled, reusable concentric-disk compute pipeline, twinning the `CPU`
/// golden `concentric_disk`.
pub struct GpuConcentricDisk {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuConcentricDisk {
    /// Compiles the concentric-disk kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuConcentricDisk {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_concentric_disk"),
            source: ShaderSource::Wgsl(CONCENTRIC_DISK_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_concentric_disk_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_concentric_disk_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_concentric_disk_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuConcentricDisk {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`ConcentricDiskResult`]
    /// per input, in order.
    ///
    /// Each `(disk_x, disk_y)` matches the reference to the module's tolerance
    /// and the `valid` flag exactly. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ConcentricDiskQuery],
    ) -> Vec<ConcentricDiskResult> {
        let count = queries.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_concentric_disk_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_concentric_disk_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_concentric_disk_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_concentric_disk_bind_group"),
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
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_concentric_disk_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_concentric_disk_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_concentric_disk_pass"),
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
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.iter().map(decode_result).collect()
    }
}
