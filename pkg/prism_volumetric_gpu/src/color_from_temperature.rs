//! `wgpu` compute twin of the full correlated-color-temperature to linear-`RGB`
//! chain, from the `CPU` golden `prism_math::color::LinearRgba::from_temperature`.
//!
//! A blackbody temperature in Kelvin is mapped to a unit-luminance linear sRGB
//! color: first to a `CIE` 1931 `(x, y)` chromaticity on the Planckian locus
//! using the Kim et al. (2002) cubic-spline approximation, then to `XYZ` with
//! luminance normalized to `Y = 1`, then through the `XYZ` to linear-`RGB`
//! matrix, with negative components clamped to `0`. One thread resolves one
//! query, so a passing real-device parity test is direct evidence the ported
//! kernel computes the same color the reference does, not merely that the
//! shader compiles.
//!
//! This is the complete `CCT` to linear-`RGB` composite. It is a different
//! function from the `planckian_locus` twin, which stops at the `(x, y)`
//! chromaticity, and from the `color_temperature` twin, which reproduces the
//! `WhiteBalance` red/blue gain rational pieces. The three share no outputs.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `from_temperature`:
//!
//! * `t = clamp(kelvin, 1667, 25000)`, `inv = 1 / t`, `inv2`, `inv3`.
//! * `x` uses the low-temperature cubic in `inv` when `t <= 4000`, else the
//!   high-temperature cubic.
//! * `y` uses the first `y` cubic in `x` when `t <= 2222`, the second when
//!   `t <= 4000`, else the third.
//! * `X = x / y`, `Y = 1`, `Z = (1 - x - y) / y`.
//! * The linear-`RGB` components come from the fixed `XYZ` matrix, each clamped
//!   to a `0` floor.
//!
//! A finite `kelvin` is valid because it is clamped before use, so `valid` is
//! `1`; a non-finite `kelvin` (`NaN` or infinity) yields `valid = 0` and a zero
//! color.
//!
//! # Correctness model
//!
//! The golden is entirely `f32`, so the host oracle is a straight `f32`
//! re-implementation. Each continuous output is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly. The parity sweep keeps samples at least a
//! few Kelvin away from the `2222` and `4000` breakpoints so the piecewise
//! decision cannot be flipped by round-off between host and device.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `max`,
//! `select`, `+ - * /` and unsigned index arithmetic — with no `u64`/`i64`/`f64`,
//! no `round`, no `f32` remainder, and no bare `f32` equality (the breakpoints
//! are ordered `<=` comparisons), so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`. The spline and matrix coefficients are written as `f32` literals
//! matching the reference byte-for-byte.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::color::temperature` 与 `prism_math::color::linear`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` color-from-temperature kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the `CPU` golden `from_temperature`; see the module documentation
/// for the closed forms.
const COLOR_FROM_TEMPERATURE_WGSL: &str = r#"
// Color-from-temperature twin: one thread per query reproduces the full
// CCT -> linear-RGB chain. It uses only the portable core-WGSL subset (clamp,
// max, select, + - * / plus unsigned index math) and has no loop and no branch
// beyond the bounds guard, so it provably terminates. All breakpoint decisions
// are ordered <= comparisons; there is no bare f32 equality. A non-finite
// kelvin is detected with an ordered magnitude test and reported as invalid.

struct Query {
    // Correlated color temperature in Kelvin (clamped on use).
    kelvin: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Color {
    // Unit-luminance linear sRGB components, each floored at 0.
    r: f32,
    g: f32,
    b: f32,
    // 1 when kelvin is finite (clamped and evaluated), 0 otherwise.
    valid: u32,
}

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Color>;

const MIN_KELVIN: f32 = 1667.0;
const MAX_KELVIN: f32 = 25000.0;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    // Finiteness via an ordered magnitude test, which rejects both infinities
    // and NaN (NaN fails every ordered comparison).
    let is_finite = abs(q.kelvin) < 3.0e38;

    let t = clamp(q.kelvin, MIN_KELVIN, MAX_KELVIN);
    let inv = 1.0 / t;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;

    // x: cubic in 1/t, chosen by the 4000 K breakpoint.
    let x_low = -0.2661239e9 * inv3 - 0.2343589e6 * inv2 + 0.8776956e3 * inv + 0.179910;
    let x_high = -3.0258469e9 * inv3 + 2.107038e6 * inv2 + 0.2226347e3 * inv + 0.240390;
    let x = select(x_high, x_low, t <= 4000.0);

    let x2 = x * x;
    let x3 = x2 * x;

    // y: cubic in x, chosen by the 2222 K and 4000 K breakpoints.
    let y_a = -1.1063814 * x3 - 1.3481102 * x2 + 2.1855583 * x - 0.20219683;
    let y_b = -0.9549476 * x3 - 1.3741859 * x2 + 2.09137 * x - 0.16748867;
    let y_c = 3.081758 * x3 - 5.873387 * x2 + 3.7511299 * x - 0.37001483;
    let y = select(select(y_c, y_b, t <= 4000.0), y_a, t <= 2222.0);

    // xyY (Y = 1) -> XYZ.
    let big_x = x / y;
    let big_y = 1.0;
    let big_z = (1.0 - x - y) / y;

    // XYZ -> linear sRGB.
    let r = 3.2404542 * big_x - 1.5371385 * big_y - 0.4985314 * big_z;
    let g = -0.969266 * big_x + 1.8760108 * big_y + 0.0415560 * big_z;
    let b = 0.0556434 * big_x - 0.2040259 * big_y + 1.0572252 * big_z;

    var out: Color;
    out.r = select(0.0, max(r, 0.0), is_finite);
    out.g = select(0.0, max(g, 0.0), is_finite);
    out.b = select(0.0, max(b, 0.0), is_finite);
    out.valid = select(0u, 1u, is_finite);
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
/// The temperature is padded to `4` `f32` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    kelvin: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Color` struct:
/// three continuous `f32` components followed by the `u32` flag — `4` words
/// (`16` bytes), all `4`-byte aligned with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    r: f32,
    g: f32,
    b: f32,
    valid: u32,
}

/// One color-from-temperature query: a correlated color temperature in Kelvin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorFromTemperatureQuery {
    /// The correlated color temperature in Kelvin; clamped to `[1667, 25000]`.
    pub kelvin: f32,
}

impl ColorFromTemperatureQuery {
    /// Builds a query from a correlated color temperature in Kelvin.
    #[must_use]
    pub fn new(kelvin: f32) -> ColorFromTemperatureQuery {
        ColorFromTemperatureQuery { kelvin }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `from_temperature` for that temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorFromTemperatureResult {
    /// Unit-luminance linear sRGB red component, floored at `0`.
    pub r: f32,
    /// Unit-luminance linear sRGB green component, floored at `0`.
    pub g: f32,
    /// Unit-luminance linear sRGB blue component, floored at `0`.
    pub b: f32,
    /// `1` when `kelvin` is finite (clamped and evaluated), `0` otherwise.
    pub valid: u32,
}

/// Encodes one [`ColorFromTemperatureQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &ColorFromTemperatureQuery) -> GpuQuery {
    GpuQuery {
        kelvin: q.kelvin,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`ColorFromTemperatureResult`].
fn decode_result(raw: &GpuResult) -> ColorFromTemperatureResult {
    ColorFromTemperatureResult {
        r: raw.r,
        g: raw.g,
        b: raw.b,
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

/// A compiled, reusable color-from-temperature compute pipeline, twinning the
/// `CPU` golden `from_temperature`.
pub struct GpuColorFromTemperature {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuColorFromTemperature {
    /// Compiles the color-from-temperature kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuColorFromTemperature {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_color_from_temperature"),
            source: ShaderSource::Wgsl(COLOR_FROM_TEMPERATURE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_color_from_temperature_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_color_from_temperature_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_color_from_temperature_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuColorFromTemperature {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`ColorFromTemperatureResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the continuous `r`,
    /// `g`, `b` scalars to the module's tolerance. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[ColorFromTemperatureQuery],
    ) -> Vec<ColorFromTemperatureResult> {
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
            label: Some("prism_volumetric_color_from_temperature_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_color_from_temperature_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_color_from_temperature_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_color_from_temperature_bind_group"),
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
            label: Some("prism_volumetric_color_from_temperature_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_color_from_temperature_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_color_from_temperature_pass"),
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
