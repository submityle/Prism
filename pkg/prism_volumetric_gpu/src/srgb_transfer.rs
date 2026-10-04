//! `wgpu` compute twin of the sRGB electro-optical transfer-function family,
//! from the `CPU` golden `prism_math::color::transfer`'s four functions
//! `srgb_to_linear`, `linear_to_srgb`, `fast_srgb_to_linear` and
//! `fast_linear_to_srgb`.
//!
//! These convert between non-linear sRGB component values (as stored in
//! textures and framebuffers) and linear-light values. The exact pair is the
//! IEC 61966-2-1 piecewise curve; the `fast_*` pair is the single-`pow`
//! gamma-2.2 approximation. This module ports those four stateless closed forms
//! onto the device behind an integer selector, so one thread resolves one
//! query: a passing real-device parity test is direct evidence the ported
//! kernel computes the same transfer the reference does, not merely that the
//! shader compiles.
//!
//! This is deliberately distinct from the crate's `tonemap` twin, which uses
//! the `e*e` / `sqrt` fast approximation; here the exact piecewise IEC curve
//! and the gamma-2.2 variants are reproduced byte-for-byte in their constants.
//!
//! # What is twinned
//!
//! The `func_id` selects which golden function the thread reproduces for its
//! scalar component `c`:
//!
//! * `func_id == 0` mirrors `srgb_to_linear`: `c <= 0.040_448_237` gives
//!   `c / 12.92`, else `pow((c + 0.055) / 1.055, 2.4)`.
//! * `func_id == 1` mirrors `linear_to_srgb`: `c <= 0.003_130_8` gives
//!   `c * 12.92`, else `1.055 * pow(c, 1.0 / 2.4) - 0.055`.
//! * `func_id == 2` mirrors `fast_srgb_to_linear`: `pow(c, 2.2)`.
//! * `func_id == 3` mirrors `fast_linear_to_srgb`: `pow(c, 1.0 / 2.2)`.
//!
//! Any `func_id > 3` yields `valid = 0` with `value = 0`. For a known function
//! id the result is always `valid = 1`, since the golden extrapolates
//! monotonically outside `[0, 1]`.
//!
//! # Correctness model
//!
//! The golden evaluates every constant and `pow` in `f32`, and the kernel
//! reproduces the same `f32` operator order with the device `pow`; `CPU` and
//! `GPU` are therefore not bit-exact (a `GPU` may contract a multiply-add and
//! its `pow` differs in the last units in the last place). The valid `value`
//! scalar is compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly. The
//! parity sweep keeps the component away from the two piecewise knees
//! (`0.003_130_8` and `0.040_448_237`) so the branch decision cannot flip under
//! round-off.
//!
//! # Degenerate inputs
//!
//! An out-of-range `func_id` (greater than `3`) yields `valid = 0` with a
//! cleared `value`. Every branch is evaluated unconditionally and combined with
//! `select`, so no un-taken branch can misbehave. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `pow`, `+ - * /`,
//! `select`, ordered compares and unsigned index arithmetic — with no `sin`,
//! `cos`, `tan`, `exp`, `log`, no `round`, no `f32` remainder and no banned
//! wide types, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! piecewise branch uses the ordered compare `c <= threshold`, and the
//! function selector uses integer `==` on the `u32` id; there is no `f32`
//! equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::color::transfer`；无第三方引擎源码或衍生代码。
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

/// Function id for `srgb_to_linear` (exact piecewise decode).
pub const FUNC_SRGB_TO_LINEAR: u32 = 0;
/// Function id for `linear_to_srgb` (exact piecewise encode).
pub const FUNC_LINEAR_TO_SRGB: u32 = 1;
/// Function id for `fast_srgb_to_linear` (gamma-2.2 decode).
pub const FUNC_FAST_SRGB_TO_LINEAR: u32 = 2;
/// Function id for `fast_linear_to_srgb` (gamma-2.2 encode).
pub const FUNC_FAST_LINEAR_TO_SRGB: u32 = 3;

/// The portable core-`WGSL` sRGB transfer kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// four `CPU` golden transfer functions selected by `func_id`; see the module
/// documentation for the closed forms.
const SRGB_TRANSFER_WGSL: &str = r#"
// sRGB transfer twin: one thread per query reproduces one of four prism_math
// color transfer functions selected by func_id. It uses only the portable
// core-WGSL subset (pow, + - * /, select, ordered compares plus unsigned index
// math), has no loop and no data-dependent control flow (every branch is
// computed and combined with select), so it provably terminates. The piecewise
// split uses the ordered compare c <= threshold and the selector uses integer
// == on the u32 id; there is no bare f32 equality.
// Provenance: 孪生自本仓 prism_math::color::transfer。

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Which golden transfer function to reproduce.
    func_id: u32,
    // Scalar colour component to transform.
    c: f32,
    // Padding words to a 4-word (16-byte) stride.
    pad0: u32,
    pad1: u32,
}

struct Result {
    // The transferred component when valid, else 0.
    value: f32,
    // 1 when func_id is a known function (<= 3), else 0.
    valid: u32,
    // Padding words to a 4-word (16-byte) stride.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let func_id = q.func_id;
    let c = q.c;

    // func 0: srgb_to_linear (exact piecewise decode).
    let s2l = select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.040448237);
    // func 1: linear_to_srgb (exact piecewise encode).
    let l2s = select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
    // func 2: fast_srgb_to_linear (gamma 2.2).
    let fs2l = pow(c, 2.2);
    // func 3: fast_linear_to_srgb (inverse gamma 2.2).
    let fl2s = pow(c, 1.0 / 2.2);

    // Select the chosen function by integer id; unknown ids fall through to 0.
    var value = 0.0;
    value = select(value, s2l, func_id == 0u);
    value = select(value, l2s, func_id == 1u);
    value = select(value, fs2l, func_id == 2u);
    value = select(value, fl2s, func_id == 3u);

    let ok = func_id <= 3u;

    var out: Result;
    out.value = select(0.0, value, ok);
    out.valid = select(0u, 1u, ok);
    out.pad0 = 0u;
    out.pad1 = 0u;
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct:
/// the function id, the scalar component and padding to a 4-word (16-byte)
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    func_id: u32,
    c: f32,
    pad0: u32,
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the transferred value, the validity flag and padding to a 4-word
/// (16-byte) stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    value: f32,
    valid: u32,
    pad0: u32,
    pad1: u32,
}

/// One sRGB transfer query: a function selector and the scalar component to
/// transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrgbTransferQuery {
    /// Which golden transfer function to reproduce (see the `FUNC_*`
    /// constants).
    pub func_id: u32,
    /// The scalar colour component to transform.
    pub c: f32,
}

impl SrgbTransferQuery {
    /// Builds a query selecting `func_id` for component `c`.
    #[must_use]
    pub fn new(func_id: u32, c: f32) -> SrgbTransferQuery {
        SrgbTransferQuery { func_id, c }
    }
}

/// One resolved answer for a single query, mirroring the selected golden
/// transfer function for that component.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrgbTransferResult {
    /// The transferred component when valid, else `0`.
    pub value: f32,
    /// `1` when `func_id` is a known function (`<= 3`), else `0`.
    pub valid: u32,
}

/// Encodes one [`SrgbTransferQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SrgbTransferQuery) -> GpuQuery {
    GpuQuery {
        func_id: q.func_id,
        c: q.c,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SrgbTransferResult`].
fn decode_result(raw: &GpuResult) -> SrgbTransferResult {
    SrgbTransferResult {
        value: raw.value,
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

/// A compiled, reusable sRGB transfer compute pipeline, twinning the four `CPU`
/// golden `prism_math::color::transfer` functions.
pub struct GpuSrgbTransfer {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSrgbTransfer {
    /// Compiles the sRGB transfer kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSrgbTransfer {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_srgb_transfer"),
            source: ShaderSource::Wgsl(SRGB_TRANSFER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_srgb_transfer_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_srgb_transfer_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_srgb_transfer_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSrgbTransfer {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`SrgbTransferResult`]
    /// per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `value` scalar to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SrgbTransferQuery],
    ) -> Vec<SrgbTransferResult> {
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
            label: Some("prism_volumetric_srgb_transfer_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_srgb_transfer_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_srgb_transfer_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_srgb_transfer_bind_group"),
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
            label: Some("prism_volumetric_srgb_transfer_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_srgb_transfer_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_srgb_transfer_pass"),
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
