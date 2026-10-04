//! `wgpu` compute twin of the scalar easing-function family, from the `CPU`
//! golden `prism_math::curve::easing`.
//!
//! The golden module exposes 14 stateless easing curves that map a parameter
//! `t` to an eased value, each satisfying `f(0) == 0` and `f(1) == 1`. This
//! module ports every one of them onto the device behind a single kernel: each
//! query carries a function id (`0..=13`) and the parameter `t`, and one thread
//! resolves one query, so a passing real-device parity test is direct evidence
//! the ported curves evaluate the same values the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! The function id selects one of the golden curves, reproduced byte-for-byte
//! in operator order and constants:
//!
//! * `0` `smoothstep`, `1` `smootherstep` — both clamp `t` to `[0, 1]` first.
//! * `2` `quad_in`, `3` `quad_out`, `4` `quad_in_out`.
//! * `5` `cubic_in`, `6` `cubic_out`, `7` `cubic_in_out`.
//! * `8` `sine_in`, `9` `sine_out`, `10` `sine_in_out` — using `FRAC_PI_2`/`PI`.
//! * `11` `expo_in`, `12` `expo_out`, `13` `expo_in_out` — base-2 exponentials
//!   with the endpoints enforced exactly.
//!
//! Every id in `0..=13` is valid (`valid = 1`). Any id `> 13` is rejected with
//! `valid = 0` and `eased = 0`.
//!
//! # Correctness model
//!
//! The transcendental curves route through the device `sin`, `cos` and `pow`
//! built-ins, which differ slightly from the golden's `libm` backend, so `CPU`
//! and `GPU` are not bit-exact; the `eased` scalar is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete
//! `valid` flag is compared exactly; the parity sweep keeps `t` away from the
//! piecewise knees so the branch decision cannot flip under round-off.
//!
//! # Degenerate inputs
//!
//! A function id `> 13` yields `valid = 0` with `eased = 0`. The exponential
//! branches guard their endpoints with ordered compares fed to `select`, so an
//! un-taken branch never contributes a value. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset plus the `sin`, `cos`
//! and `pow` built-ins this family requires, with no `round`, no `f32`
//! remainder and no bare `f32` equality. Branch selection keys off `u32`
//! comparisons of the function id, and every piecewise decision uses an ordered
//! `f32` compare fed to `select`, so it runs unmodified on `Metal`, `Vulkan`
//! and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_math::curve::easing`；无第三方引擎源码或衍生代码。
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

/// The inline easing kernel. The single entry point `solve` mirrors the `CPU`
/// golden `prism_math::curve::easing` family; see the module documentation for
/// the id-to-curve mapping and the closed forms.
const EASING_WGSL: &str = r#"
// Easing twin: one thread per query evaluates one of the 14 golden easing
// curves selected by a u32 function id. Branch selection uses u32 id compares
// (not bare f32 equality); each piecewise curve uses an ordered f32 compare fed
// to select. The transcendental curves use the sin/cos/pow built-ins this
// family requires; there is no round, no f32 remainder and no bare f32 equality.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Easing-function id in 0..=13; any other value is rejected.
    func_id: u32,
    // The easing parameter t.
    t: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Eased value for a valid id, else 0.
    eased: f32,
    // 1 when the function id is in 0..=13, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Golden transcendental constants, byte-exact to core::f32::consts after the
// WGSL f32 literal rounding.
const FRAC_PI_2: f32 = 1.5707963267948966;
const PI_CONST: f32 = 3.141592653589793;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let fid = q.func_id;
    let raw = q.t;
    let valid_id = fid <= 13u;

    // Clamp used by the smoothstep family (ids 0 and 1).
    let tc = clamp(raw, 0.0, 1.0);

    var eased = 0.0;
    if (fid == 0u) {
        eased = tc * tc * (3.0 - 2.0 * tc);
    } else if (fid == 1u) {
        eased = tc * tc * tc * (tc * (tc * 6.0 - 15.0) + 10.0);
    } else if (fid == 2u) {
        eased = raw * raw;
    } else if (fid == 3u) {
        eased = raw * (2.0 - raw);
    } else if (fid == 4u) {
        let branch_lo = 2.0 * raw * raw;
        let hv = -2.0 * raw + 2.0;
        let branch_hi = 1.0 - hv * hv * 0.5;
        eased = select(branch_hi, branch_lo, raw < 0.5);
    } else if (fid == 5u) {
        eased = raw * raw * raw;
    } else if (fid == 6u) {
        let uv = 1.0 - raw;
        eased = 1.0 - uv * uv * uv;
    } else if (fid == 7u) {
        let branch_lo = 4.0 * raw * raw * raw;
        let hv = -2.0 * raw + 2.0;
        let branch_hi = 1.0 - hv * hv * hv * 0.5;
        eased = select(branch_hi, branch_lo, raw < 0.5);
    } else if (fid == 8u) {
        eased = 1.0 - cos(raw * FRAC_PI_2);
    } else if (fid == 9u) {
        eased = sin(raw * FRAC_PI_2);
    } else if (fid == 10u) {
        eased = -0.5 * (cos(PI_CONST * raw) - 1.0);
    } else if (fid == 11u) {
        let ev = pow(2.0, 10.0 * (raw - 1.0));
        eased = select(ev, 0.0, raw <= 0.0);
    } else if (fid == 12u) {
        let ev = 1.0 - pow(2.0, -10.0 * raw);
        eased = select(ev, 1.0, raw >= 1.0);
    } else if (fid == 13u) {
        let lo = 0.5 * pow(2.0, 20.0 * raw - 10.0);
        let hi = 1.0 - 0.5 * pow(2.0, -20.0 * raw + 10.0);
        let mid = select(hi, lo, raw < 0.5);
        let hi_clamped = select(mid, 1.0, raw >= 1.0);
        eased = select(hi_clamped, 0.0, raw <= 0.0);
    }

    var out: Result;
    out.eased = select(0.0, eased, valid_id);
    out.valid = select(0u, 1u, valid_id);
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
/// The id and parameter are padded to `4` words (`16` bytes), aligned to `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    func_id: u32,
    t: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the eased value and the validity flag — `2` words (`8` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    eased: f32,
    valid: u32,
}

/// One easing query: the function id and the parameter `t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EasingQuery {
    /// Easing-function id in `0..=13`; any other value is rejected as invalid.
    pub func_id: u32,
    /// The easing parameter `t`.
    pub t: f32,
}

impl EasingQuery {
    /// Builds a query from the function id and the parameter `t`.
    #[must_use]
    pub fn new(func_id: u32, t: f32) -> EasingQuery {
        EasingQuery { func_id, t }
    }
}

/// One resolved answer for a single query, mirroring the reference easing curve
/// for that id and parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EasingResult {
    /// The eased value for a valid id, else `0`.
    pub eased: f32,
    /// `1` when the function id is in `0..=13`, else `0`.
    pub valid: u32,
}

/// Encodes one [`EasingQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &EasingQuery) -> GpuQuery {
    GpuQuery {
        func_id: q.func_id,
        t: q.t,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`EasingResult`].
fn decode_result(raw: &GpuResult) -> EasingResult {
    EasingResult {
        eased: raw.eased,
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

/// A compiled, reusable easing compute pipeline, twinning the `CPU` golden
/// `prism_math::curve::easing` family.
pub struct GpuEasing {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuEasing {
    /// Compiles the easing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus the `sin`,
    /// `cos` and `pow` built-ins this family requires, so no optional device
    /// feature is needed.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuEasing {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_easing"),
            source: ShaderSource::Wgsl(EASING_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_easing_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_easing_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_easing_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuEasing {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`EasingResult`] per
    /// input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `eased` scalar to
    /// the module's tolerance. An empty `queries` batch returns an empty vector
    /// with no dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[EasingQuery]) -> Vec<EasingResult> {
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
            label: Some("prism_volumetric_easing_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_easing_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_easing_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_easing_bind_group"),
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
            label: Some("prism_volumetric_easing_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_easing_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_easing_pass"),
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
