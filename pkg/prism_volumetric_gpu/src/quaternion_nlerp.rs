//! `wgpu` compute twin of the normalized quaternion interpolation
//! (`crate`-external golden `particle::quaternion_rotate::Quat::nlerp`).
//!
//! Normalized linear interpolation (`nlerp`) blends two orientation
//! quaternions along the chord between them and renormalizes the result onto
//! the unit hypersphere. It is the cheap, commutative alternative to spherical
//! `slerp`: no transcendental functions, just a component-wise lerp with a
//! sign flip that selects the shorter arc, followed by one `normalize`.
//!
//! # What is twinned
//!
//! One thread resolves one query. For endpoints `a` and `b` and a parameter
//! `t`, the kernel first forms the four-component dot product
//! `a . b = ax*bx + ay*by + az*bz + aw*bw`; when it is negative the twin flips
//! `b`'s sign so the blend follows the shorter arc. It then lerps each
//! component `blended = a + t * (sign * b - a)` and divides by the blended
//! length. A near-zero blend (length `< 1e-6`, direction undefined) clamps to
//! the identity `(0, 0, 0, 1)`, exactly as the reference `normalize`. The twin
//! spells out the same closed form with the same ordered compares, so a passing
//! real-device parity test is direct evidence the ported kernel interpolates
//! identically, not merely that the shader compiles.
//!
//! # What stays on the host
//!
//! Nothing of the per-query math stays on the host: the whole evaluation is a
//! fixed, bounded sequence of arithmetic and one `sqrt` that runs on device.
//! The host only flattens the query batch into a `std430` storage buffer and
//! short-circuits an empty batch (a storage buffer cannot be zero-sized).
//!
//! # Correctness model
//!
//! Each blended component is a *continuous* quantity, so the parity test
//! compares with an absolute-or-relative tolerance (`abs <= 1e-5 ||
//! rel <= 1e-4`). The sign-flip branch folds into that continuous output;
//! fixtures sit clear of the exactly-antipodal case where the chord collapses.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `max`, `sqrt`,
//! `+ - * /` and ordered comparisons — with no `sin`, `cos`, `exp`, `log`,
//! `pow`, `tan`, no inverse trigonometry, no `round` and no
//! `u64`/`u16`/`i64`/`f64`. No optional device feature is required, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread
//! performs a fixed, bounded sequence of arithmetic, so the kernel provably
//! terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture` 的 `particle::quaternion_rotate::Quat::nlerp`；无第三方引擎源码或衍生代码。
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

/// Minimum blended length below which the direction is undefined and the result
/// clamps to the identity, matching the golden `MIN_LENGTH`.
const MIN_LENGTH: f32 = 1.0e-6;

/// Host-side independent reimplementation of the golden
/// `particle::quaternion_rotate::Quat::nlerp`, reproduced without importing the
/// golden so the twin stays self-contained and free of any cross-crate
/// dependency.
///
/// The sign is `-1` when the four-component dot `a . b` is negative (shorter
/// arc), each component is lerped as `a + t * (sign * b - a)`, and the blend is
/// normalized; a near-zero blend clamps to the identity `(0, 0, 0, 1)`.
#[must_use]
pub fn nlerp_components(
    ax: f32,
    ay: f32,
    az: f32,
    aw: f32,
    bx: f32,
    by: f32,
    bz: f32,
    bw: f32,
    t: f32,
) -> [f32; 4] {
    let dot_ab = ax * bx + ay * by + az * bz + aw * bw;
    let sign = if dot_ab < 0.0 { -1.0 } else { 1.0 };
    let x = ax + t * (sign * bx - ax);
    let y = ay + t * (sign * by - ay);
    let z = az + t * (sign * bz - az);
    let w = aw + t * (sign * bw - aw);
    let len = (x * x + y * y + z * z + w * w).sqrt();
    if len < MIN_LENGTH {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv = 1.0 / len;
    [x * inv, y * inv, z * inv, w * inv]
}

/// The portable core-`WGSL` normalized-interpolation kernel, embedded inline so
/// the twin ships as a single source file. The single entry point `solve`
/// mirrors the golden `Quat::nlerp`; see the module documentation.
const QUATERNION_NLERP_WGSL: &str = r#"
// Normalized quaternion interpolation twin: one thread blends one pair of
// endpoints, mirroring the CPU golden
// `particle::quaternion_rotate::Quat::nlerp` with only max/sqrt, + - * / and
// ordered comparisons. A negative four-component dot flips `b`'s sign to take
// the shorter arc; a near-zero blend clamps to the identity.
//
// Provenance: 孪生自本仓 prism_render_architecture 的
// particle::quaternion_rotate::Quat::nlerp；无第三方引擎源码或衍生代码。

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Endpoint `a` components.
    ax: f32,
    ay: f32,
    az: f32,
    aw: f32,
    // Endpoint `b` components.
    bx: f32,
    by: f32,
    bz: f32,
    bw: f32,
    // Interpolation parameter.
    param: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct NlerpResult {
    // Normalized blended quaternion.
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<NlerpResult>;

const MIN_LENGTH: f32 = 1.0e-6;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Four-component dot picks the shorter arc via a sign flip on `b`.
    let dot_ab = q.ax * q.bx + q.ay * q.by + q.az * q.bz + q.aw * q.bw;
    var sign: f32 = 1.0;
    if (dot_ab < 0.0) {
        sign = -1.0;
    }

    // Component-wise lerp `a + t * (sign * b - a)`.
    let bx = q.ax + q.param * (sign * q.bx - q.ax);
    let by = q.ay + q.param * (sign * q.by - q.ay);
    let bz = q.az + q.param * (sign * q.bz - q.az);
    let bw = q.aw + q.param * (sign * q.bw - q.aw);

    let len = sqrt(bx * bx + by * by + bz * bz + bw * bw);

    var out: NlerpResult;
    if (len < MIN_LENGTH) {
        // Undefined direction clamps to the identity, matching `normalize`.
        out.x = 0.0;
        out.y = 0.0;
        out.z = 0.0;
        out.w = 1.0;
    } else {
        let inv = 1.0 / len;
        out.x = bx * inv;
        out.y = by * inv;
        out.z = bz * inv;
        out.w = bw * inv;
    }
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte uniform struct matching `Params` in
/// [`QUATERNION_NLERP_WGSL`].
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

/// `repr(C)` `std430` layout of one interpolation query: the two endpoint
/// quaternions and the parameter, matching the `WGSL` `Query` struct's
/// `48`-byte stride (nine `f32` plus three pad words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Endpoint `a` `x`.
    ax: f32,
    /// Endpoint `a` `y`.
    ay: f32,
    /// Endpoint `a` `z`.
    az: f32,
    /// Endpoint `a` `w`.
    aw: f32,
    /// Endpoint `b` `x`.
    bx: f32,
    /// Endpoint `b` `y`.
    by: f32,
    /// Endpoint `b` `z`.
    bz: f32,
    /// Endpoint `b` `w`.
    bw: f32,
    /// Interpolation parameter.
    param: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// `repr(C)` `std430` layout of one interpolation result, matching the `WGSL`
/// `NlerpResult` struct: the four normalized components in a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Normalized `x`.
    x: f32,
    /// Normalized `y`.
    y: f32,
    /// Normalized `z`.
    z: f32,
    /// Normalized `w`.
    w: f32,
}

/// One interpolation query: the two endpoint quaternions `a` and `b` plus the
/// parameter `t`.
///
/// The fields mirror the golden `Quat::nlerp` arguments; the host enqueues one
/// query per blend, and an empty batch is short-circuited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuaternionNlerpQuery {
    /// Endpoint `a` `x`.
    pub ax: f32,
    /// Endpoint `a` `y`.
    pub ay: f32,
    /// Endpoint `a` `z`.
    pub az: f32,
    /// Endpoint `a` `w`.
    pub aw: f32,
    /// Endpoint `b` `x`.
    pub bx: f32,
    /// Endpoint `b` `y`.
    pub by: f32,
    /// Endpoint `b` `z`.
    pub bz: f32,
    /// Endpoint `b` `w`.
    pub bw: f32,
    /// Interpolation parameter `t`.
    pub t: f32,
}

impl QuaternionNlerpQuery {
    /// Builds a query from the two endpoint quaternions and the parameter.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "a quaternion pair plus parameter is nine flat scalars by design"
    )]
    pub const fn new(
        ax: f32,
        ay: f32,
        az: f32,
        aw: f32,
        bx: f32,
        by: f32,
        bz: f32,
        bw: f32,
        t: f32,
    ) -> QuaternionNlerpQuery {
        QuaternionNlerpQuery {
            ax,
            ay,
            az,
            aw,
            bx,
            by,
            bz,
            bw,
            t,
        }
    }
}

/// One resolved interpolation: the normalized blended quaternion `(x, y, z, w)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuaternionNlerpResult {
    /// Normalized `x`.
    pub x: f32,
    /// Normalized `y`.
    pub y: f32,
    /// Normalized `z`.
    pub z: f32,
    /// Normalized `w`.
    pub w: f32,
}

/// Encodes one [`QuaternionNlerpQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QuaternionNlerpQuery) -> GpuQuery {
    GpuQuery {
        ax: q.ax,
        ay: q.ay,
        az: q.az,
        aw: q.aw,
        bx: q.bx,
        by: q.by,
        bz: q.bz,
        bw: q.bw,
        param: q.t,
        pad0: 0.0,
        pad1: 0.0,
        pad2: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`QuaternionNlerpResult`].
fn decode_result(raw: &GpuResult) -> QuaternionNlerpResult {
    QuaternionNlerpResult {
        x: raw.x,
        y: raw.y,
        z: raw.z,
        w: raw.w,
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

/// A compiled, reusable normalized-interpolation compute pipeline, twinning the
/// golden `particle::quaternion_rotate::Quat::nlerp`.
pub struct GpuQuaternionNlerp {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuaternionNlerp {
    /// Compiles the normalized-interpolation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuaternionNlerp {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quaternion_nlerp"),
            source: ShaderSource::Wgsl(QUATERNION_NLERP_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quaternion_nlerp_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quaternion_nlerp_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quaternion_nlerp_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuaternionNlerp {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`QuaternionNlerpResult`] per input, in order.
    ///
    /// Each component equals the reference within floating-point tolerance. An
    /// empty `queries` batch returns an empty vector with no dispatch issued,
    /// since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[QuaternionNlerpQuery],
    ) -> Vec<QuaternionNlerpResult> {
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
            label: Some("prism_volumetric_quaternion_nlerp_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quaternion_nlerp_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quaternion_nlerp_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quaternion_nlerp_bind_group"),
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
            label: Some("prism_volumetric_quaternion_nlerp_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quaternion_nlerp_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quaternion_nlerp_pass"),
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
