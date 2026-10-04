//! `wgpu` compute twin of the mean-curvature closed form, from the `CPU`
//! golden `prism_physics_core::collider::curvature_tensor`'s
//! `PrincipalCurvature::mean`.
//!
//! Given the two principal curvature magnitudes `k1` and `k2` at a surface
//! point, the mean curvature is `H = 0.5 * (k1 + k2)`. The golden `mean`
//! computes exactly that; the curvature-tensor eigen-decomposition that
//! produces `k1` and `k2` is out of scope here. This module ports the
//! stateless, no-`RNG`, branch-free average onto the device: one compute thread
//! resolves one `(k1, k2)` pair, so a passing real-device parity test is direct
//! evidence the kernel reproduces the exact average, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Each query is one curvature pair `k1` and `k2`. The kernel reproduces the
//! reference closed form `mean = 0.5 * (k1 + k2)`.
//!
//! # Correctness model
//!
//! The value is one add and one multiply, so `CPU` and `GPU` are not required
//! to be bit-exact. The parity test asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on `mean`, and compares the discrete
//! `valid` flag exactly.
//!
//! # Degenerate inputs
//!
//! A non-finite `k1` or `k2` (an infinity or `NaN`) yields `valid = 0` with
//! `mean = 0`; this matches the golden's finite inputs, and the all-zero flat
//! vertex (`k1 = k2 = 0`) maps to `mean = 0` as well. Finiteness is an ordered
//! `abs(x) < 3.0e38` compare rather than a bare `x == x`, so no fast-math
//! sentinel is involved; the un-taken branch is fed through `select`. An empty
//! query batch short-circuits on the host with no dispatch, since a storage
//! buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `+`, `*`,
//! `select` plus unsigned index arithmetic — with no `sin`, `cos`, `tan`,
//! `exp`, `log`, `pow`, no `sqrt`, no `round`, no float modulo and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::curvature_tensor::PrincipalCurvature::mean`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` mean-curvature kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden `PrincipalCurvature::mean`; see the module documentation for
/// the closed form.
const PRINCIPAL_CURVATURE_MEAN_WGSL: &str = r#"
// Mean-curvature twin: one thread per curvature pair reproduces the mean
// curvature the golden PrincipalCurvature::mean forms as 0.5 * (k1 + k2). It
// uses only the portable core-WGSL subset (abs, + *, select plus unsigned
// index math), takes no optional feature, and has no loop and no branch, so it
// provably terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN, since every comparison with NaN is false), fed to select;
// there is no bare f32 equality anywhere.
//
// Provenance: 孪生自本仓
// prism_physics_core::collider::curvature_tensor::PrincipalCurvature::mean；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of curvature pairs in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Larger-magnitude principal curvature.
    k1: f32,
    // Smaller-magnitude principal curvature.
    k2: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Mean curvature 0.5 * (k1 + k2) when valid, else 0.
    mean: f32,
    // 1 when both curvatures are finite, else 0.
    valid: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

const FINITE_LIMIT: f32 = 3.0e38;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let k1_finite = abs(q.k1) < FINITE_LIMIT;
    let k2_finite = abs(q.k2) < FINITE_LIMIT;
    let ok = k1_finite && k2_finite;

    // Sanitize each curvature so the un-taken (invalid) branch never carries a
    // NaN or infinity into the average.
    let k1 = select(0.0, q.k1, k1_finite);
    let k2 = select(0.0, q.k2, k2_finite);
    let mean = 0.5 * (k1 + k2);

    var out: Result;
    out.mean = select(0.0, mean, ok);
    out.valid = select(0u, 1u, ok);
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
/// the two principal curvatures padded to a `16`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    k1: f32,
    k2: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the mean curvature and its validity flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    mean: f32,
    valid: u32,
}

/// One mean-curvature query: the two principal curvature magnitudes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipalCurvatureMeanQuery {
    /// Larger-magnitude principal curvature `k1`.
    pub k1: f32,
    /// Smaller-magnitude principal curvature `k2`.
    pub k2: f32,
}

impl PrincipalCurvatureMeanQuery {
    /// Builds a query from the two principal curvatures.
    #[must_use]
    pub fn new(k1: f32, k2: f32) -> PrincipalCurvatureMeanQuery {
        PrincipalCurvatureMeanQuery { k1, k2 }
    }
}

/// One resolved answer for a single pair: the mean curvature and whether both
/// inputs were finite, mirroring the reference `PrincipalCurvature::mean`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipalCurvatureMeanResult {
    /// Mean curvature `0.5 * (k1 + k2)` when valid, else `0`.
    pub mean: f32,
    /// `1` when both curvatures are finite, else `0`.
    pub valid: u32,
}

/// Encodes one [`PrincipalCurvatureMeanQuery`] into its `std430` [`GpuQuery`]
/// slot.
fn encode_query(q: &PrincipalCurvatureMeanQuery) -> GpuQuery {
    GpuQuery {
        k1: q.k1,
        k2: q.k2,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`PrincipalCurvatureMeanResult`].
fn decode_result(raw: &GpuResult) -> PrincipalCurvatureMeanResult {
    PrincipalCurvatureMeanResult {
        mean: raw.mean,
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

/// A compiled, reusable mean-curvature compute pipeline, twinning the `CPU`
/// golden `PrincipalCurvature::mean`.
pub struct GpuPrincipalCurvatureMean {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPrincipalCurvatureMean {
    /// Compiles the mean-curvature kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPrincipalCurvatureMean {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_principal_curvature_mean"),
            source: ShaderSource::Wgsl(PRINCIPAL_CURVATURE_MEAN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_principal_curvature_mean_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_principal_curvature_mean_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_principal_curvature_mean_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPrincipalCurvatureMean {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every curvature pair in `queries` and returns one
    /// [`PrincipalCurvatureMeanResult`] per input, in order.
    ///
    /// `mean` matches the reference to within the tolerance documented on this
    /// module and `valid` matches exactly. An empty `queries` batch returns an
    /// empty vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[PrincipalCurvatureMeanQuery],
    ) -> Vec<PrincipalCurvatureMeanResult> {
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
            label: Some("prism_volumetric_principal_curvature_mean_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_principal_curvature_mean_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_principal_curvature_mean_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_principal_curvature_mean_bind_group"),
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
            label: Some("prism_volumetric_principal_curvature_mean_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_principal_curvature_mean_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_principal_curvature_mean_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per curvature pair, flattened to a 1-D dispatch.
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
