//! `wgpu` compute twin of the Gaussian-curvature closed form, from the `CPU`
//! golden `prism_physics_core::collider::curvature_tensor`'s
//! `PrincipalCurvature::gaussian`.
//!
//! The Gaussian curvature `K` of a surface vertex is the product of its two
//! principal curvatures, `K = k1 * k2`. This module ports that single
//! stateless closed form onto the device: one thread resolves one `(k1, k2)`
//! pair, so a passing real-device parity test is direct evidence the ported
//! kernel computes the same Gaussian curvature the reference does, not merely
//! that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `gaussian` for one principal-curvature
//! pair with explicit `k1` and `k2`:
//!
//! * If either curvature is non-finite, the pair is invalid (`valid = 0`,
//!   `gaussian = 0`).
//! * Otherwise `gaussian = k1 * k2`. A flat vertex, where `k1 = k2 = 0`,
//!   yields `0 * 0 = 0`, matching the reference.
//!
//! # Correctness model
//!
//! The continuous arithmetic is a single multiply, so `CPU` and `GPU` evaluate
//! the same closed form; the valid `gaussian` scalar is compared with an
//! `abs <= 1e-4 || rel <= 1e-3` tolerance (`REL_FLOOR = 1e-6`). The discrete
//! `valid` flag is compared exactly; the parity test keeps random curvatures
//! finite and well inside range so the validity decision cannot be flipped by
//! round-off.
//!
//! # Degenerate inputs
//!
//! A non-finite `k1` or `k2` yields `valid = 0` with `gaussian = 0`. Before the
//! multiply each curvature is routed through a `select` so an invalid
//! (non-finite) operand cannot taint the taken branch with an `inf * 0 = NaN`;
//! the product is then masked to `0` for an invalid pair. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot
//! be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `*`, `select`
//! and unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no `round`, no `f32` remainder and no `sqrt`, so it runs unmodified on
//! `Metal`, `Vulkan` and `DX12`. Finiteness is tested with the ordered compare
//! `abs(x) < 3.0e38` (which rejects both infinities and `NaN`) rather than a
//! bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::curvature_tensor`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` Gaussian-curvature kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `PrincipalCurvature::gaussian`; see the module
/// documentation for the closed form.
const PRINCIPAL_CURVATURE_GAUSSIAN_WGSL: &str = r#"
// Gaussian-curvature twin: one thread per query reproduces gaussian = k1 * k2.
// It uses only the portable core-WGSL subset (abs, *, select plus unsigned
// index math), takes no optional feature, and has no loop and no branch, so it
// provably terminates. Finiteness is an ordered abs < 3.0e38 compare (rejecting
// infinities and NaN) fed to select; there is no bare f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First (larger-magnitude) principal curvature.
    k1: f32,
    // Second (smaller-magnitude) principal curvature.
    k2: f32,
    // Padding words to a 16-byte-friendly stride.
    pad0: f32,
    pad1: f32,
}

struct Result {
    // Gaussian curvature k1 * k2 when valid, else 0.
    gaussian: f32,
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
    let k1 = q.k1;
    let k2 = q.k2;

    // Finiteness via ordered abs < 3.0e38 (rejects +/-inf and NaN, since every
    // comparison with NaN is false). No bare f32 equality anywhere.
    let k1_ok = abs(k1) < FINITE_LIMIT;
    let k2_ok = abs(k2) < FINITE_LIMIT;
    let ok = k1_ok && k2_ok;

    // Sanitize each operand so an invalid (non-finite) curvature never reaches
    // the multiply as inf * 0 = NaN; the product is masked to 0 when invalid.
    let a = select(0.0, k1, k1_ok);
    let b = select(0.0, k2, k2_ok);
    let g = a * b;

    var out: Result;
    out.gaussian = select(0.0, g, ok);
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

/// `repr(C)` `std430` layout of one query, matching the `WGSL` `Query` struct.
/// The two curvatures are padded to `4` `f32` words (`16` bytes), aligned to
/// `4`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    k1: f32,
    k2: f32,
    pad0: f32,
    pad1: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the Gaussian curvature and the validity flag — `2` words (`8`
/// bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    gaussian: f32,
    valid: u32,
}

/// One Gaussian-curvature query: the two principal curvatures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipalCurvatureGaussianQuery {
    /// First (larger-magnitude) principal curvature.
    pub k1: f32,
    /// Second (smaller-magnitude) principal curvature.
    pub k2: f32,
}

impl PrincipalCurvatureGaussianQuery {
    /// Builds a query from the two principal curvatures.
    #[must_use]
    pub fn new(k1: f32, k2: f32) -> PrincipalCurvatureGaussianQuery {
        PrincipalCurvatureGaussianQuery { k1, k2 }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `PrincipalCurvature::gaussian` output for that curvature pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrincipalCurvatureGaussianResult {
    /// The Gaussian curvature `k1 * k2` when valid, else `0`.
    pub gaussian: f32,
    /// `1` when both curvatures are finite, else `0`.
    pub valid: u32,
}

/// Encodes one [`PrincipalCurvatureGaussianQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &PrincipalCurvatureGaussianQuery) -> GpuQuery {
    GpuQuery {
        k1: q.k1,
        k2: q.k2,
        pad0: 0.0,
        pad1: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`PrincipalCurvatureGaussianResult`].
fn decode_result(raw: &GpuResult) -> PrincipalCurvatureGaussianResult {
    PrincipalCurvatureGaussianResult {
        gaussian: raw.gaussian,
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

/// A compiled, reusable Gaussian-curvature compute pipeline, twinning the `CPU`
/// golden `PrincipalCurvature::gaussian`.
pub struct GpuPrincipalCurvatureGaussian {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPrincipalCurvatureGaussian {
    /// Compiles the Gaussian-curvature kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPrincipalCurvatureGaussian {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_principal_curvature_gaussian"),
            source: ShaderSource::Wgsl(PRINCIPAL_CURVATURE_GAUSSIAN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_principal_curvature_gaussian_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_principal_curvature_gaussian_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_principal_curvature_gaussian_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPrincipalCurvatureGaussian {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one
    /// [`PrincipalCurvatureGaussianResult`] per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `gaussian` scalar
    /// to the module's tolerance. An empty `queries` batch returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[PrincipalCurvatureGaussianQuery],
    ) -> Vec<PrincipalCurvatureGaussianResult> {
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
            label: Some("prism_volumetric_principal_curvature_gaussian_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_principal_curvature_gaussian_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_principal_curvature_gaussian_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_principal_curvature_gaussian_bind_group"),
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
            label: Some("prism_volumetric_principal_curvature_gaussian_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_principal_curvature_gaussian_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_principal_curvature_gaussian_pass"),
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
