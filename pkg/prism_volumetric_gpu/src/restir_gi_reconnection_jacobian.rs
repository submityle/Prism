//! `wgpu` compute twin of the `ReSTIR` GI reconnection Jacobian
//! ([`reconnection_jacobian`](prism_render_architecture::lighting::restir_gi::reconnection_jacobian)).
//!
//! Importing an indirect-illumination sample point into a different shading
//! point changes the solid angle that fixed world-space point subtends, so the
//! reuse weight must be multiplied by a change-of-measure factor — the
//! *reconnection Jacobian* — to keep the `ReSTIR` GI estimator unbiased across
//! pixels with different geometry. That factor is a stateless, closed-form ratio
//! of cosine-weighted inverse-square terms:
//!
//! ```text
//!       cosθ_dst / ‖x_dst − x_s‖²
//! J  =  ─────────────────────────
//!       cosθ_src / ‖x_src − x_s‖²
//! ```
//!
//! where `θ` is measured at the sample point's normal toward each shading point.
//!
//! [`GpuRestirGiReconnectionJacobian`] is the on-device twin that runs one
//! thread per `(src, dst, sample)` triple and reproduces that ratio step for
//! step, so a passing real-device parity test is direct evidence the ported
//! kernel computes the same Jacobian the reference does, not merely that the
//! shader compiles.
//!
//! # What is twinned
//!
//! For one triple the kernel reproduces the full
//! [`reconnection_jacobian`](prism_render_architecture::lighting::restir_gi::reconnection_jacobian)
//! closed form: the two reconnection vectors `x_src − x_s` and `x_dst − x_s`,
//! their squared lengths, the degenerate rejection when either squared length
//! or the source density falls below `GEOM_EPS` (returning `0`), the two
//! sample-normal cosines, and the final `num / den` ratio.
//!
//! # What is not twinned (host-only)
//!
//! The surrounding reservoir machinery — the weighted-reservoir update, the
//! `M`-cap, the unbiased weight `W`, and the generic `combine_*` passes that
//! take a caller target-function closure — is stateful or closure-parameterized
//! and stays on the host. Only the pure, stateless Jacobian is twinned here.
//! The `radiance` payload of the sample is irrelevant to the Jacobian and is
//! not uploaded.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+ - * /`, `abs`,
//! `max`, `sqrt` and the ordered comparisons — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, inverse trigonometry, `smoothstep`, `round` or `cbrt`, and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. There is no loop: each thread performs a fixed, bounded sequence of
//! arithmetic, so the kernel provably terminates.
//!
//! The degenerate guards are expressed without an `f32` equality: the compute
//! path runs only when `d_src2 >= GEOM_EPS && d_dst2 >= GEOM_EPS` and then only
//! when `den >= GEOM_EPS`, the exact negation of the golden
//! `d_src2 < GEOM_EPS || d_dst2 < GEOM_EPS` and `den < GEOM_EPS` early returns,
//! so the result stays `0` on a degenerate reconnection.
//!
//! # Correctness model
//!
//! Away from a `GEOM_EPS` boundary the branch taken is identical on both sides,
//! so the only divergence is a last-place rounding difference in a shared
//! `+ - * / sqrt` sequence. The parity test asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the Jacobian and keeps its
//! random fixtures clear of the degenerate thresholds (well-separated points,
//! non-grazing normals) so the rejection branch cannot be flipped by a
//! last-place difference; the deliberate degenerate fixtures return an exact
//! `0` on both sides.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_gi::reconnection_jacobian`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` reconnection-Jacobian kernel, embedded inline so the
/// twin ships no external shader asset. It reproduces the
/// [`reconnection_jacobian`](prism_render_architecture::lighting::restir_gi::reconnection_jacobian)
/// closed form; see the module documentation for the algorithm.
const RESTIR_GI_RECONNECTION_JACOBIAN_WGSL: &str = r#"
// ReSTIR GI reconnection-Jacobian twin: one thread maps one (src, dst, sample)
// triple through the cosine-weighted inverse-square ratio, mirroring the CPU
// golden `lighting::restir_gi::reconnection_jacobian` with only + - * /, abs,
// max, sqrt and ordered comparisons.
//
// Provenance: 孪生自本仓 prism_render_architecture::lighting::restir_gi::reconnection_jacobian；
// 无第三方引擎源码或衍生代码。

// Below this (squared) distance or source density the reconnection is treated
// as degenerate and the Jacobian is zero, matching the golden GEOM_EPS.
const GEOM_EPS: f32 = 1.0e-8;

struct Params {
    // Number of triples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Source shading-point position x_src.
    sx: f32,
    sy: f32,
    sz: f32,
    // Destination shading-point position x_dst.
    dx: f32,
    dy: f32,
    dz: f32,
    // Sample point x_s.
    px: f32,
    py: f32,
    pz: f32,
    // Unit sample-point normal.
    nx: f32,
    ny: f32,
    nz: f32,
}

struct Res {
    jacobian: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

@compute @workgroup_size(64)
fn jacobian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // Reconnection vectors x_src - x_s and x_dst - x_s.
    let tsx = q.sx - q.px;
    let tsy = q.sy - q.py;
    let tsz = q.sz - q.pz;
    let tdx = q.dx - q.px;
    let tdy = q.dy - q.py;
    let tdz = q.dz - q.pz;

    let d_src2 = tsx * tsx + tsy * tsy + tsz * tsz;
    let d_dst2 = tdx * tdx + tdy * tdy + tdz * tdz;

    var j: f32 = 0.0;
    // Negation of the golden `d_src2 < GEOM_EPS || d_dst2 < GEOM_EPS` early 0.
    if (d_src2 >= GEOM_EPS && d_dst2 >= GEOM_EPS) {
        let d_src = sqrt(d_src2);
        let d_dst = sqrt(d_dst2);
        // Cosine at the sample-point normal toward each shading point.
        let cos_src = abs(q.nx * tsx + q.ny * tsy + q.nz * tsz) / d_src;
        let cos_dst = abs(q.nx * tdx + q.ny * tdy + q.nz * tdz) / d_dst;
        let den = cos_src / d_src2;
        // Negation of the golden `den < GEOM_EPS` early 0.
        if (den >= GEOM_EPS) {
            let num = cos_dst / d_dst2;
            j = num / den;
        }
    }

    var out: Res;
    out.jacobian = j;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the triple count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`RESTIR_GI_RECONNECTION_JACOBIAN_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid triples in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// `repr(C)` `std430` layout of one Jacobian query: the two shading-point
/// positions, the sample point and the sample normal, a `48`-byte scalar-packed
/// stride matching the `WGSL` `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Source position `x`.
    sx: f32,
    /// Source position `y`.
    sy: f32,
    /// Source position `z`.
    sz: f32,
    /// Destination position `x`.
    dx: f32,
    /// Destination position `y`.
    dy: f32,
    /// Destination position `z`.
    dz: f32,
    /// Sample point `x`.
    px: f32,
    /// Sample point `y`.
    py: f32,
    /// Sample point `z`.
    pz: f32,
    /// Sample normal `x`.
    nx: f32,
    /// Sample normal `y`.
    ny: f32,
    /// Sample normal `z`.
    nz: f32,
}

/// `repr(C)` `std430` layout of one Jacobian result, matching the `WGSL` `Res`
/// struct: the single scalar Jacobian, a `4`-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The reconnection Jacobian (`0` on a degenerate reconnection).
    jacobian: f32,
}

/// One reconnection-Jacobian query: the source and destination shading-point
/// positions, the sample point `x_s` and its unit normal.
///
/// The sample radiance plays no part in the Jacobian and is omitted. A
/// degenerate reconnection (coincident points or a grazing sample normal)
/// resolves to `0`, exactly as the golden
/// [`reconnection_jacobian`](prism_render_architecture::lighting::restir_gi::reconnection_jacobian)
/// resolves it.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_gi::reconnection_jacobian`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirGiReconnectionJacobianQuery {
    /// Source shading-point world position `x_src`.
    pub src_position: [f32; 3],
    /// Destination shading-point world position `x_dst`.
    pub dst_position: [f32; 3],
    /// Sample point world position `x_s`.
    pub sample_point: [f32; 3],
    /// Unit surface normal at the sample point.
    pub sample_normal: [f32; 3],
}

impl RestirGiReconnectionJacobianQuery {
    /// Builds a query from the source / destination positions, the sample point
    /// and its normal.
    #[must_use]
    pub const fn new(
        src_position: [f32; 3],
        dst_position: [f32; 3],
        sample_point: [f32; 3],
        sample_normal: [f32; 3],
    ) -> RestirGiReconnectionJacobianQuery {
        RestirGiReconnectionJacobianQuery {
            src_position,
            dst_position,
            sample_point,
            sample_normal,
        }
    }
}

/// One triple's resolved reconnection Jacobian, mirroring the `CPU` golden
/// [`reconnection_jacobian`](prism_render_architecture::lighting::restir_gi::reconnection_jacobian).
///
/// `jacobian` is the cosine-weighted inverse-square ratio converting the source
/// pixel's solid-angle density into the destination pixel's; it is `0` for a
/// degenerate reconnection.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::lighting::restir_gi::reconnection_jacobian`；无第三方引擎源码或衍生代码。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestirGiReconnectionJacobianResult {
    /// The reconnection Jacobian (`0` on a degenerate reconnection).
    pub jacobian: f32,
}

/// Encodes one [`RestirGiReconnectionJacobianQuery`] into its `std430`
/// [`GpuQuery`] slot.
fn encode_query(q: &RestirGiReconnectionJacobianQuery) -> GpuQuery {
    GpuQuery {
        sx: q.src_position[0],
        sy: q.src_position[1],
        sz: q.src_position[2],
        dx: q.dst_position[0],
        dy: q.dst_position[1],
        dz: q.dst_position[2],
        px: q.sample_point[0],
        py: q.sample_point[1],
        pz: q.sample_point[2],
        nx: q.sample_normal[0],
        ny: q.sample_normal[1],
        nz: q.sample_normal[2],
    }
}

/// Decodes one packed [`GpuResult`] into the public
/// [`RestirGiReconnectionJacobianResult`].
fn decode_result(raw: &GpuResult) -> RestirGiReconnectionJacobianResult {
    RestirGiReconnectionJacobianResult {
        jacobian: raw.jacobian,
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

/// A compiled, reusable reconnection-Jacobian compute pipeline, twinning the
/// `CPU` golden
/// [`reconnection_jacobian`](prism_render_architecture::lighting::restir_gi::reconnection_jacobian).
pub struct GpuRestirGiReconnectionJacobian {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRestirGiReconnectionJacobian {
    /// Compiles the reconnection-Jacobian kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRestirGiReconnectionJacobian {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian"),
            source: ShaderSource::Wgsl(RESTIR_GI_RECONNECTION_JACOBIAN_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("jacobian"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRestirGiReconnectionJacobian {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every triple in `queries` and returns one
    /// [`RestirGiReconnectionJacobianResult`] per input, in order.
    ///
    /// Each Jacobian equals the matching `CPU` golden
    /// [`reconnection_jacobian`](prism_render_architecture::lighting::restir_gi::reconnection_jacobian)
    /// to within a last-place rounding slack for a non-degenerate reconnection,
    /// and is an exact `0` for a degenerate one. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[RestirGiReconnectionJacobianQuery],
    ) -> Vec<RestirGiReconnectionJacobianResult> {
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
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian_bind_group"),
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
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_restir_gi_reconnection_jacobian_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_restir_gi_reconnection_jacobian_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per triple, flattened to a 1-D dispatch.
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
