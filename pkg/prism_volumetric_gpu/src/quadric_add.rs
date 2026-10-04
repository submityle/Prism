//! `wgpu` compute twin of the quadric-accumulation helper from the `CPU` golden
//! `prism_physics_core::collider::quadric::Quadric::add`.
//!
//! A Garland--Heckbert error quadric is the symmetric `4x4` matrix stored as its
//! ten distinct upper-triangular coefficients `a2, ab, ac, ad, b2, bc, bd, c2,
//! cd, d2`. Quadrics are additive: merging two vertices, or accumulating the
//! plane quadrics of a vertex's incident faces, sums their coefficients. The
//! golden `Quadric::add` does exactly that — ten independent scalar additions in
//! a fixed order. This module ports that stateless, no-`RNG`, branch-free closed
//! form onto the device: one compute thread adds one pair of quadrics, so a
//! passing real-device parity test is direct evidence the kernel reproduces the
//! exact coefficient order, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query is one pair of quadrics `qa` and `qb`. The kernel reproduces the
//! reference closed form coefficient for coefficient:
//!
//! * `a2 = qa.a2 + qb.a2`, `ab = qa.ab + qb.ab`, `ac = qa.ac + qb.ac`,
//!   `ad = qa.ad + qb.ad`;
//! * `b2 = qa.b2 + qb.b2`, `bc = qa.bc + qb.bc`, `bd = qa.bd + qb.bd`;
//! * `c2 = qa.c2 + qb.c2`, `cd = qa.cd + qb.cd`, `d2 = qa.d2 + qb.d2`.
//!
//! There is no division and no branch: the computation is pure addition, so the
//! kernel provably terminates and `valid` is always `1`.
//!
//! # Correctness model
//!
//! Each coefficient is a single add, so `CPU` and `GPU` are not required to be
//! bit-exact. The parity test asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`) on each of the ten coefficients; the
//! discrete `valid` flag is compared exactly. The kernel has no comparisons at
//! all, so no bare float equality and no fast-math `NaN` sentinel is involved.
//!
//! # Degenerate inputs
//!
//! There are no degenerate inputs: every pair of quadrics yields ten
//! well-defined sums, so `valid` is always `1`. An empty query batch
//! short-circuits on the host with no dispatch, since a storage buffer cannot be
//! zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `+` on `f32` and
//! unsigned index arithmetic — with no `sin`, `cos`, `tan`, `exp`, `log`,
//! `pow`, no `round`, no float modulo and no optional device feature, so it runs
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::quadric::Quadric::add`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` quadric-accumulation kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `add`; see the module documentation for the algorithm.
const QUADRIC_ADD_WGSL: &str = r#"
// Quadric-add twin: one thread per pair reproduces the ten upper-triangular
// quadric coefficients the golden Quadric::add forms by summing two quadrics
// coefficient for coefficient. It mirrors the CPU golden operation for
// operation, uses only the portable core-WGSL subset (f32 addition plus
// unsigned index math), takes no optional feature, and has a fixed-trip loop,
// so the kernel provably terminates. There is no division and no branch, so no
// float equality or fast-math sentinel is involved.
//
// Provenance: 孪生自本仓
// prism_physics_core::collider::quadric::Quadric::add；
// 无第三方引擎源码或衍生代码。

struct Params {
    // Number of quadric pairs in the storage arrays; threads past this
    // short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // The ten coefficients of the first quadric, in the golden's order:
    // a2, ab, ac, ad, b2, bc, bd, c2, cd, d2.
    lhs: array<f32, 10>,
    // The ten coefficients of the second quadric, same order.
    rhs: array<f32, 10>,
}

struct Result {
    // The ten summed coefficients, in the golden's order.
    coeffs: array<f32, 10>,
    // Always 1: the closed form has no degenerate branch.
    valid: u32,
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

    var out: Result;
    for (var i = 0u; i < 10u; i = i + 1u) {
        out.coeffs[i] = q.lhs[i] + q.rhs[i];
    }
    out.valid = 1u;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`QUADRIC_ADD_WGSL`].
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
/// Both coefficient blocks are flat scalar `f32` arrays with element stride `4`,
/// so the `80`-byte slot is `4`-byte aligned and the host and device agree on
/// the array stride byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    lhs: [f32; 10],
    rhs: [f32; 10],
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct. The ten coefficients are stored as a flat scalar array in the
/// golden's order; the trailing `valid` word keeps the discrete flag beside
/// them, matching the `44`-byte device struct byte for byte.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    coeffs: [f32; 10],
    valid: u32,
}

/// One query for the quadric-add twin: the two quadrics to sum, each as its ten
/// upper-triangular coefficients in the golden's order
/// `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricAddQuery {
    /// The first quadric's ten coefficients.
    pub lhs: [f32; 10],
    /// The second quadric's ten coefficients.
    pub rhs: [f32; 10],
}

impl QuadricAddQuery {
    /// Builds a query from the two quadrics' coefficient arrays.
    #[must_use]
    pub fn new(lhs: [f32; 10], rhs: [f32; 10]) -> QuadricAddQuery {
        QuadricAddQuery { lhs, rhs }
    }
}

/// One resolved answer for a single pair: the ten summed quadric coefficients in
/// the golden's order, plus the `valid` flag (always `1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadricAddResult {
    /// The ten summed coefficients, in the golden's order:
    /// `a2, ab, ac, ad, b2, bc, bd, c2, cd, d2`.
    pub coeffs: [f32; 10],
    /// Always `1`: the closed form has no degenerate branch.
    pub valid: u32,
}

/// Encodes one [`QuadricAddQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &QuadricAddQuery) -> GpuQuery {
    GpuQuery {
        lhs: q.lhs,
        rhs: q.rhs,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`QuadricAddResult`].
fn decode_result(raw: &GpuResult) -> QuadricAddResult {
    QuadricAddResult {
        coeffs: raw.coeffs,
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

/// A compiled, reusable quadric-add compute pipeline, twinning the `CPU` golden
/// `Quadric::add`.
pub struct GpuQuadricAdd {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuQuadricAdd {
    /// Compiles the quadric-add kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuQuadricAdd {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_quadric_add"),
            source: ShaderSource::Wgsl(QUADRIC_ADD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_quadric_add_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_quadric_add_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_quadric_add_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuQuadricAdd {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every quadric pair in `queries` and returns one
    /// [`QuadricAddResult`] per input, in order.
    ///
    /// Each coefficient matches the reference to within the tolerance
    /// documented on this module; the `valid` flag matches exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[QuadricAddQuery]) -> Vec<QuadricAddResult> {
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
            label: Some("prism_volumetric_quadric_add_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_quadric_add_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_quadric_add_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_quadric_add_bind_group"),
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
            label: Some("prism_volumetric_quadric_add_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_quadric_add_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_quadric_add_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per quadric pair, flattened to a 1-D dispatch.
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
