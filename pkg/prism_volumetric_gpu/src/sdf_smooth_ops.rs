//! `wgpu` compute twin of the stateless signed-distance-field (`SDF`) boolean
//! and polynomial-smooth combinators from the capsule `SDF` contract
//! ([`capsule_sdf`](prism_render_architecture::particle::capsule_sdf)).
//!
//! The `CPU` golden exposes six pure scalar combinators that fold two signed
//! distances `d1`, `d2` (and, for the smooth variants, a blend radius `k`) into
//! one signed distance:
//! [`op_union`](prism_render_architecture::particle::capsule_sdf::op_union),
//! [`op_subtract`](prism_render_architecture::particle::capsule_sdf::op_subtract),
//! [`op_intersect`](prism_render_architecture::particle::capsule_sdf::op_intersect),
//! [`op_smooth_union`](prism_render_architecture::particle::capsule_sdf::op_smooth_union),
//! [`op_smooth_subtract`](prism_render_architecture::particle::capsule_sdf::op_smooth_subtract)
//! and
//! [`op_smooth_intersect`](prism_render_architecture::particle::capsule_sdf::op_smooth_intersect).
//! Each is a fixed, bounded sequence of `min`/`max`/`clamp`/`mix` arithmetic
//! with no loop, no container and no transcendental, so each is a clean
//! one-thread-per-query device twin.
//!
//! [`GpuSdfSmoothOps`] evaluates all six combinators for one `(d1, d2, k)`
//! triple per thread, reproducing the reference's exact closed form so a
//! passing real-device parity test is direct evidence the ported kernel folds
//! distances the same way the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! For a query `(d1, d2, k)` the twin returns all six folded distances:
//!
//! - the hard union `min(d1, d2)`,
//! - the hard subtraction `max(-d1, d2)`,
//! - the hard intersection `max(d1, d2)`,
//! - the smooth union, subtraction and intersection, each of which falls back
//!   to its hard counterpart when `k <= CMP_EPS` (`CMP_EPS = 1.0e-6`) and
//!   otherwise blends with the clamped interpolant `h` and the quadratic
//!   `k * h * (1 - h)` correction, where `mix(a, b, t) = a + (b - a) * t`.
//!
//! # What stays on the host
//!
//! Nothing of the six combinators themselves stays on the host: each is a pure
//! fixed-width scalar fold. The surrounding field-sampling, capsule-geometry
//! and variable-length `SDF`-tree work in
//! [`capsule_sdf`](prism_render_architecture::particle::capsule_sdf) is not part
//! of this twin; the host enqueues one triple per combinator evaluation it
//! wants, so a storage buffer is never zero-sized.
//!
//! # Correctness model
//!
//! Every output is a continuous distance threaded through a subtract, a divide
//! by `k`, a `clamp` and a `mix`, so `CPU` and `GPU` are not bit-exact: a `GPU`
//! fused multiply-add or divide may land a few units in the last place from the
//! scalar reference. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-5` or `rel_diff <= 1e-4`) on each of the six outputs, tight
//! enough to catch a genuinely wrong port (a dropped negation, a swapped
//! `mix` order, a wrong `h` sign) yet loose enough to admit a legal last-place
//! difference. The `k <= CMP_EPS` fallback is a magnitude comparison, so both
//! sides take the same branch for fixtures kept clear of the `CMP_EPS`
//! threshold.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `mix` and `+ - * /` — with no `sin`, `cos`, `exp`, `log`, `pow`,
//! `sqrt`, `tan`, no inverse trigonometry, no `round` and no `ceil`, and no wide
//! integer type. No optional device feature is required, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`. There is no loop: each thread performs a
//! fixed, bounded sequence of arithmetic, so the kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::capsule_sdf`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `SDF` combinator kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`capsule_sdf`](prism_render_architecture::particle::capsule_sdf) scalar
/// combinators; see the module documentation for the algorithm.
const SDF_SMOOTH_OPS_WGSL: &str = r#"
// SDF boolean and polynomial-smooth combinator twin: one thread folds one
// (d1, d2, k) triple into the six reference distances, mirroring the CPU golden
// `particle::capsule_sdf` closed form with only min/max/clamp/mix and + - * /.
// It owns no field sampling and no variable-length SDF tree; those stay on the
// host.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::capsule_sdf；无第三方
// 引擎源码或衍生代码。

// Blend-radius epsilon below which the smooth combinators fall back to their
// hard counterparts, matching the reference `CMP_EPS`.
const CMP_EPS: f32 = 1.0e-6;

struct Params {
    // Number of triples in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // First signed distance d1.
    d1: f32,
    // Second signed distance d2.
    d2: f32,
    // Blend radius k (k <= CMP_EPS folds the smooth ops to the hard ops).
    k: f32,
    pad0: f32,
}

struct Outcome {
    // Hard union min(d1, d2).
    hard_union: f32,
    // Hard subtraction max(-d1, d2).
    hard_subtract: f32,
    // Hard intersection max(d1, d2).
    hard_intersect: f32,
    // Polynomial smooth union.
    smooth_union: f32,
    // Polynomial smooth subtraction.
    smooth_subtract: f32,
    // Polynomial smooth intersection.
    smooth_intersect: f32,
    pad0: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Outcome>;

// Hard union: the point belongs to whichever solid is nearer.
fn op_union(d1: f32, d2: f32) -> f32 {
    return min(d1, d2);
}

// Hard subtraction d2 \ d1: carve the first solid out of the second.
fn op_subtract(d1: f32, d2: f32) -> f32 {
    return max(-d1, d2);
}

// Hard intersection: the point must be inside both solids.
fn op_intersect(d1: f32, d2: f32) -> f32 {
    return max(d1, d2);
}

// Polynomial smooth union; falls back to the hard union when k <= CMP_EPS.
fn op_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    if (k <= CMP_EPS) {
        return op_union(d1, d2);
    }
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// Polynomial smooth subtraction; falls back to the hard subtraction when
// k <= CMP_EPS.
fn op_smooth_subtract(d1: f32, d2: f32, k: f32) -> f32 {
    if (k <= CMP_EPS) {
        return op_subtract(d1, d2);
    }
    let h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

// Polynomial smooth intersection; falls back to the hard intersection when
// k <= CMP_EPS.
fn op_smooth_intersect(d1: f32, d2: f32, k: f32) -> f32 {
    if (k <= CMP_EPS) {
        return op_intersect(d1, d2);
    }
    let h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let d1 = q.d1;
    let d2 = q.d2;
    let k = q.k;

    var out: Outcome;
    out.hard_union = op_union(d1, d2);
    out.hard_subtract = op_subtract(d1, d2);
    out.hard_intersect = op_intersect(d1, d2);
    out.smooth_union = op_smooth_union(d1, d2, k);
    out.smooth_subtract = op_smooth_subtract(d1, d2, k);
    out.smooth_intersect = op_smooth_intersect(d1, d2, k);
    out.pad0 = 0.0;
    out.pad1 = 0.0;
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the triple count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`SDF_SMOOTH_OPS_WGSL`].
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

/// `repr(C)` `std430` layout of one combinator query: two signed distances and
/// a blend radius plus one pad word to a `16`-byte stride, matching the `WGSL`
/// `Query` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First signed distance `d1`.
    d1: f32,
    /// Second signed distance `d2`.
    d2: f32,
    /// Blend radius `k`.
    k: f32,
    /// Padding word.
    pad0: f32,
}

/// `repr(C)` `std430` layout of one combinator result, matching the `WGSL`
/// `Outcome` struct: the six folded distances plus two pad words to a `32`-byte
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Hard union `min(d1, d2)`.
    hard_union: f32,
    /// Hard subtraction `max(-d1, d2)`.
    hard_subtract: f32,
    /// Hard intersection `max(d1, d2)`.
    hard_intersect: f32,
    /// Polynomial smooth union.
    smooth_union: f32,
    /// Polynomial smooth subtraction.
    smooth_subtract: f32,
    /// Polynomial smooth intersection.
    smooth_intersect: f32,
    /// Padding word.
    pad0: f32,
    /// Padding word.
    pad1: f32,
}

/// One combinator query for the `SDF` smooth-ops twin: the two signed distances
/// `d1`, `d2` and the blend radius `k`.
///
/// The host owns the surrounding field sampling and the variable-length `SDF`
/// tree, and enqueues one [`SdfSmoothOpsQuery`] per combinator evaluation it
/// wants, matching the reference
/// [`capsule_sdf`](prism_render_architecture::particle::capsule_sdf) scalar
/// combinators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSmoothOpsQuery {
    /// First signed distance `d1`.
    pub d1: f32,
    /// Second signed distance `d2`.
    pub d2: f32,
    /// Blend radius `k`; `k <= 1.0e-6` folds the smooth ops to the hard ops.
    pub k: f32,
}

impl SdfSmoothOpsQuery {
    /// Builds a query for the two signed distances `d1`, `d2` and blend radius
    /// `k`.
    #[must_use]
    pub const fn new(d1: f32, d2: f32, k: f32) -> SdfSmoothOpsQuery {
        SdfSmoothOpsQuery { d1, d2, k }
    }
}

/// One resolved combinator fold, mirroring the six scalar outputs the reference
/// [`capsule_sdf`](prism_render_architecture::particle::capsule_sdf) combinators
/// produce for a `(d1, d2, k)` triple.
///
/// The `union`, `subtract` and `intersect` fields are renamed `hard_union`,
/// `hard_subtract` and `hard_intersect` because `union` is a reserved word in
/// both `WGSL` and `Rust`; the smooth variants keep their descriptive names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfSmoothOpsResult {
    /// Hard union `min(d1, d2)` (reference `op_union`).
    pub hard_union: f32,
    /// Hard subtraction `max(-d1, d2)` (reference `op_subtract`).
    pub hard_subtract: f32,
    /// Hard intersection `max(d1, d2)` (reference `op_intersect`).
    pub hard_intersect: f32,
    /// Polynomial smooth union (reference `op_smooth_union`).
    pub smooth_union: f32,
    /// Polynomial smooth subtraction (reference `op_smooth_subtract`).
    pub smooth_subtract: f32,
    /// Polynomial smooth intersection (reference `op_smooth_intersect`).
    pub smooth_intersect: f32,
}

/// Encodes one [`SdfSmoothOpsQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &SdfSmoothOpsQuery) -> GpuQuery {
    GpuQuery {
        d1: q.d1,
        d2: q.d2,
        k: q.k,
        pad0: 0.0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`SdfSmoothOpsResult`].
fn decode_result(raw: &GpuResult) -> SdfSmoothOpsResult {
    SdfSmoothOpsResult {
        hard_union: raw.hard_union,
        hard_subtract: raw.hard_subtract,
        hard_intersect: raw.hard_intersect,
        smooth_union: raw.smooth_union,
        smooth_subtract: raw.smooth_subtract,
        smooth_intersect: raw.smooth_intersect,
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

/// A compiled, reusable `SDF` combinator compute pipeline, twinning the six
/// scalar combinators of the `CPU` golden
/// [`capsule_sdf`](prism_render_architecture::particle::capsule_sdf).
pub struct GpuSdfSmoothOps {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSdfSmoothOps {
    /// Compiles the `SDF` combinator kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSdfSmoothOps {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sdf_smooth_ops"),
            source: ShaderSource::Wgsl(SDF_SMOOTH_OPS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sdf_smooth_ops_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sdf_smooth_ops_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sdf_smooth_ops_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSdfSmoothOps {
            module,
            layout,
            pipeline,
        }
    }

    /// Folds every triple in `queries` and returns one [`SdfSmoothOpsResult`]
    /// per input, in order.
    ///
    /// All six outputs match the reference to within the tolerance documented on
    /// this module. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[SdfSmoothOpsQuery],
    ) -> Vec<SdfSmoothOpsResult> {
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
            label: Some("prism_volumetric_sdf_smooth_ops_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sdf_smooth_ops_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sdf_smooth_ops_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sdf_smooth_ops_bind_group"),
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
            label: Some("prism_volumetric_sdf_smooth_ops_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sdf_smooth_ops_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sdf_smooth_ops_pass"),
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
