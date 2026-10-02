//! `wgpu` compute twin of the effect-instancing localization and capacity
//! arithmetic
//! ([`instancing`](prism_render_architecture::particle::instancing),
//! particle design §26, §30).
//!
//! The `CPU` golden
//! [`instancing`](prism_render_architecture::particle::instancing) owns the
//! small, verifiable scalar arithmetic an effect-instance placement needs: the
//! hand-rolled three-component vector
//! ([`Vec3::length`](prism_render_architecture::particle::instancing::Vec3::length),
//! [`Vec3::scaled`](prism_render_architecture::particle::instancing::Vec3::scaled),
//! [`Vec3::add`](prism_render_architecture::particle::instancing::Vec3::add) and
//! [`Vec3::approx_eq`](prism_render_architecture::particle::instancing::Vec3::approx_eq)),
//! the uniform-scale radius localization
//! ([`InstanceTransform::local_to_world_scale`](prism_render_architecture::particle::instancing::InstanceTransform::local_to_world_scale)),
//! the scale-then-translate point localization
//! ([`InstanceTransform::apply`](prism_render_architecture::particle::instancing::InstanceTransform::apply)),
//! and the saturating per-template particle budget
//! ([`EffectTemplate::template_particle_capacity`](prism_render_architecture::particle::instancing::EffectTemplate::template_particle_capacity)
//! and
//! [`instance_particle_upper_bound`](prism_render_architecture::particle::instancing::instance_particle_upper_bound)).
//! [`GpuInstancing`] is the on-device twin: one thread answers one
//! [`InstancingQuery`], so a passing real-device parity test is direct evidence
//! the ported kernel computes the same lengths, scaled and summed vectors,
//! localized radii and positions, equality verdicts and saturating capacities
//! the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! A single batched kernel evaluates a tagged union of independent queries,
//! one per thread, mirroring the reference tap for tap:
//!
//! - [`InstancingQuery::Vec3Length`] reproduces
//!   [`Vec3::length`](prism_render_architecture::particle::instancing::Vec3::length):
//!   `sqrt(x*x + y*y + z*z)`, the only radical the contract uses.
//! - [`InstancingQuery::Vec3Scaled`] reproduces
//!   [`Vec3::scaled`](prism_render_architecture::particle::instancing::Vec3::scaled):
//!   a component-wise multiply by the uniform factor.
//! - [`InstancingQuery::Vec3Add`] reproduces
//!   [`Vec3::add`](prism_render_architecture::particle::instancing::Vec3::add):
//!   a component-wise sum.
//! - [`InstancingQuery::Vec3ApproxEq`] reproduces
//!   [`Vec3::approx_eq`](prism_render_architecture::particle::instancing::Vec3::approx_eq):
//!   every component within the reference `CMP_EPS` tolerance, reported as a
//!   `bool`.
//! - [`InstancingQuery::LocalToWorldScale`] reproduces
//!   [`InstanceTransform::local_to_world_scale`](prism_render_architecture::particle::instancing::InstanceTransform::local_to_world_scale):
//!   `local_radius * uniform_scale`.
//! - [`InstancingQuery::Apply`] reproduces
//!   [`InstanceTransform::apply`](prism_render_architecture::particle::instancing::InstanceTransform::apply):
//!   `local * uniform_scale + translation` component-wise, so a zero uniform
//!   scale collapses to the pure translation with no special case.
//! - [`InstancingQuery::TemplateParticleCapacity`] and
//!   [`InstancingQuery::InstanceParticleUpperBound`] reproduce
//!   [`EffectTemplate::template_particle_capacity`](prism_render_architecture::particle::instancing::EffectTemplate::template_particle_capacity)
//!   and
//!   [`instance_particle_upper_bound`](prism_render_architecture::particle::instancing::instance_particle_upper_bound):
//!   the `u32` saturating product `emitter_count * particle_capacity_per_emitter`,
//!   reported as `u32::MAX` on overflow.
//!
//! # What stays on the host (not twinned)
//!
//! `WGSL` has no `u64`, so the wide-accumulator helpers must stay on the host:
//! the `u64` saturating sum of
//! [`total_particle_upper_bound`](prism_render_architecture::particle::instancing::total_particle_upper_bound)
//! and the `u64` saturating byte size of
//! [`override_buffer_bytes`](prism_render_architecture::particle::instancing::override_buffer_bytes).
//! The pooling state machine
//! [`InstancePool`](prism_render_architecture::particle::instancing::InstancePool)
//! (spawn / despawn / counts) and the sort-and-dedup
//! [`distinct_template_count`](prism_render_architecture::particle::instancing::distinct_template_count)
//! are host-side control flow and allocation, not per-element arithmetic, so
//! they also remain on the host.
//!
//! # Correctness model
//!
//! The continuous answers (length, scaled, add, localized radius and position)
//! thread through multiplies, adds and one `sqrt`, so `CPU` and `GPU` are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on every
//! continuous quantity. The discrete answers (the `approx_eq` verdict and the
//! saturating capacities) are integer / `bool` and are compared exactly.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::instancing`；无第三方引擎源码或衍生代码。
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

/// Operation tag selecting [`InstancingQuery::Vec3Length`] on device.
const OP_VEC3_LENGTH: u32 = 0;
/// Operation tag selecting [`InstancingQuery::Vec3Scaled`] on device.
const OP_VEC3_SCALED: u32 = 1;
/// Operation tag selecting [`InstancingQuery::Vec3Add`] on device.
const OP_VEC3_ADD: u32 = 2;
/// Operation tag selecting [`InstancingQuery::Vec3ApproxEq`] on device.
const OP_VEC3_APPROX_EQ: u32 = 3;
/// Operation tag selecting [`InstancingQuery::LocalToWorldScale`] on device.
const OP_LOCAL_TO_WORLD_SCALE: u32 = 4;
/// Operation tag selecting [`InstancingQuery::Apply`] on device.
const OP_APPLY: u32 = 5;
/// Operation tag selecting [`InstancingQuery::TemplateParticleCapacity`] on
/// device.
const OP_TEMPLATE_CAPACITY: u32 = 6;
/// Operation tag selecting [`InstancingQuery::InstanceParticleUpperBound`] on
/// device.
const OP_INSTANCE_UPPER_BOUND: u32 = 7;

/// The portable core-`WGSL` instancing kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`instancing`](prism_render_architecture::particle::instancing) function for
/// function; see the module documentation for the algorithm.
const INSTANCING_WGSL: &str = r#"
// Effect-instancing twin: one thread per query reproduces the hand-rolled Vec3
// algebra (`length`, `scaled`, `add`, `approx_eq`), the uniform-scale radius and
// point localization (`InstanceTransform::local_to_world_scale` and
// `InstanceTransform::apply`) and the u32 saturating per-template particle
// budget (`EffectTemplate::template_particle_capacity` and
// `instance_particle_upper_bound`). It mirrors the CPU golden
// `particle::instancing` function for function, uses only the portable
// core-WGSL subset (abs/sqrt and + - * / plus unsigned index math), needs no
// transcendental call and takes no optional feature, so it runs unmodified on
// Metal, Vulkan and DX12. The kernel has no loop, so it provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::instancing；无第三方
// 引擎源码或衍生代码。

// Per-component magnitude below which two Vec3 lanes are treated as equal,
// matching the reference `CMP_EPS`; the compare rule used instead of an f32 `==`.
const CMP_EPS: f32 = 1.0e-6;

// Operation tags, matching the host op codes.
const OP_VEC3_LENGTH: u32 = 0u;
const OP_VEC3_SCALED: u32 = 1u;
const OP_VEC3_ADD: u32 = 2u;
const OP_VEC3_APPROX_EQ: u32 = 3u;
const OP_LOCAL_TO_WORLD_SCALE: u32 = 4u;
const OP_APPLY: u32 = 5u;
const OP_TEMPLATE_CAPACITY: u32 = 6u;
const OP_INSTANCE_UPPER_BOUND: u32 = 7u;

struct Params {
    // Number of queries in the storage arrays; threads past this short-circuit.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Operation tag selecting which twinned function this thread evaluates.
    op: u32,
    // Template emitter count for the capacity ops.
    emitter_count: u32,
    // Per-emitter particle capacity for the capacity ops.
    capacity_per_emitter: u32,
    pad0: u32,
    // Primary vector: `length`/`scaled` input, `add`/`approx_eq` left operand,
    // or `apply` local point. A pad lane follows to keep the vec3 16-byte
    // aligned.
    vec_a: vec3<f32>,
    // Scalar: `scaled` factor, `apply` uniform scale, or
    // `local_to_world_scale` uniform scale.
    scalar_a: f32,
    // Secondary vector: `add`/`approx_eq` right operand or `apply` translation.
    vec_b: vec3<f32>,
    // Scalar: `local_to_world_scale` local radius.
    scalar_b: f32,
}

struct Result {
    // Operation tag echoed back so the host decodes the right union variant.
    op: u32,
    // Boolean verdict (0 / 1) for `approx_eq`.
    bool_flag: u32,
    // Unsigned result for the saturating capacity ops.
    uint_val: u32,
    pad0: u32,
    // Vector result for `scaled`, `add` and `apply`. A pad lane follows.
    vec: vec3<f32>,
    // Scalar result for `length` and `local_to_world_scale`.
    scalar: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Whether every component of `a` equals `b`'s within `CMP_EPS`, matching the
// reference `Vec3::approx_eq`.
fn vec3_approx_eq(a: vec3<f32>, b: vec3<f32>) -> bool {
    return abs(a.x - b.x) < CMP_EPS
        && abs(a.y - b.y) < CMP_EPS
        && abs(a.z - b.z) < CMP_EPS;
}

// Saturating u32 multiply, mirroring the reference `u32::saturating_mul`:
// returns `u32::MAX` on overflow rather than the wrapped product WGSL would
// otherwise produce. A zero operand short-circuits to zero so the overflow
// probe never divides by zero.
fn sat_mul_u32(a: u32, b: u32) -> u32 {
    if (a == 0u || b == 0u) {
        return 0u;
    }
    let prod = a * b;
    if (prod / a != b) {
        return 0xffffffffu;
    }
    return prod;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let op = q.op;

    var out: Result;
    out.op = op;
    out.bool_flag = 0u;
    out.uint_val = 0u;
    out.pad0 = 0u;
    out.vec = vec3<f32>(0.0, 0.0, 0.0);
    out.scalar = 0.0;

    if (op == OP_VEC3_LENGTH) {
        // length = sqrt(x*x + y*y + z*z).
        out.scalar = sqrt(dot(q.vec_a, q.vec_a));
    } else if (op == OP_VEC3_SCALED) {
        // scaled = component-wise multiply by the uniform factor.
        out.vec = q.vec_a * q.scalar_a;
    } else if (op == OP_VEC3_ADD) {
        // add = component-wise sum.
        out.vec = q.vec_a + q.vec_b;
    } else if (op == OP_VEC3_APPROX_EQ) {
        if (vec3_approx_eq(q.vec_a, q.vec_b)) {
            out.bool_flag = 1u;
        }
    } else if (op == OP_LOCAL_TO_WORLD_SCALE) {
        // local_radius * uniform_scale (scalar_b * scalar_a).
        out.scalar = q.scalar_b * q.scalar_a;
    } else if (op == OP_APPLY) {
        // local.scaled(uniform_scale).add(translation); a zero scale collapses
        // to the pure translation with no special case.
        out.vec = q.vec_a * q.scalar_a + q.vec_b;
    } else {
        // TemplateParticleCapacity / InstanceParticleUpperBound: both are the
        // u32 saturating product emitter_count * capacity_per_emitter.
        out.uint_val = sat_mul_u32(q.emitter_count, q.capacity_per_emitter);
    }

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`INSTANCING_WGSL`].
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
/// Each `vec3` lane carries a trailing pad word so every member stays `16`-byte
/// aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Operation tag selecting the twinned function.
    op: u32,
    /// Template emitter count for the capacity ops.
    emitter_count: u32,
    /// Per-emitter particle capacity for the capacity ops.
    capacity_per_emitter: u32,
    /// Padding word.
    pad0: u32,
    /// Primary vector operand.
    vec_a: [f32; 3],
    /// Scalar operand (`scaled` factor or uniform scale).
    scalar_a: f32,
    /// Secondary vector operand.
    vec_b: [f32; 3],
    /// Scalar operand (`local_to_world_scale` local radius).
    scalar_b: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Operation tag echoed back for decoding.
    op: u32,
    /// Boolean verdict (`0` / `1`) for the `approx_eq` op.
    bool_flag: u32,
    /// Unsigned result for the saturating capacity ops.
    uint_val: u32,
    /// Padding word.
    pad0: u32,
    /// Vector result for `scaled`, `add` and `apply`.
    vec: [f32; 3],
    /// Scalar result for `length` and `local_to_world_scale`.
    scalar: f32,
}

/// One query for the instancing twin: a tagged union selecting which of the
/// twinned reference functions this element evaluates.
///
/// The vector variants carry raw `[f32; 3]` lane triples (the reference
/// [`Vec3`](prism_render_architecture::particle::instancing::Vec3) flattened);
/// the transform variants carry the uniform scale, translation and local
/// quantity; the capacity variants carry the raw `EffectTemplate` fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InstancingQuery {
    /// Euclidean length, twinning
    /// [`Vec3::length`](prism_render_architecture::particle::instancing::Vec3::length).
    Vec3Length {
        /// Vector whose length is taken.
        vector: [f32; 3],
    },
    /// Uniform scale, twinning
    /// [`Vec3::scaled`](prism_render_architecture::particle::instancing::Vec3::scaled).
    Vec3Scaled {
        /// Vector to scale.
        vector: [f32; 3],
        /// Uniform scale factor.
        scale: f32,
    },
    /// Component-wise sum, twinning
    /// [`Vec3::add`](prism_render_architecture::particle::instancing::Vec3::add).
    Vec3Add {
        /// Left operand.
        lhs: [f32; 3],
        /// Right operand.
        rhs: [f32; 3],
    },
    /// Tolerant equality, twinning
    /// [`Vec3::approx_eq`](prism_render_architecture::particle::instancing::Vec3::approx_eq).
    Vec3ApproxEq {
        /// Left operand.
        lhs: [f32; 3],
        /// Right operand.
        rhs: [f32; 3],
    },
    /// Uniform-scale radius localization, twinning
    /// [`InstanceTransform::local_to_world_scale`](prism_render_architecture::particle::instancing::InstanceTransform::local_to_world_scale).
    LocalToWorldScale {
        /// Instance uniform scale.
        uniform_scale: f32,
        /// Template-local radius to localize.
        local_radius: f32,
    },
    /// Scale-then-translate point localization, twinning
    /// [`InstanceTransform::apply`](prism_render_architecture::particle::instancing::InstanceTransform::apply).
    Apply {
        /// Instance world-space translation.
        translation: [f32; 3],
        /// Instance uniform scale.
        uniform_scale: f32,
        /// Template-local point to localize.
        local: [f32; 3],
    },
    /// Saturating per-template particle budget, twinning
    /// [`EffectTemplate::template_particle_capacity`](prism_render_architecture::particle::instancing::EffectTemplate::template_particle_capacity).
    TemplateParticleCapacity {
        /// Number of emitters the template runs.
        emitter_count: u32,
        /// Fixed particle-pool capacity of each emitter.
        particle_capacity_per_emitter: u32,
    },
    /// Saturating per-instance particle upper bound, twinning
    /// [`instance_particle_upper_bound`](prism_render_architecture::particle::instancing::instance_particle_upper_bound).
    InstanceParticleUpperBound {
        /// Number of emitters the template runs.
        emitter_count: u32,
        /// Fixed particle-pool capacity of each emitter.
        particle_capacity_per_emitter: u32,
    },
}

/// One resolved answer for a single [`InstancingQuery`], mirroring the value
/// the reference reports for the corresponding function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InstancingResult {
    /// Euclidean length, matching
    /// [`Vec3::length`](prism_render_architecture::particle::instancing::Vec3::length).
    Vec3Length {
        /// Vector length.
        length: f32,
    },
    /// Scaled vector, matching
    /// [`Vec3::scaled`](prism_render_architecture::particle::instancing::Vec3::scaled).
    Vec3Scaled {
        /// Scaled vector.
        vector: [f32; 3],
    },
    /// Summed vector, matching
    /// [`Vec3::add`](prism_render_architecture::particle::instancing::Vec3::add).
    Vec3Add {
        /// Summed vector.
        vector: [f32; 3],
    },
    /// Equality verdict, matching
    /// [`Vec3::approx_eq`](prism_render_architecture::particle::instancing::Vec3::approx_eq).
    Vec3ApproxEq {
        /// Whether the two vectors are equal within tolerance.
        equal: bool,
    },
    /// Localized radius, matching
    /// [`InstanceTransform::local_to_world_scale`](prism_render_architecture::particle::instancing::InstanceTransform::local_to_world_scale).
    LocalToWorldScale {
        /// World-space radius.
        scale: f32,
    },
    /// Localized point, matching
    /// [`InstanceTransform::apply`](prism_render_architecture::particle::instancing::InstanceTransform::apply).
    Apply {
        /// World-space point.
        world: [f32; 3],
    },
    /// Saturating template capacity, matching
    /// [`EffectTemplate::template_particle_capacity`](prism_render_architecture::particle::instancing::EffectTemplate::template_particle_capacity).
    TemplateParticleCapacity {
        /// Total particle capacity of the template.
        capacity: u32,
    },
    /// Saturating per-instance upper bound, matching
    /// [`instance_particle_upper_bound`](prism_render_architecture::particle::instancing::instance_particle_upper_bound).
    InstanceParticleUpperBound {
        /// Per-instance particle upper bound.
        upper_bound: u32,
    },
}

/// Encodes one [`InstancingQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &InstancingQuery) -> GpuQuery {
    let mut slot = GpuQuery::zeroed();
    match q {
        InstancingQuery::Vec3Length { vector } => {
            slot.op = OP_VEC3_LENGTH;
            slot.vec_a = *vector;
        }
        InstancingQuery::Vec3Scaled { vector, scale } => {
            slot.op = OP_VEC3_SCALED;
            slot.vec_a = *vector;
            slot.scalar_a = *scale;
        }
        InstancingQuery::Vec3Add { lhs, rhs } => {
            slot.op = OP_VEC3_ADD;
            slot.vec_a = *lhs;
            slot.vec_b = *rhs;
        }
        InstancingQuery::Vec3ApproxEq { lhs, rhs } => {
            slot.op = OP_VEC3_APPROX_EQ;
            slot.vec_a = *lhs;
            slot.vec_b = *rhs;
        }
        InstancingQuery::LocalToWorldScale {
            uniform_scale,
            local_radius,
        } => {
            slot.op = OP_LOCAL_TO_WORLD_SCALE;
            slot.scalar_a = *uniform_scale;
            slot.scalar_b = *local_radius;
        }
        InstancingQuery::Apply {
            translation,
            uniform_scale,
            local,
        } => {
            slot.op = OP_APPLY;
            slot.vec_a = *local;
            slot.scalar_a = *uniform_scale;
            slot.vec_b = *translation;
        }
        InstancingQuery::TemplateParticleCapacity {
            emitter_count,
            particle_capacity_per_emitter,
        } => {
            slot.op = OP_TEMPLATE_CAPACITY;
            slot.emitter_count = *emitter_count;
            slot.capacity_per_emitter = *particle_capacity_per_emitter;
        }
        InstancingQuery::InstanceParticleUpperBound {
            emitter_count,
            particle_capacity_per_emitter,
        } => {
            slot.op = OP_INSTANCE_UPPER_BOUND;
            slot.emitter_count = *emitter_count;
            slot.capacity_per_emitter = *particle_capacity_per_emitter;
        }
    }
    slot
}

/// Decodes one packed [`GpuResult`] into the public [`InstancingResult`],
/// selecting the union variant from the echoed operation tag.
fn decode_result(raw: &GpuResult) -> InstancingResult {
    match raw.op {
        OP_VEC3_LENGTH => InstancingResult::Vec3Length { length: raw.scalar },
        OP_VEC3_SCALED => InstancingResult::Vec3Scaled { vector: raw.vec },
        OP_VEC3_ADD => InstancingResult::Vec3Add { vector: raw.vec },
        OP_VEC3_APPROX_EQ => InstancingResult::Vec3ApproxEq {
            equal: raw.bool_flag != 0,
        },
        OP_LOCAL_TO_WORLD_SCALE => InstancingResult::LocalToWorldScale { scale: raw.scalar },
        OP_APPLY => InstancingResult::Apply { world: raw.vec },
        OP_TEMPLATE_CAPACITY => InstancingResult::TemplateParticleCapacity {
            capacity: raw.uint_val,
        },
        _ => InstancingResult::InstanceParticleUpperBound {
            upper_bound: raw.uint_val,
        },
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

/// A compiled, reusable instancing compute pipeline, twinning the `CPU` golden
/// [`instancing`](prism_render_architecture::particle::instancing).
pub struct GpuInstancing {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuInstancing {
    /// Compiles the instancing kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuInstancing {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_instancing"),
            source: ShaderSource::Wgsl(INSTANCING_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_instancing_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_instancing_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_instancing_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuInstancing {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries` and returns one [`InstancingResult`]
    /// per input, in order.
    ///
    /// The continuous answers match the reference to within the tolerance
    /// documented on this module; the discrete answers match exactly. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[InstancingQuery]) -> Vec<InstancingResult> {
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
            label: Some("prism_volumetric_instancing_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_instancing_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_instancing_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_instancing_bind_group"),
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
            label: Some("prism_volumetric_instancing_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_instancing_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_instancing_pass"),
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
