//! `wgpu` compute twin of the contact-material combination closed form, from
//! the `CPU` golden `prism_physics_core::collider::material`'s
//! `PhysicsMaterial::new` + `PhysicsMaterial::combine`.
//!
//! Two contacting surfaces each carry a friction coefficient and a restitution
//! coefficient. The reference first clamps each surface's raw inputs
//! (`PhysicsMaterial::new`: friction to non-negative, restitution to
//! `0.0..=1.0`), then combines the pair (`PhysicsMaterial::combine`): friction
//! by the geometric mean `sqrt(fa * fb)` and restitution by the maximum. This
//! module ports that single stateless closed form onto the device: one thread
//! resolves one query, so a passing real-device parity test is direct evidence
//! the ported kernel computes the same effective material the reference does,
//! not merely that the shader compiles.
//!
//! # What is twinned
//!
//! For each query the kernel reproduces `new` then `combine` on two raw surface
//! parameter pairs, in exactly the golden operator order:
//!
//! * `fa = max(friction_a, 0)`, `ra = clamp(restitution_a, 0, 1)`.
//! * `fb = max(friction_b, 0)`, `rb = clamp(restitution_b, 0, 1)`.
//! * `out_friction = sqrt(max(fa * fb, 0))`.
//! * `out_restitution = max(ra, rb)`.
//!
//! # Correctness model
//!
//! The continuous arithmetic (two clamps per side, a multiply, a `max`-with-0
//! and a `sqrt`) threads through operators a `GPU` may contract, so `CPU` and
//! `GPU` are not necessarily bit-exact; the `friction` and `restitution`
//! scalars are compared with an `abs <= 1e-4 || rel <= 1e-3` tolerance
//! (`REL_FLOOR = 1e-6`). The discrete `valid` flag is compared exactly.
//!
//! # Degenerate inputs
//!
//! A pair is valid only when all four raw inputs are finite; a non-finite input
//! yields `valid = 0` with both outputs `0`. To stop a non-finite input from
//! poisoning the arithmetic through `NaN` propagation, each raw input is first
//! routed through a `select` that substitutes `0` when it is non-finite, so the
//! `sqrt` base `max(fa * fb, 0)` stays non-negative and no spurious `NaN`
//! survives. An empty query batch short-circuits on the host with no dispatch,
//! since a storage buffer cannot be zero-sized.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `min`, `max`,
//! `clamp`, `sqrt`, `select`, `+ - *` and unsigned index arithmetic — with no
//! `sin`, `cos`, `tan`, `exp`, `log`, `pow`, no `round` and no `f32` remainder,
//! so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Finiteness is tested
//! with the ordered compare `abs(x) < 3.0e38` (which rejects both infinities and
//! `NaN`) rather than a bare `x == x`; there is no `f32` equality anywhere.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::material`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` material-combination kernel, embedded inline so the
/// twin ships as a single source file. The single entry point `solve` mirrors
/// the `CPU` golden `PhysicsMaterial::new` + `PhysicsMaterial::combine`; see the
/// module documentation for the closed form.
const MATERIAL_COMBINE_WGSL: &str = r#"
// Material-combination twin: one thread per query reproduces new + combine. It
// uses only the portable core-WGSL subset (abs, min, max, clamp, sqrt, select,
// + - * plus unsigned index math), takes no optional feature, and has no loop
// and no branch, so it provably terminates. Finiteness is an ordered
// abs < 3.0e38 compare (rejecting infinities and NaN) fed to select; no bare
// f32 equality anywhere.

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Raw friction coefficient of the first surface.
    friction_a: f32,
    // Raw restitution coefficient of the first surface.
    restitution_a: f32,
    // Raw friction coefficient of the second surface.
    friction_b: f32,
    // Raw restitution coefficient of the second surface.
    restitution_b: f32,
}

struct Result {
    // Combined friction sqrt(max(fa*fb, 0)) when valid, else 0.
    friction: f32,
    // Combined restitution max(ra, rb) when valid, else 0.
    restitution: f32,
    // 1 when all four raw inputs are finite, else 0.
    valid: u32,
    // Padding word to a 16-byte stride.
    pad0: u32,
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
    let fa_finite = abs(q.friction_a) < FINITE_LIMIT;
    let ra_finite = abs(q.restitution_a) < FINITE_LIMIT;
    let fb_finite = abs(q.friction_b) < FINITE_LIMIT;
    let rb_finite = abs(q.restitution_b) < FINITE_LIMIT;
    let ok = fa_finite && ra_finite && fb_finite && rb_finite;

    // Substitute 0 for any non-finite input so NaN/inf cannot poison the
    // arithmetic that feeds the (discarded) invalid branch.
    let raw_fa = select(0.0, q.friction_a, fa_finite);
    let raw_ra = select(0.0, q.restitution_a, ra_finite);
    let raw_fb = select(0.0, q.friction_b, fb_finite);
    let raw_rb = select(0.0, q.restitution_b, rb_finite);

    // Golden PhysicsMaterial::new clamps per side.
    let fa = max(raw_fa, 0.0);
    let ra = clamp(raw_ra, 0.0, 1.0);
    let fb = max(raw_fb, 0.0);
    let rb = clamp(raw_rb, 0.0, 1.0);

    // Golden PhysicsMaterial::combine: geometric-mean friction, max restitution.
    let friction = sqrt(max(fa * fb, 0.0));
    let restitution = max(ra, rb);

    var out: Result;
    out.friction = select(0.0, friction, ok);
    out.restitution = select(0.0, restitution, ok);
    out.valid = select(0u, 1u, ok);
    out.pad0 = 0u;
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
/// the four raw surface scalars — `4` `f32` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    friction_a: f32,
    restitution_a: f32,
    friction_b: f32,
    restitution_b: f32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct: the two combined scalars, the validity flag and a padding word —
/// `4` words (`16` bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    friction: f32,
    restitution: f32,
    valid: u32,
    pad0: u32,
}

/// One material-combination query: the raw friction and restitution of two
/// contacting surfaces, before clamping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialCombineQuery {
    /// Raw friction coefficient of the first surface.
    pub friction_a: f32,
    /// Raw restitution coefficient of the first surface.
    pub restitution_a: f32,
    /// Raw friction coefficient of the second surface.
    pub friction_b: f32,
    /// Raw restitution coefficient of the second surface.
    pub restitution_b: f32,
}

impl MaterialCombineQuery {
    /// Builds a query from the two surfaces' raw friction and restitution.
    #[must_use]
    pub fn new(
        friction_a: f32,
        restitution_a: f32,
        friction_b: f32,
        restitution_b: f32,
    ) -> MaterialCombineQuery {
        MaterialCombineQuery {
            friction_a,
            restitution_a,
            friction_b,
            restitution_b,
        }
    }
}

/// One resolved answer for a single query, mirroring the reference
/// `PhysicsMaterial::new` + `PhysicsMaterial::combine` output for that pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialCombineResult {
    /// The combined friction `sqrt(max(fa * fb, 0))` when valid, else `0`.
    pub friction: f32,
    /// The combined restitution `max(ra, rb)` when valid, else `0`.
    pub restitution: f32,
    /// `1` when all four raw inputs are finite, else `0`.
    pub valid: u32,
}

/// Encodes one [`MaterialCombineQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &MaterialCombineQuery) -> GpuQuery {
    GpuQuery {
        friction_a: q.friction_a,
        restitution_a: q.restitution_a,
        friction_b: q.friction_b,
        restitution_b: q.restitution_b,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`MaterialCombineResult`].
fn decode_result(raw: &GpuResult) -> MaterialCombineResult {
    MaterialCombineResult {
        friction: raw.friction,
        restitution: raw.restitution,
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

/// A compiled, reusable material-combination compute pipeline, twinning the
/// `CPU` golden `PhysicsMaterial::new` + `PhysicsMaterial::combine`.
pub struct GpuMaterialCombine {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuMaterialCombine {
    /// Compiles the material-combination kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMaterialCombine {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_material_combine"),
            source: ShaderSource::Wgsl(MATERIAL_COMBINE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_material_combine_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_material_combine_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_material_combine_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuMaterialCombine {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query in `queries` and returns one [`MaterialCombineResult`]
    /// per input, in order.
    ///
    /// The `valid` flag matches the reference exactly and the `friction` and
    /// `restitution` scalars to the module's tolerance. An empty `queries` batch
    /// returns an empty vector with no dispatch issued, since a storage buffer
    /// cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[MaterialCombineQuery],
    ) -> Vec<MaterialCombineResult> {
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
            label: Some("prism_volumetric_material_combine_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_material_combine_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_material_combine_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_material_combine_bind_group"),
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
            label: Some("prism_volumetric_material_combine_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_material_combine_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_material_combine_pass"),
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
