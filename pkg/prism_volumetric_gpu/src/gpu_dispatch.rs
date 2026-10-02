//! `wgpu` compute twin of the deterministic particle §9 `GPU` dispatch contract
//! ([`gpu_dispatch`](prism_render_architecture::particle::gpu_dispatch)).
//!
//! The `CPU` golden standard owns the contract: the fixed pipeline order
//! ([`ParticleComputePass::order`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::order)),
//! the per-pass 1-D workgroup size
//! ([`ParticleComputePass::workgroup_size`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::workgroup_size)),
//! the direct/indirect classification
//! ([`ParticleComputePass::is_indirect`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::is_indirect)),
//! and the ceil-division from element count to workgroup count
//! ([`workgroup_count`](prism_render_architecture::particle::gpu_dispatch::workgroup_count)).
//! [`GpuDispatch`] is the on-device twin that runs one thread per
//! [`GpuDispatchQuery`] and reproduces all four lanes **bit-for-bit**, so a
//! passing real-device parity test is direct evidence the ported kernel derives
//! the identical integer contract, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries a `pass_code` (`0..9`, the golden enum order, equal to
//! [`ParticleComputePass::order`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::order))
//! and an `element_count` (the pass domain size). The kernel emits, per query:
//!
//! - `order`: the pass position, identical to `pass_code` by construction.
//! - `workgroup_size`: `256` for the `Sort` pass (`pass_code == 7`), `64` for
//!   every other pass, mirroring
//!   [`ParticleComputePass::workgroup_size`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::workgroup_size).
//! - `is_indirect`: `1` for the `GPU`-sized passes `Simulate`, `EventScatter`,
//!   `Bounds`, `Cull`, `Sort` (`pass_code` in `{2, 3, 5, 6, 7}`) and `0` for the
//!   `CPU`-sized passes, mirroring
//!   [`ParticleComputePass::is_indirect`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::is_indirect).
//! - `workgroup_count`: the ceil-division `ceil(element_count / workgroup_size)`
//!   mirroring
//!   [`workgroup_count`](prism_render_architecture::particle::gpu_dispatch::workgroup_count),
//!   with the same degenerate guards (an empty domain yields `0`, and a zero
//!   workgroup size yields `0` rather than dividing by zero).
//!
//! # Portability
//!
//! `WGSL` has no `u32::div_ceil`, so the kernel open-codes the ceiling as the
//! guarded `if (ws == 0u) { 0u } else { (ec + ws - 1u) / ws }`. The `ec + ws - 1`
//! addition cannot wrap because the pass-derived `ws` is small (`64` or `256`)
//! and the fixture keeps `element_count < 2^31`; the golden's `div_ceil` avoids
//! the addition entirely, but on the agreed input domain the two forms coincide.
//! The kernel uses only the portable core-`WGSL` subset — unsigned `+ - * /`,
//! comparison and `||` — with no transcendental call and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Every lane is pure unsigned integer arithmetic with no reordering, so `CPU`
//! and `GPU` are **bit-exact**; the parity test asserts a precise `==` with no
//! tolerance on all four classification/count lanes.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! `prism_render_architecture::particle::gpu_dispatch` plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

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

/// Number of threads per workgroup for the twin's own 1-D dispatch. `64` is the
/// portable, warp-friendly default shared by the other twins in this crate.
///
/// This is the dispatch granularity of the *parity harness itself*; it is
/// unrelated to the per-pass [`GpuDispatchResult::workgroup_size`] the kernel
/// computes.
const WORKGROUP_SIZE: u32 = 64;

/// One per-element dispatch query: a particle compute pass (as its golden enum
/// order `pass_code`, `0..9`) and the `element_count` of that pass's domain.
///
/// `pass_code` matches
/// [`ParticleComputePass::order`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::order);
/// `8`-byte `std430` stride (two `u32`, no padding) matching `Query` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuDispatchQuery {
    /// The pass identity as its golden enum order, `0..9`.
    pub pass_code: u32,
    /// The number of domain elements this pass dispatches over.
    pub element_count: u32,
}

/// One per-element dispatch result: the derived pass `order`, `workgroup_size`,
/// `is_indirect` flag (`0`/`1`) and `workgroup_count`.
///
/// `16`-byte `std430` stride (four `u32`, no padding) matching `Result` in the
/// shader. Every field is an integer or classification code, so the twin and
/// golden compare with an exact `==`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuDispatchResult {
    /// The pass's position in the fixed §9 order, `0..9`.
    pub order: u32,
    /// The pass's 1-D `@workgroup_size` (`256` for `Sort`, else `64`).
    pub workgroup_size: u32,
    /// Whether the pass dispatches indirectly from a `GPU`-computed count
    /// (`1`) or directly from a `CPU`-known count (`0`).
    pub is_indirect: u32,
    /// Workgroups needed to cover `element_count` at `workgroup_size`:
    /// `ceil(element_count / workgroup_size)`, `0` on an empty or degenerate
    /// input.
    pub workgroup_count: u32,
}

/// The portable core-`WGSL` dispatch-contract kernel, embedded inline so the
/// twin ships as a single source file. The entry point `dispatch_main` mirrors
/// the `CPU` golden
/// [`ParticleComputePass`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass)
/// methods and
/// [`workgroup_count`](prism_render_architecture::particle::gpu_dispatch::workgroup_count)
/// lane for lane; see the module documentation for the contract.
const GPU_DISPATCH_WGSL: &str = r#"
// Particle §9 dispatch-contract twin: one thread per query derives the pass
// order, 1-D workgroup size, direct/indirect flag and ceil-division workgroup
// count from a pass code (0..9, the golden enum order) and an element count.
// All math is pure unsigned integer arithmetic, so CPU and GPU are bit-exact.
// WGSL has no u32::div_ceil, so the ceiling is open-coded as the guarded
// `if (ws == 0u) { 0u } else { (ec + ws - 1u) / ws }`; the `ec + ws - 1`
// addition cannot wrap on the agreed domain (ws is 64 or 256, element_count
// below 2^31). The kernel uses only the portable core-WGSL subset and takes no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: twinned from this repository's
// prism_render_architecture::particle::gpu_dispatch; no third-party engine
// source or derived code.

struct Params {
    // Number of valid queries; threads at or beyond this index return.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    pass_code: u32,
    element_count: u32,
}

struct Result {
    order: u32,
    workgroup_size: u32,
    is_indirect: u32,
    workgroup_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Pass codes in canonical §9 order, matching `ParticleComputePass::order`.
const PASS_SORT: u32 = 7u;
const WORKGROUP_SIZE_1D: u32 = 64u;
const WORKGROUP_SIZE_SORT: u32 = 256u;

// The 1-D @workgroup_size for a pass code: wider for Sort, 64 for every other
// pass. Mirrors the golden `ParticleComputePass::workgroup_size`.
fn pass_workgroup_size(code: u32) -> u32 {
    if (code == PASS_SORT) {
        return WORKGROUP_SIZE_SORT;
    }
    return WORKGROUP_SIZE_1D;
}

// Whether a pass dispatches indirectly from a GPU-computed count. True for
// Simulate (2), EventScatter (3), Bounds (5), Cull (6) and Sort (7); false for
// the CPU-sized passes. Mirrors the golden `ParticleComputePass::is_indirect`.
fn pass_is_indirect(code: u32) -> u32 {
    if (code == 2u || code == 3u || code == 5u || code == 6u || code == 7u) {
        return 1u;
    }
    return 0u;
}

// Ceil-division from element count to workgroup count, open-coded because WGSL
// has no u32::div_ceil. An empty domain needs 0 groups and a degenerate zero
// workgroup size returns 0 rather than dividing by zero, matching the golden
// `workgroup_count` guards.
fn workgroup_count(ec: u32, ws: u32) -> u32 {
    if (ws == 0u) {
        return 0u;
    }
    return (ec + ws - 1u) / ws;
}

@compute @workgroup_size(64)
fn dispatch_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    let ws = pass_workgroup_size(q.pass_code);
    var r: Result;
    // The pass code already is the §9 order, so order == pass_code.
    r.order = q.pass_code;
    r.workgroup_size = ws;
    r.is_indirect = pass_is_indirect(q.pass_code);
    r.workgroup_count = workgroup_count(q.element_count, ws);
    results[idx] = r;
}
"#;

/// Uniform parameters for one dispatch: the element count plus three pad words
/// so the struct is a `16`-byte, `16`-byte-aligned uniform. Layout matches
/// `Params` in [`GPU_DISPATCH_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable particle dispatch-contract pipeline.
pub struct GpuDispatch {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDispatch {
    /// Compiles the dispatch-contract kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDispatch {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_dispatch"),
            source: ShaderSource::Wgsl(GPU_DISPATCH_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_dispatch_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_dispatch_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_dispatch_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("dispatch_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDispatch {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every query in `queries`, returning one [`GpuDispatchResult`]
    /// per query in input order.
    ///
    /// The returned result for query `q` mirrors
    /// [`ParticleComputePass::order`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::order),
    /// [`ParticleComputePass::workgroup_size`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::workgroup_size),
    /// [`ParticleComputePass::is_indirect`](prism_render_architecture::particle::gpu_dispatch::ParticleComputePass::is_indirect)
    /// and
    /// [`workgroup_count`](prism_render_architecture::particle::gpu_dispatch::workgroup_count)
    /// evaluated on the pass identified by `q.pass_code` and `q.element_count`.
    /// An empty `queries` slice yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return before any dispatch.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GpuDispatchQuery]) -> Vec<GpuDispatchResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();
        let out_bytes = (count * size_of::<GpuDispatchResult>()) as u64;

        let gpu_params = Params {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_dispatch_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_dispatch_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_dispatch_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_dispatch_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_dispatch_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gpu_dispatch_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_dispatch_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, GpuDispatchResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(out.len(), count);
        out
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
