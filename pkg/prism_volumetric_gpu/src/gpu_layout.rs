//! `wgpu` compute twin of the device-free `std430` byte-layout primitives
//! ([`gpu_layout`](prism_render_architecture::particle::gpu_layout), particle
//! design §5, §9).
//!
//! The `CPU` golden
//! [`gpu_layout`](prism_render_architecture::particle::gpu_layout) owns the
//! smallest shared pieces every per-pass `GPU` buffer contract reuses: the
//! `std430` stride constants
//! ([`U32_STRIDE`](prism_render_architecture::particle::gpu_layout::U32_STRIDE),
//! [`VEC2_STRIDE`](prism_render_architecture::particle::gpu_layout::VEC2_STRIDE)
//! and
//! [`VEC4_STRIDE`](prism_render_architecture::particle::gpu_layout::VEC4_STRIDE)),
//! the clamp-to-one byte-size rule
//! ([`storage_bytes`](prism_render_architecture::particle::gpu_layout::storage_bytes)),
//! and the access enum
//! ([`ParticleBufferAccess`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess))
//! with its writability predicate
//! ([`ParticleBufferAccess::is_writable`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess::is_writable)).
//! [`GpuLayout`] is the on-device twin: one thread resolves one
//! [`GpuLayoutQuery`] into one [`GpuLayoutResult`], so a passing real-device
//! parity test is direct evidence the ported kernel computes the same byte size
//! and the same writability flag the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Each query carries a `stride`, an element `count` and an `access_code` that
//! encodes a [`ParticleBufferAccess`] variant (`0` for `Read`, `1` for
//! `ReadWrite`, in the golden enum's declaration order). The kernel reproduces
//! `storage_bytes` as `stride * max(count, 1)` — the same clamp-to-one-element
//! rule a non-empty `WebGPU` storage binding needs — and reproduces
//! `is_writable` as the discrete `1`/`0` flag the golden `match` yields.
//!
//! # Correctness model
//!
//! Every value is a `u32`, a classification code or a `bool`-flavoured flag:
//! the byte size is pure unsigned multiply-and-max with no rounding, and the
//! writability flag is a discrete classification. `CPU` and `GPU` therefore
//! compute identical bit patterns, and the parity test asserts an exact `==` on
//! every field with no tolerance. Fixtures keep `stride * count` well below
//! `2^31`, so the device `u32` multiply never wraps where the golden
//! `saturating_mul` would otherwise clamp.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `*`, the
//! `max` built-in, an unsigned comparison and index arithmetic. There is no
//! `sqrt`, no divide, no transcendental call, no `u64` and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There is no
//! loop: each thread performs a fixed, bounded sequence of integer work, so the
//! kernel provably terminates.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_layout`；无第三方引擎源码或衍生代码。
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

/// The `std430` byte-layout kernel, mirroring the `CPU` golden
/// [`gpu_layout`](prism_render_architecture::particle::gpu_layout) field for
/// field. The single entry point `solve` resolves one query per thread,
/// embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_layout`。
const GPU_LAYOUT_WGSL: &str = r#"
// gpu_layout twin: one thread per query reproduces the CPU golden
// `particle::gpu_layout`. `storage_bytes` is `stride * max(count, 1u)` — the
// clamp-to-one-element rule a non-empty WebGPU storage binding needs — and
// `is_writable` is the discrete 1u/0u flag the golden `match` yields (1u for
// the ReadWrite variant, access_code 1u). Pure u32 arithmetic: one multiply,
// one `max`, one unsigned compare. There is no sqrt, no divide, no
// transcendental call and no u64, so the kernel runs unmodified on Metal,
// Vulkan and DX12. There is no loop, so the kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gpu_layout；无第三方
// 引擎源码或衍生代码。

// Access code for the `ReadWrite` variant, matching the golden enum's
// declaration order (`Read` == 0u, `ReadWrite` == 1u). It is the only writable
// access.
const ACCESS_READ_WRITE: u32 = 1u;

// Dispatch parameters. 16-byte uniform block: the valid query count plus three
// pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 12-byte std430 stride of three scalar words, matching the host
// `GpuLayoutQuery`: the element stride, the element count and the access code.
struct LayoutQuery {
    stride: u32,
    count: u32,
    access_code: u32,
}

// One result. 8-byte std430 stride of two scalar words, matching the host
// `GpuLayoutResult`: the clamped total byte size and the writability flag.
struct LayoutResult {
    storage_bytes: u32,
    is_writable: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<LayoutQuery>;
@group(0) @binding(2) var<storage, read_write> results: array<LayoutResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    // storage_bytes: a non-empty storage binding reserves at least one element,
    // so the count is clamped up to one before the multiply.
    let effective_count = max(q.count, 1u);
    var out: LayoutResult;
    out.storage_bytes = q.stride * effective_count;

    // is_writable: only the ReadWrite variant may be written, mirroring the
    // golden `matches!(self, Self::ReadWrite)`.
    var writable: u32 = 0u;
    if (q.access_code == ACCESS_READ_WRITE) {
        writable = 1u;
    }
    out.is_writable = writable;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`GPU_LAYOUT_WGSL`]: the valid query count plus three pad words —
/// `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query for the `std430` byte-layout twin: an element `stride`, an element
/// `count` and an `access_code`.
///
/// The `access_code` encodes a
/// [`ParticleBufferAccess`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess)
/// variant in the golden enum's declaration order: `0` for `Read` and `1` for
/// `ReadWrite`. The `repr(C)` layout — three `u32` words, `12` bytes with no
/// padding — matches the `WGSL` `LayoutQuery` struct exactly, so it is uploaded
/// to the device without a separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_layout`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuLayoutQuery {
    /// Byte stride of a single storage element.
    pub stride: u32,
    /// Number of storage elements in the buffer.
    pub count: u32,
    /// Access-mode code: `0` for `Read`, `1` for `ReadWrite`.
    pub access_code: u32,
}

/// One resolved answer for a single [`GpuLayoutQuery`], mirroring the golden
/// `storage_bytes` and `is_writable` outputs.
///
/// The `repr(C)` layout — two `u32` words, `8` bytes — matches the `WGSL`
/// `LayoutResult` struct exactly, so device results are read back without a
/// separate decode step. `is_writable` is the discrete `1`/`0` flag the golden
/// [`ParticleBufferAccess::is_writable`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess::is_writable)
/// yields.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_layout`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuLayoutResult {
    /// Total byte size `stride * max(count, 1)`, matching the golden
    /// [`storage_bytes`](prism_render_architecture::particle::gpu_layout::storage_bytes).
    pub storage_bytes: u32,
    /// Writability flag, `1` for a writable buffer and `0` otherwise.
    pub is_writable: u32,
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

/// A compiled, reusable `std430` byte-layout compute pipeline, twinning the
/// `CPU` golden
/// [`gpu_layout`](prism_render_architecture::particle::gpu_layout).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_layout`。
pub struct GpuLayout {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuLayout {
    /// Compiles the byte-layout kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_layout`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuLayout {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_layout"),
            source: ShaderSource::Wgsl(GPU_LAYOUT_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_layout_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_layout_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_layout_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuLayout {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one [`GpuLayoutResult`] per
    /// input, in order.
    ///
    /// Each result equals the matching golden pair exactly — `storage_bytes`
    /// mirrors
    /// [`storage_bytes`](prism_render_architecture::particle::gpu_layout::storage_bytes)
    /// and `is_writable` mirrors
    /// [`ParticleBufferAccess::is_writable`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess::is_writable)
    /// — because the whole path is integer bit algebra. An empty `queries`
    /// batch returns an empty vector with no dispatch issued, since a storage
    /// buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_layout`。
    #[must_use]
    pub fn evaluate(&self, ctx: &GpuContext, queries: &[GpuLayoutQuery]) -> Vec<GpuLayoutResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_layout_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_layout_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<GpuLayoutResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_layout_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_layout_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_layout_bind_group"),
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
            label: Some("prism_volumetric_gpu_layout_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_layout_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            let groups = (queries.len() as u32).div_ceil(WORKGROUP_SIZE);
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuLayoutResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
    }
}
