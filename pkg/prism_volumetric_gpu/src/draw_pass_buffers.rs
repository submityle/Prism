//! `wgpu` compute twin of the device-free `std430` draw-pass bind-group
//! contracts
//! ([`draw_pass_buffers`](prism_render_architecture::particle::draw_pass_buffers),
//! particle design §9: `FillDrawArgs` and `RenderDraw`).
//!
//! The `CPU` golden
//! [`draw_pass_buffers`](prism_render_architecture::particle::draw_pass_buffers)
//! owns two storage-buffer enums — one per draw-preparation pass — and reports,
//! for each buffer, its `@binding` index, element stride, access mode,
//! output flag, element count and clamped total byte size against a
//! [`ParticleDrawExtent`](prism_render_architecture::particle::draw_pass_buffers::ParticleDrawExtent).
//! [`GpuDrawPassBuffers`] is the on-device twin: one thread resolves one
//! [`GpuDrawPassBufferQuery`] — a `(which-enum, variant)` pair plus an extent —
//! into one [`GpuDrawPassBufferResult`] carrying all six answers, so a passing
//! real-device parity test is direct evidence the ported kernel reproduces the
//! same `ABI` the reference publishes, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each query carries a `which_enum` selector (`0` for
//! [`FillDrawArgsBuffer`](prism_render_architecture::particle::draw_pass_buffers::FillDrawArgsBuffer),
//! `1` for
//! [`RenderDrawBuffer`](prism_render_architecture::particle::draw_pass_buffers::RenderDrawBuffer)),
//! a `variant_code` (the index into that enum's `ALL` array, which is also the
//! `@binding` index), and the pool `capacity` / `live_count` of the extent. The
//! kernel reproduces the golden `match` tables for `binding`, `stride`,
//! `access` (as the `0`/`1` code
//! [`ParticleBufferAccess`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess)
//! uses), `is_output`, `element_count` and `byte_size` (the clamp-to-one
//! `stride * max(count, 1)` rule). The single-element draw-args record and the
//! `live_count` scalar are closed-form constants; only the sorted-index and
//! per-instance arrays scale with `capacity`.
//!
//! # Correctness model
//!
//! Every value is a `u32`, a classification code or a `bool`-flavoured flag:
//! the byte size is pure unsigned multiply-and-max with no rounding, and the
//! binding, stride, access code and output flag are discrete classifications.
//! `CPU` and `GPU` therefore compute identical bit patterns, and the parity
//! test asserts an exact `==` on every field with no tolerance. Fixtures keep
//! `stride * capacity` well below `2^31`, so the device `u32` multiply never
//! wraps where the golden `saturating_mul` would otherwise clamp.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset: unsigned `*`, the
//! `max` built-in, unsigned comparisons and index arithmetic. There is no
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
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::draw_pass_buffers`；无第三方引擎源码或衍生代码。
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

/// The draw-pass bind-group kernel, mirroring the `CPU` golden
/// [`draw_pass_buffers`](prism_render_architecture::particle::draw_pass_buffers)
/// field for field. The single entry point `solve` resolves one query per
/// thread, embedded inline so the twin ships as a single source file.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::draw_pass_buffers`。
const DRAW_PASS_BUFFERS_WGSL: &str = r#"
// draw_pass_buffers twin: one thread per query reproduces the CPU golden
// `particle::draw_pass_buffers`. A query is a (which_enum, variant_code) pair
// plus the pool extent; the kernel reproduces the golden `match` tables for
// binding, stride, access, is_output, element_count and byte_size. `byte_size`
// is `stride * max(element_count, 1u)` — the clamp-to-one-element rule a
// non-empty WebGPU storage binding needs — and `is_output` is the discrete
// 1u/0u flag the golden `is_writable` yields (1u for the ReadWrite variant,
// access_code 1u). Pure u32 arithmetic: comparisons, one multiply and one
// `max`. There is no sqrt, no divide, no transcendental call and no u64, so the
// kernel runs unmodified on Metal, Vulkan and DX12. There is no loop, so the
// kernel provably terminates.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::draw_pass_buffers；无第三方
// 引擎源码或衍生代码。

// which_enum selector: FillDrawArgsBuffer versus RenderDrawBuffer.
const WHICH_FILL_DRAW_ARGS: u32 = 0u;
const WHICH_RENDER_DRAW: u32 = 1u;

// std430 strides mirrored from the golden `gpu_layout` and `draw_pass_buffers`:
// a scalar u32, the five-u32 DrawIndexedIndirectArgs record and the 64-byte
// per-instance particle descriptor.
const U32_STRIDE: u32 = 4u;
const DRAW_INDEXED_INDIRECT_STRIDE: u32 = 20u;
const INSTANCE_STRIDE: u32 = 64u;

// Access codes in the golden enum's declaration order: Read then ReadWrite.
const ACCESS_READ: u32 = 0u;
const ACCESS_READ_WRITE: u32 = 1u;

// Dispatch parameters. 16-byte uniform block: the valid query count plus three
// pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 16-byte std430 stride of four scalar words, matching the host
// `GpuDrawPassBufferQuery`: the which-enum selector, the variant code, and the
// pool capacity and live_count of the extent.
struct DrawPassQuery {
    which_enum: u32,
    variant_code: u32,
    capacity: u32,
    live_count: u32,
}

// One result. 24-byte std430 stride of six scalar words, matching the host
// `GpuDrawPassBufferResult`: binding, stride, access code, output flag,
// element count and clamped total byte size.
struct DrawPassResult {
    binding: u32,
    stride: u32,
    access_code: u32,
    is_output: u32,
    element_count: u32,
    byte_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<DrawPassQuery>;
@group(0) @binding(2) var<storage, read_write> results: array<DrawPassResult>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    var stride: u32 = U32_STRIDE;
    var access_code: u32 = ACCESS_READ;
    var element_count: u32 = 1u;

    if (q.which_enum == WHICH_FILL_DRAW_ARGS) {
        // FillDrawArgsBuffer::ALL = [LiveCount(0), SortedIndices(1), DrawArgs(2)].
        if (q.variant_code == 2u) {
            // DrawArgs: single indirect record, written in place.
            stride = DRAW_INDEXED_INDIRECT_STRIDE;
            access_code = ACCESS_READ_WRITE;
            element_count = 1u;
        } else {
            // LiveCount and SortedIndices are both read-only u32 arrays.
            stride = U32_STRIDE;
            access_code = ACCESS_READ;
            if (q.variant_code == 1u) {
                // SortedIndices spans the whole pool capacity.
                element_count = q.capacity;
            } else {
                // LiveCount is the single survivor scalar.
                element_count = 1u;
            }
        }
    } else {
        // RenderDrawBuffer::ALL = [Instances(0), SortedIndices(1), DrawArgs(2)];
        // every buffer this raster pass binds is read-only.
        access_code = ACCESS_READ;
        if (q.variant_code == 0u) {
            // Instances: per-instance descriptor array across the pool capacity.
            stride = INSTANCE_STRIDE;
            element_count = q.capacity;
        } else {
            if (q.variant_code == 1u) {
                // SortedIndices: the same u32 index list across the capacity.
                stride = U32_STRIDE;
                element_count = q.capacity;
            } else {
                // DrawArgs: the single indirect record, aliased read-only.
                stride = DRAW_INDEXED_INDIRECT_STRIDE;
                element_count = 1u;
            }
        }
    }

    var out: DrawPassResult;
    // binding is the index into the enum's ALL array, i.e. the variant code.
    out.binding = q.variant_code;
    out.stride = stride;
    out.access_code = access_code;

    // is_output: only the ReadWrite variant is written, mirroring the golden
    // `access().is_writable()`.
    var writable: u32 = 0u;
    if (access_code == ACCESS_READ_WRITE) {
        writable = 1u;
    }
    out.is_output = writable;

    out.element_count = element_count;

    // byte_size: a non-empty storage binding reserves at least one element, so
    // the count is clamped up to one before the multiply.
    let effective_count = max(element_count, 1u);
    out.byte_size = stride * effective_count;

    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`DRAW_PASS_BUFFERS_WGSL`]: the valid query count plus three pad
/// words — `16` bytes with no interior padding.
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

/// One query for the draw-pass bind-group twin: a `which_enum` selector, a
/// `variant_code`, and the pool `capacity` / `live_count` of the extent.
///
/// `which_enum` is `0` for the
/// [`FillDrawArgsBuffer`](prism_render_architecture::particle::draw_pass_buffers::FillDrawArgsBuffer)
/// pass and `1` for the
/// [`RenderDrawBuffer`](prism_render_architecture::particle::draw_pass_buffers::RenderDrawBuffer)
/// pass. `variant_code` is the index into that enum's `ALL` array (which is
/// also the `@binding` index): `0`/`1`/`2`. `capacity` and `live_count` mirror
/// the fields of
/// [`ParticleDrawExtent`](prism_render_architecture::particle::draw_pass_buffers::ParticleDrawExtent).
/// The `repr(C)` layout — four `u32` words, `16` bytes with no padding —
/// matches the `WGSL` `DrawPassQuery` struct exactly, so it is uploaded to the
/// device without a separate encode step.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::draw_pass_buffers`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuDrawPassBufferQuery {
    /// Which draw pass's enum: `0` for `FillDrawArgs`, `1` for `RenderDraw`.
    pub which_enum: u32,
    /// Index into the selected enum's `ALL` array; also the `@binding` index.
    pub variant_code: u32,
    /// Particle pool capacity — the worst-case element count of the
    /// sorted-index and per-instance arrays.
    pub capacity: u32,
    /// Post-cull/sort survivor count carried in the extent; a non-domain scalar
    /// that never changes any buffer's size.
    pub live_count: u32,
}

/// One resolved answer for a single [`GpuDrawPassBufferQuery`], mirroring the
/// golden `binding`, `stride`, `access`, `is_output`, `element_count` and
/// `byte_size` outputs.
///
/// The `repr(C)` layout — six `u32` words, `24` bytes — matches the `WGSL`
/// `DrawPassResult` struct exactly, so device results are read back without a
/// separate decode step. `access_code` encodes a
/// [`ParticleBufferAccess`](prism_render_architecture::particle::gpu_layout::ParticleBufferAccess)
/// variant (`0` for `Read`, `1` for `ReadWrite`) and `is_output` is the
/// discrete `1`/`0` writability flag the golden `is_output` yields.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::draw_pass_buffers`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuDrawPassBufferResult {
    /// The `@group(0)` binding index of the buffer.
    pub binding: u32,
    /// Byte stride of one element, matching the `WESL` scalar / record / struct
    /// layout.
    pub stride: u32,
    /// Access-mode code: `0` for `Read`, `1` for `ReadWrite`.
    pub access_code: u32,
    /// Output flag, `1` when the pass writes this buffer and `0` otherwise.
    pub is_output: u32,
    /// Element count for the pool: `1` for the single-element records and
    /// `capacity` for the per-pool arrays.
    pub element_count: u32,
    /// Total byte size `stride * max(element_count, 1)`, matching the golden
    /// clamp-to-one `byte_size`.
    pub byte_size: u32,
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

/// A compiled, reusable draw-pass bind-group compute pipeline, twinning the
/// `CPU` golden
/// [`draw_pass_buffers`](prism_render_architecture::particle::draw_pass_buffers).
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::draw_pass_buffers`。
pub struct GpuDrawPassBuffers {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDrawPassBuffers {
    /// Compiles the draw-pass bind-group kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::draw_pass_buffers`。
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDrawPassBuffers {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_draw_pass_buffers_module"),
            source: ShaderSource::Wgsl(DRAW_PASS_BUFFERS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_draw_pass_buffers_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_draw_pass_buffers_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_draw_pass_buffers_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDrawPassBuffers {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves every query in `queries` and returns one
    /// [`GpuDrawPassBufferResult`] per input, in order.
    ///
    /// Each result equals the matching golden tuple exactly — `binding`,
    /// `stride`, `access`, `is_output`, `element_count` and `byte_size` all
    /// mirror the `CPU`
    /// [`draw_pass_buffers`](prism_render_architecture::particle::draw_pass_buffers)
    /// reference — because the whole path is integer bit algebra. An empty
    /// `queries` batch returns an empty vector with no dispatch issued, since a
    /// storage buffer cannot be zero-sized.
    ///
    /// Provenance: 孪生自本仓 `prism_render_architecture::particle::draw_pass_buffers`。
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[GpuDrawPassBufferQuery],
    ) -> Vec<GpuDrawPassBufferResult> {
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
            label: Some("prism_volumetric_draw_pass_buffers_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_draw_pass_buffers_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (queries.len() as u64) * (size_of::<GpuDrawPassBufferResult>() as u64);
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_draw_pass_buffers_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_draw_pass_buffers_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_draw_pass_buffers_bind_group"),
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
            label: Some("prism_volumetric_draw_pass_buffers_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_draw_pass_buffers_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuDrawPassBufferResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());

        gpu_results
    }
}
