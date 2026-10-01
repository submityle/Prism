//! `wgpu` compute twin of the single-pass `u16` counting-sort `histogram`
//! builder
//! ([`histogram`](prism_render_architecture::particle::counting_sort_u16::histogram),
//! design §12 sort ordering).
//!
//! A production `GPU` VFX stack quantizes view-depth or draw-order keys into a
//! 16-bit domain and bucket-sorts millions of particles per frame; when the
//! domain fits in 16 bits a *single* counting pass sorts the whole key space.
//! The first and only parallel step of that pass is building a `65536`-bucket
//! `histogram` of the keys. The `CPU` golden
//! [`histogram`](prism_render_architecture::particle::counting_sort_u16::histogram)
//! owns that counting; [`GpuCountingSortU16`] is the on-device twin that runs
//! one thread per *key* and `atomicAdd`s each key into its bucket, so a passing
//! real-device parity test is direct evidence the ported kernel counts keys
//! into the same buckets the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! Only the pure counting kernel is ported: one `atomic<u32>` bucket per `u16`
//! value, [`BUCKET_COUNT`](prism_render_architecture::particle::counting_sort_u16::BUCKET_COUNT)
//! `= 1 << 16 = 65536` buckets, incremented once per input key. This is a
//! self-contained single dispatch — the exclusive prefix sum and the scatter
//! that complete the golden's full `sort` are serial, order-dependent passes
//! outside this twin's scope; the parity harness compares the histogram against
//! the golden's own
//! [`exclusive_prefix_sum`](prism_render_architecture::particle::counting_sort_u16::exclusive_prefix_sum)
//! only on the `CPU` side.
//!
//! Keys reach the device as zero-extended `u32` words (one key per element),
//! because the portable core-`WGSL` subset has no `u16` storage type. The
//! kernel masks each word to its low 16 bits (`& 0xffffu`) before indexing, so
//! the bucket index is always in `[0, 65536)` and the lookup is in bounds.
//!
//! # `atomicAdd` versus `usize` addition
//!
//! The `CPU` golden accumulates in `usize`; the `WGSL` kernel accumulates with
//! `atomicAdd`, which wraps on `u32` overflow. This is observable only when a
//! single bucket would exceed `2^32` elements sharing one key, far above any
//! real-device parity fixture, so the two accumulators are equivalent at test
//! scale. The parity test keeps every bucket count well under `u32::MAX`.
//!
//! # No `u64`
//!
//! The twinned `histogram` and `exclusive_prefix_sum` are `usize`/`u32`
//! arithmetic only. The `u64` that appears in the golden lives exclusively in
//! its internal test `RNG` (`next_u64` / seeding) and is **not** part of any
//! ported function; neither this module nor the `WGSL` kernel contains any
//! `u64`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — an integer mask (`&`)
//! and `atomicAdd` — with no transcendental call, no `sqrt`, no `smoothstep`,
//! no optional device feature and no `u64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! A histogram bucket count is an integer, so the parity test asserts **exact
//! per-bucket equality**, not a float tolerance: any mismatch is a genuine port
//! bug (a dropped increment, a wrong mask, a miscounted key). There is no
//! floating point anywhere on the path and therefore no rounding, no `ULP`
//! boundary and no degenerate region to avoid.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::counting_sort_u16`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::counting_sort_u16::BUCKET_COUNT;
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
/// that divides evenly across `Metal`, `Vulkan` and `DX12`.
const WORKGROUP_SIZE: u32 = 64;

/// Core-`WGSL` counting kernel, inlined so this twin lives entirely in the
/// crate with no external `.wesl`. One thread per key masks the key to its low
/// 16 bits and `atomicAdd`s it into the `65536`-bucket histogram, mirroring the
/// `CPU` golden `counting_sort_u16::histogram`.
const COUNTING_SORT_U16_WGSL: &str = r#"
// Counting-sort u16 histogram twin: one thread per key. Each key is a
// zero-extended u16 in a u32 word; masking to the low 16 bits yields a bucket
// index in [0, 65536), and `atomicAdd` counts it. Mirrors the CPU golden
// `particle::counting_sort_u16::histogram`. No u64, no transcendental, no sqrt.

// Bucket count: one atomic<u32> per distinct u16 key (1u << 16).
const BUCKET_COUNT: u32 = 65536u;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;

@compute @workgroup_size(64)
fn counting_sort_u16_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // Low 16 bits give the bucket, always within [0, BUCKET_COUNT).
    let bucket = keys[idx] & (BUCKET_COUNT - 1u);
    atomicAdd(&hist[bucket], 1u);
}
"#;

/// One histogram request: the batch of `u16` keys to count.
///
/// The returned histogram has length
/// [`BUCKET_COUNT`](prism_render_architecture::particle::counting_sort_u16::BUCKET_COUNT)
/// and each bucket counts the keys equal to it, matching
/// [`histogram`](prism_render_architecture::particle::counting_sort_u16::histogram).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountingSortU16Query {
    /// The `u16` keys to count, in any order (counting is commutative).
    pub keys: Vec<u16>,
}

/// Uniform parameters for one dispatch. `16`-byte `repr(C)` matching `Params`
/// in the inlined shader: the key count plus padding to the uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable `u16` counting-sort `histogram` pipeline.
pub struct GpuCountingSortU16 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCountingSortU16 {
    /// Compiles the `u16` counting-sort `histogram` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCountingSortU16 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_shader"),
            source: ShaderSource::Wgsl(COUNTING_SORT_U16_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("counting_sort_u16_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCountingSortU16 {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds the `65536`-bucket key `histogram` for `query` on-device.
    ///
    /// Returns a `Vec<u32>` of length
    /// [`BUCKET_COUNT`](prism_render_architecture::particle::counting_sort_u16::BUCKET_COUNT)
    /// whose entry `k` equals the count
    /// [`histogram`](prism_render_architecture::particle::counting_sort_u16::histogram)
    /// produces for bucket `k`, exactly (see the module-level correctness
    /// model). An empty key batch issues **no dispatch** — a storage buffer may
    /// not be zero-sized, and an empty input simply yields the all-zero
    /// histogram of the correct length — so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &CountingSortU16Query) -> Vec<u32> {
        // Empty input: no key to dispatch, and a storage buffer cannot be
        // zero-sized, so return the correctly sized zero histogram directly.
        if query.keys.is_empty() {
            return alloc_zeros(BUCKET_COUNT);
        }

        let device = ctx.device();

        let gpu_params = GpuParams {
            count: query.keys.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        // Keys zero-extended to u32 words: the portable core-`WGSL` subset has
        // no u16 storage type, so each key travels in its own 32-bit element.
        let keys_u32: Vec<u32> = query.keys.iter().map(|&key| u32::from(key)).collect();

        let out_bytes = (BUCKET_COUNT as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let keys_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_keys"),
            contents: bytemuck::cast_slice(&keys_u32),
            usage: BufferUsages::STORAGE,
        });
        // Zero-initialized explicitly so every `atomicAdd` accumulates from `0`
        // regardless of the backend's buffer-clearing policy.
        let zeros = alloc_zeros(BUCKET_COUNT);
        let hist_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_hist"),
            contents: bytemuck::cast_slice(&zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let hist_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_hist_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: keys_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: hist_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_counting_sort_u16_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_counting_sort_u16_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per key, flattened to a 1-D dispatch.
            let groups = (query.keys.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&hist_buf, 0, &hist_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        hist_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = hist_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let counts = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        hist_stage.unmap();
        debug_assert_eq!(counts.len(), BUCKET_COUNT);
        counts
    }
}

/// Allocates a zero-filled histogram of `len` buckets.
fn alloc_zeros(len: usize) -> Vec<u32> {
    vec![0u32; len]
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
