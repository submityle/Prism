//! `wgpu` compute twin of the particle-subsystem `radix` `histogram` count
//! step
//! ([`histogram`](prism_render_architecture::particle::gpu_radix_histogram::histogram),
//! design §12 sort / §13 cull ordering).
//!
//! A least-significant-`digit` (`LSD`) `radix` sort begins each pass by
//! extracting a `bits`-wide `digit` from every key and tallying how many keys
//! land in each of the `1 << bits` buckets. That count — the single-pass
//! `histogram` — is the one piece of the sort that is self-contained in a
//! single `dispatch`: it needs no host-side prefix `scan`, no multi-pass
//! `scatter` orchestration and no inter-`workgroup` communication, so it ports
//! cleanly to one `atomicAdd`-per-key kernel. The `CPU` golden
//! [`histogram`](prism_render_architecture::particle::gpu_radix_histogram::histogram)
//! owns that math; [`GpuRadixHistogram`] is the on-device twin that runs one
//! thread per *key* and `atomicAdd`s each key into its bucket, so a passing
//! real-device parity test is direct evidence the ported kernel bins keys into
//! the same buckets the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Only the pure counting kernel is reproduced. The twin inlines the golden
//! [`extract_digit`](prism_render_architecture::particle::gpu_radix_histogram::extract_digit)
//! — a right shift by `pass * bits` followed by a low-`bits` mask, with a shift
//! that reaches bit `32` yielding `0` so the zero-padded high `digit`s of a
//! final pass match — and then `atomicAdd`s `1` into `hist[digit]`. The
//! downstream pieces of the sort are **not** twinned here: the exclusive prefix
//! `scan`
//! ([`bucket_offsets`](prism_render_architecture::particle::gpu_radix_histogram::bucket_offsets))
//! and the stable multi-pass `scatter`
//! ([`radix_sort_u32`](prism_render_architecture::particle::gpu_radix_histogram::radix_sort_u32))
//! need host-side multi-`dispatch` orchestration and remain `CPU`-side; the
//! `GpuRadixHistogram` deliberately covers only the one-`dispatch` count that is
//! self-consistent on its own.
//!
//! # `atomicAdd` versus `wrapping_add`
//!
//! The `CPU` golden accumulates with `wrapping_add` and the `WGSL` kernel with
//! `atomicAdd`, which also wraps on `u32` overflow — so the two agree exactly:
//! both wrap identically, and neither saturates. A bucket count only wraps when
//! a single `digit` collects more than `2^32` keys, far above any real-device
//! parity fixture, so the accumulators are indistinguishable at test scale.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer shifts and
//! masks, `+ - * /`, a comparison and `atomicAdd` — with no transcendental, no
//! `smoothstep`, no `u64` and no optional device feature, so it runs unmodified
//! on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! A `histogram` bucket count is a `u32` integer, so the parity test asserts
//! **exact per-bucket equality**, not a float tolerance — precisely the check
//! that catches an `extract_digit` ported one bit wrong (an off-by-one shift or
//! mask). There is no float math anywhere in the kernel, so there is no
//! `ULP`-boundary degenerate region to avoid: integer counting is bit-exact by
//! construction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_radix_histogram`；
//! 无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::gpu_radix_histogram::RadixConfig;
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

/// Core-`WGSL` `radix`-`histogram` kernel, inlined so this twin lives entirely
/// in the crate and binds no external `.wesl`. One thread per key inlines the
/// golden
/// [`extract_digit`](prism_render_architecture::particle::gpu_radix_histogram::extract_digit)
/// and `atomicAdd`s each key into its `digit`'s bucket, mirroring
/// [`histogram`](prism_render_architecture::particle::gpu_radix_histogram::histogram).
const GPU_RADIX_HISTOGRAM_WGSL: &str = r#"
// radix-histogram twin: one thread per key extracts the `bits`-wide digit at
// `pass_index` and atomicAdds the key into that bucket, mirroring the CPU golden
// `histogram` / `extract_digit` in
// `prism_render_architecture::particle::gpu_radix_histogram`.
//
// `extract_digit` is inlined bit-for-bit: a right shift by `pass_index * bits`
// then a low-`bits` mask, with a shift that reaches bit 32 yielding 0 (the
// zero-padded high digit of a final pass). Accumulation uses `atomicAdd` and the
// CPU uses `wrapping_add`; both wrap on u32 overflow, so they agree exactly.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gpu_radix_histogram;
// 无第三方引擎源码或衍生代码。

struct Params {
    count: u32,
    pass_index: u32,
    bits: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;

// Inlined `extract_digit`: shift right by `pass_index * bits`, mask off the low
// `bits` bits. A shift that reaches or passes bit 32 yields 0, matching the
// golden's zero-padded high digits on the final pass.
fn extract_digit(key: u32, pass_index: u32, bits: u32) -> u32 {
    let shift = pass_index * bits;
    if (shift >= 32u) {
        return 0u;
    }
    let mask = (1u << bits) - 1u;
    return (key >> shift) & mask;
}

@compute @workgroup_size(64)
fn gpu_radix_histogram_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let digit = extract_digit(keys[idx], params.pass_index, params.bits);
    atomicAdd(&hist[digit], 1u);
}
"#;

/// One `histogram` request: the batch of `u32` keys, the [`RadixConfig`] fixing
/// the per-pass `bits`, and which `digit` `pass` to count.
///
/// The returned `histogram` has length [`RadixConfig::bucket_count`] and entry
/// `d` counts the keys whose `pass` `digit` equals `d`, matching
/// [`histogram`](prism_render_architecture::particle::gpu_radix_histogram::histogram).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadixHistogramQuery {
    /// The `u32` keys to count, in any order (counting is commutative).
    pub keys: Vec<u32>,
    /// The `radix` configuration fixing the per-pass `digit` width (`bits`).
    pub config: RadixConfig,
    /// Which `digit` pass to count (`0` is the least-significant `digit`).
    pub pass: u32,
}

/// Uniform parameters for one dispatch. `16`-byte `repr(C)`: the key count, the
/// `digit` pass index, the per-pass `bits` width and a tail pad to the uniform
/// stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    count: u32,
    pass_index: u32,
    bits: u32,
    pad: u32,
}

/// A compiled, reusable `radix`-`histogram` pipeline.
pub struct GpuRadixHistogram {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRadixHistogram {
    /// Compiles the `radix`-`histogram` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRadixHistogram {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gpu_radix_histogram_shader"),
            source: ShaderSource::Wgsl(GPU_RADIX_HISTOGRAM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gpu_radix_histogram_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gpu_radix_histogram_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gpu_radix_histogram_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gpu_radix_histogram_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRadixHistogram {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the single-pass `histogram` on the device and reads it back.
    ///
    /// Returns a `Vec<u32>` of length [`RadixConfig::bucket_count`] whose entry
    /// `d` equals the count
    /// [`histogram`](prism_render_architecture::particle::gpu_radix_histogram::histogram)
    /// produces for bucket `d`, exactly (see the module-level correctness
    /// model). An empty key batch issues **no dispatch** — a storage buffer may
    /// not be zero-sized, and an empty input simply yields the all-zero
    /// `histogram` of the correct length — so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &RadixHistogramQuery) -> Vec<u32> {
        let bucket_count = query.config.bucket_count();

        // Empty input: no key to dispatch, and a storage buffer cannot be
        // zero-sized, so return the correctly sized zero histogram directly.
        // `bucket_count` is always at least `2` (bits clamp to `1..=8`), so the
        // output is never zero-length.
        if query.keys.is_empty() {
            return alloc_zeros(bucket_count);
        }

        let device = ctx.device();

        let gpu_params = GpuParams {
            count: query.keys.len() as u32,
            pass_index: query.pass,
            bits: query.config.bits,
            pad: 0,
        };

        let out_bytes = (bucket_count as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_radix_histogram_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let keys_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_radix_histogram_keys"),
            contents: bytemuck::cast_slice(&query.keys),
            usage: BufferUsages::STORAGE,
        });
        // Zero-initialized explicitly so every `atomicAdd` accumulates from `0`
        // regardless of the backend's buffer-clearing policy.
        let zeros = alloc_zeros(bucket_count);
        let hist_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gpu_radix_histogram_hist"),
            contents: bytemuck::cast_slice(&zeros),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let hist_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gpu_radix_histogram_hist_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gpu_radix_histogram_bind_group"),
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
            label: Some("prism_volumetric_gpu_radix_histogram_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gpu_radix_histogram_pass"),
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
        debug_assert_eq!(counts.len(), bucket_count);
        counts
    }
}

/// Allocates a zero-filled `histogram` of `len` buckets.
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
