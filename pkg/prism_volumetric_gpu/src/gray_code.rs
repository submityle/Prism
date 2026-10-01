//! `wgpu` compute twin of the reflected-binary `Gray`-code transcoder
//! ([`gray_code`](prism_render_architecture::particle::gray_code), design §27).
//!
//! A `Gray` code (reflected binary code) renumbers a counter so that
//! *consecutive* values differ in exactly one bit. The `CPU` golden
//! [`gray_code`](prism_render_architecture::particle::gray_code) owns the closed
//! form; [`GpuGrayCode`] is the on-device twin that runs one thread per element
//! and reproduces the same `u32`-domain transforms element for element. A
//! passing real-device parity test is therefore direct evidence the ported
//! kernels fold, unfold, step and diff the bits exactly as the reference does,
//! not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Only the `u32` domain is ported — `WGSL` has no `u64`, so the `u64` variants
//! of the reference are deliberately skipped. The five twinned transforms are:
//!
//! * **Encode** ([`binary_to_gray`](GpuGrayCode::binary_to_gray)): the single
//!   operation `n ^ (n >> 1)`, matching
//!   [`binary_to_gray_u32`](prism_render_architecture::particle::gray_code::binary_to_gray_u32).
//! * **Decode** ([`gray_to_binary`](GpuGrayCode::gray_to_binary)): the xor
//!   prefix scan with doubling shifts (`b ^= b >> 1; b ^= b >> 2; b ^= b >> 4;
//!   b ^= b >> 8; b ^= b >> 16;`), step for step the same five rounds as
//!   [`gray_to_binary_u32`](prism_render_architecture::particle::gray_code::gray_to_binary_u32).
//! * **Step** ([`next_gray`](GpuGrayCode::next_gray) /
//!   [`prev_gray`](GpuGrayCode::prev_gray)): decode, take a wrapping `+/- 1`
//!   step and re-encode, mirroring
//!   [`next_gray_u32`](prism_render_architecture::particle::gray_code::next_gray_u32)
//!   and
//!   [`prev_gray_u32`](prism_render_architecture::particle::gray_code::prev_gray_u32).
//!   `WGSL` unsigned arithmetic wraps modulo `2^32`, matching the reference
//!   `wrapping_add` / `wrapping_sub`.
//! * **Diff** ([`gray_diff_bit_index`](GpuGrayCode::gray_diff_bit_index)):
//!   reports the single toggled bit index when two codes are adjacent, mirroring
//!   [`gray_diff_bit_index`](prism_render_architecture::particle::gray_code::gray_diff_bit_index).
//!   The reference returns `Option<u32>`; the kernel encodes `None` as the
//!   sentinel `0xFFFF_FFFF` (unreachable as a real bit index, which is at most
//!   `31`) and the host maps the sentinel back to `None` before an exact
//!   comparison.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — the bit operators
//! `^ >> << & | ~`, the arithmetic `+ - *`, and `min`/`max` — with no
//! transcendental call, no `countOneBits`/`firstTrailingBit` intrinsic, no
//! optional device feature and no `u64`. The population count and
//! trailing-zero scan the diff kernel needs are hand-rolled from shifts, masks
//! and adds (a `SWAR` popcount and a branching binary search), so the kernels
//! run unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Every transform here is pure integer bit algebra, so `CPU` and `GPU` compute
//! identical bit patterns with no rounding anywhere on the path. The parity
//! test therefore asserts an exact `==` on every element (and on the recovered
//! `Option<u32>` for the diff), with no tolerance: any mismatch is a genuine
//! port bug (a wrong shift count, a dropped fold round, a miscounted bit).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gray_code`；无第三方引擎源码或衍生代码。

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
/// that divides evenly on every target backend.
const WORKGROUP_SIZE: u32 = 64;

/// Sentinel the diff kernel writes for a `None` result (non-adjacent or equal
/// codes). A genuine bit index is at most `31`, so `0xFFFF_FFFF` is unambiguous
/// and the host maps it back to [`None`].
const NONE_SENTINEL: u32 = 0xFFFF_FFFF;

/// The `u32`-domain `Gray`-code kernels, mirroring the `CPU` golden
/// [`gray_code`](prism_render_architecture::particle::gray_code) tap for tap. A
/// single source file hosts five entry points sharing one bind-group layout.
const GRAY_CODE_WGSL: &str = r#"
// Gray-code transcoder twin: one thread per element. Five entry points mirror
// the CPU golden `particle::gray_code` u32 domain: encode `n ^ (n >> 1)`, the
// five-round xor prefix-scan decode, the wrapping +/-1 next/prev steps, and the
// adjacency diff. They use only the portable core-WGSL subset (bit operators,
// + - *), with a hand-rolled SWAR popcount and binary-search trailing-zero scan
// instead of any intrinsic, no transcendental and no u64, so they run unmodified
// on Metal, Vulkan and DX12. WGSL unsigned arithmetic wraps modulo 2^32, which
// matches the reference `wrapping_add` / `wrapping_sub`.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::gray_code；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of valid elements; threads past this short-circuit.
    count: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> in_a: array<u32>;
@group(0) @binding(2) var<storage, read> in_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

// None sentinel: a real toggled-bit index is at most 31, so 0xFFFFFFFF is free.
const NONE_SENTINEL: u32 = 0xFFFFFFFFu;

// Encode: copy the top bit and xor every lower bit with its higher neighbour.
fn encode(n: u32) -> u32 {
    return n ^ (n >> 1u);
}

// Decode: the exact inverse xor prefix scan, the same five doubling rounds the
// reference `gray_to_binary_u32` uses (1, 2, 4, 8, 16).
fn decode(gray: u32) -> u32 {
    var b = gray;
    b = b ^ (b >> 1u);
    b = b ^ (b >> 2u);
    b = b ^ (b >> 4u);
    b = b ^ (b >> 8u);
    b = b ^ (b >> 16u);
    return b;
}

// Hand-rolled population count (SWAR), standing in for `u32::count_ones`: no
// `countOneBits` intrinsic, only masks, shifts, adds and one multiply.
fn popcount(x: u32) -> u32 {
    var v = x;
    v = v - ((v >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    return (v * 0x01010101u) >> 24u;
}

// Hand-rolled trailing-zero count (binary search), standing in for
// `u32::trailing_zeros`: no `firstTrailingBit` intrinsic. Called only on a
// value with exactly one set bit, but the search is correct for any non-zero x.
fn trailing_zeros(x: u32) -> u32 {
    var n = 0u;
    var v = x;
    if ((v & 0x0000FFFFu) == 0u) { n = n + 16u; v = v >> 16u; }
    if ((v & 0x000000FFu) == 0u) { n = n + 8u; v = v >> 8u; }
    if ((v & 0x0000000Fu) == 0u) { n = n + 4u; v = v >> 4u; }
    if ((v & 0x00000003u) == 0u) { n = n + 2u; v = v >> 2u; }
    if ((v & 0x00000001u) == 0u) { n = n + 1u; }
    return n;
}

@compute @workgroup_size(64)
fn binary_to_gray(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    dst[idx] = encode(in_a[idx]);
}

@compute @workgroup_size(64)
fn gray_to_binary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    dst[idx] = decode(in_a[idx]);
}

@compute @workgroup_size(64)
fn next_gray(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // Decode, wrapping +1, re-encode. WGSL u32 add wraps modulo 2^32.
    dst[idx] = encode(decode(in_a[idx]) + 1u);
}

@compute @workgroup_size(64)
fn prev_gray(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // Decode, wrapping -1, re-encode. WGSL u32 subtract wraps modulo 2^32.
    dst[idx] = encode(decode(in_a[idx]) - 1u);
}

@compute @workgroup_size(64)
fn gray_diff(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let diff = in_a[idx] ^ in_b[idx];
    if (popcount(diff) == 1u) {
        dst[idx] = trailing_zeros(diff);
    } else {
        dst[idx] = NONE_SENTINEL;
    }
}
"#;

/// Uniform parameters for one dispatch: the element `count` plus padding to a
/// `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`GRAY_CODE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid elements in the input and output buffers.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// A compiled, reusable set of `Gray`-code `u32` kernels (encode, decode,
/// next, prev and diff).
pub struct GpuGrayCode {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_binary_to_gray: ComputePipeline,
    pipeline_gray_to_binary: ComputePipeline,
    pipeline_next_gray: ComputePipeline,
    pipeline_prev_gray: ComputePipeline,
    pipeline_gray_diff: ComputePipeline,
}

impl GpuGrayCode {
    /// Compiles the five `Gray`-code kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGrayCode {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gray_code"),
            source: ShaderSource::Wgsl(GRAY_CODE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gray_code_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gray_code_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline_binary_to_gray = make(
            "binary_to_gray",
            "prism_volumetric_gray_code_binary_to_gray_pipeline",
        );
        let pipeline_gray_to_binary = make(
            "gray_to_binary",
            "prism_volumetric_gray_code_gray_to_binary_pipeline",
        );
        let pipeline_next_gray = make("next_gray", "prism_volumetric_gray_code_next_gray_pipeline");
        let pipeline_prev_gray = make("prev_gray", "prism_volumetric_gray_code_prev_gray_pipeline");
        let pipeline_gray_diff = make("gray_diff", "prism_volumetric_gray_code_gray_diff_pipeline");
        GpuGrayCode {
            module,
            layout,
            pipeline_binary_to_gray,
            pipeline_gray_to_binary,
            pipeline_next_gray,
            pipeline_prev_gray,
            pipeline_gray_diff,
        }
    }

    /// Encodes each `u32` counter into its reflected-binary `Gray` code,
    /// mirroring
    /// [`binary_to_gray_u32`](prism_render_architecture::particle::gray_code::binary_to_gray_u32).
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued (a storage buffer cannot be zero-sized).
    #[must_use]
    pub fn binary_to_gray(&self, ctx: &GpuContext, values: &[u32]) -> Vec<u32> {
        self.dispatch(ctx, &self.pipeline_binary_to_gray, values, values)
    }

    /// Decodes each reflected-binary `Gray` code back to its counter, mirroring
    /// [`gray_to_binary_u32`](prism_render_architecture::particle::gray_code::gray_to_binary_u32).
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    #[must_use]
    pub fn gray_to_binary(&self, ctx: &GpuContext, codes: &[u32]) -> Vec<u32> {
        self.dispatch(ctx, &self.pipeline_gray_to_binary, codes, codes)
    }

    /// Advances each code one step along the `Gray`-code cycle, mirroring
    /// [`next_gray_u32`](prism_render_architecture::particle::gray_code::next_gray_u32).
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    #[must_use]
    pub fn next_gray(&self, ctx: &GpuContext, codes: &[u32]) -> Vec<u32> {
        self.dispatch(ctx, &self.pipeline_next_gray, codes, codes)
    }

    /// Steps each code one place backwards along the `Gray`-code cycle,
    /// mirroring
    /// [`prev_gray_u32`](prism_render_architecture::particle::gray_code::prev_gray_u32).
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    #[must_use]
    pub fn prev_gray(&self, ctx: &GpuContext, codes: &[u32]) -> Vec<u32> {
        self.dispatch(ctx, &self.pipeline_prev_gray, codes, codes)
    }

    /// Reports which bit each pair of `Gray` codes disagrees on, but only when
    /// the two codes are *adjacent* (differ in a single bit), mirroring
    /// [`gray_diff_bit_index`](prism_render_architecture::particle::gray_code::gray_diff_bit_index).
    ///
    /// Returns `Some(index)` for adjacent codes and [`None`] for identical or
    /// non-adjacent codes. The kernel encodes `None` as the sentinel
    /// `0xFFFF_FFFF`; this method maps the sentinel back to [`None`] so the
    /// result compares exactly against the reference `Option<u32>`. An empty
    /// input returns an empty vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` differ in length.
    #[must_use]
    pub fn gray_diff_bit_index(&self, ctx: &GpuContext, a: &[u32], b: &[u32]) -> Vec<Option<u32>> {
        assert_eq!(a.len(), b.len(), "diff inputs must be the same length");
        let raw = self.dispatch(ctx, &self.pipeline_gray_diff, a, b);
        raw.into_iter()
            .map(|value| match value {
                NONE_SENTINEL => None,
                index => Some(index),
            })
            .collect()
    }

    /// Issues one `1-D` dispatch of `pipeline` over `a` (and `b` for the diff
    /// kernel; unary kernels bind `a` into both slots), reading the `u32`
    /// outputs back. Empty inputs short-circuit without a dispatch because a
    /// storage buffer cannot be zero-sized.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        a: &[u32],
        b: &[u32],
    ) -> Vec<u32> {
        let count = a.len();
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
            label: Some("prism_volumetric_gray_code_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let in_a = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gray_code_in_a"),
            contents: bytemuck::cast_slice(a),
            usage: BufferUsages::STORAGE,
        });
        let in_b = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gray_code_in_b"),
            contents: bytemuck::cast_slice(b),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = size_of_val(a) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gray_code_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gray_code_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: in_a.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: in_b.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gray_code_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gray_code_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gray_code_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, flattened to a 1-D dispatch.
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
        let result = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        stage.unmap();
        result
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
