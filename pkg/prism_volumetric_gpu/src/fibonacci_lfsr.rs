//! `wgpu` compute twin of the `32`-bit Fibonacci linear-feedback shift register
//! ([`fibonacci_lfsr`](prism_render_architecture::particle::fibonacci_lfsr)).
//!
//! A Fibonacci `LFSR` keeps one `32`-bit `state` register, emits the least
//! significant bit (`LSB`) each step, and shifts in a feedback bit built from
//! the maximal-length polynomial `x^32 + x^22 + x^2 + x^1 + 1`. Expressed as
//! right-shift taps that fold down to the single feedback bit, the golden
//! [`fibonacci_lfsr`](prism_render_architecture::particle::fibonacci_lfsr)
//! combines the register with its shifts by `10`, `30` and `31` before the
//! `>> 1` and re-inject into the most significant bit (`MSB`).
//!
//! [`GpuFibonacciLfsr`] is the on-device twin: one thread per element. Each
//! thread is handed a seed `state` and a step count `n`, advances the register
//! inside the kernel, and writes one `u32` output. Because the whole stream is
//! integer shifts, `XOR` and masks with no multiply, divide or floating point,
//! `CPU` and `GPU` compute the identical bit pattern, so a passing real-device
//! parity test is direct evidence the ported kernels shift, tap and pack the
//! bits exactly as the reference does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! The reference advancing logic of
//! [`FibonacciLfsr::from_state`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::from_state),
//! [`state`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::state),
//! [`next_bit`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_bit),
//! [`next_u32`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_u32)
//! and
//! [`next_array`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_array)
//! is reproduced through three per-element kernels sharing one single-step
//! helper. In each the `in_a` slot carries the seed `state` and the `in_b` slot
//! carries the per-element step count `n`:
//!
//! - [`GpuFibonacciLfsr::advance_state`] shifts the register `n` times (`n`
//!   back-to-back `next_bit` advances) and returns the resulting raw `state`,
//!   mirroring seeding with
//!   [`from_state`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::from_state),
//!   calling
//!   [`next_bit`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_bit)
//!   `n` times and reading
//!   [`state`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::state).
//! - [`GpuFibonacciLfsr::next_bit_at`] returns the output bit emitted by the
//!   `n`-th (`0`-indexed)
//!   [`next_bit`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_bit)
//!   call, which is the `LSB` of the register after `n` shifts.
//! - [`GpuFibonacciLfsr::next_u32_at`] returns the `n`-th (`0`-indexed)
//!   [`next_u32`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_u32)
//!   word: it advances the register by `n * 32` shifts, then packs the next `32`
//!   emitted bits `MSB`-first, exactly as the reference assembles a word.
//!   Mapping index `0..N` over this kernel reproduces
//!   [`next_array`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_array)
//!   element for element.
//!
//! # Degenerate all-zero state
//!
//! The all-zero register is a fixed point: it only ever emits zeros and shifts
//! to itself. The kernels contain no special case for it, so a zero seed yields
//! the identical degenerate all-zero stream on both sides and the parity holds
//! there too; keeping the seed non-zero remains the caller's responsibility, as
//! with the reference.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — the bit operators
//! `^ >> << & | ~`, the arithmetic `+ - *`, and unsigned index comparison —
//! with no transcendental call, no `countOneBits`/`firstTrailingBit` intrinsic,
//! no optional device feature and no `u64`. The `32`-bit word assembly uses a
//! constant-bound loop (`STATE_BITS` iterations); only the outer per-element
//! advance is bounded by the runtime step count. The kernels therefore run
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Every operation is pure `u32` bit algebra with no rounding anywhere on the
//! path, so `CPU` and `GPU` compute identical bit patterns. The parity test
//! asserts an exact `==` on every element with no tolerance: any mismatch is a
//! genuine port bug (a wrong tap, a flipped shift direction, a miscounted
//! step).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fibonacci_lfsr`；无第三方引擎源码或衍生代码。
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
/// that divides evenly on every target backend.
const WORKGROUP_SIZE: u32 = 64;

/// The `u32`-domain Fibonacci `LFSR` kernels, mirroring the `CPU` golden
/// [`fibonacci_lfsr`](prism_render_architecture::particle::fibonacci_lfsr) tap
/// for tap. One source file hosts three entry points sharing one bind-group
/// layout and one single-step helper.
const FIBONACCI_LFSR_WGSL: &str = r#"
// Fibonacci LFSR twin: one thread per element. Three entry points mirror the
// CPU golden `particle::fibonacci_lfsr` u32 domain. The feedback polynomial is
// x^32 + x^22 + x^2 + x^1 + 1, expressed as right-shift taps 10, 30, 31 folded
// to a single bit shifted into the MSB after a >> 1. Pure u32 shifts / xor /
// masks: no transcendental, no intrinsic, no u64, portable on Metal, Vulkan and
// DX12. The 32-bit word pack uses a constant-bound loop; only the per-element
// advance is bounded by the runtime step count `n`.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::fibonacci_lfsr；无第三方
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
@group(0) @binding(1) var<storage, read> in_seed: array<u32>;
@group(0) @binding(2) var<storage, read> in_steps: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

// Width of the shift register in bits (STATE_BITS - 1 is the MSB position).
const STATE_BITS: u32 = 32u;

// Right-shift tap positions derived from the feedback polynomial.
const TAP_A: u32 = 10u;
const TAP_B: u32 = 30u;
const TAP_C: u32 = 31u;

// Advance the register one shift, returning the next state. The feedback bit is
// the xor of the polynomial taps masked to one bit, shifted into the MSB after
// the register shifts right by one. Matches the reference `next_bit` state
// update exactly.
fn step_state(s: u32) -> u32 {
    let fb = (s ^ (s >> TAP_A) ^ (s >> TAP_B) ^ (s >> TAP_C)) & 1u;
    return (s >> 1u) | (fb << (STATE_BITS - 1u));
}

// Advance the register by `n` single-bit shifts and return the resulting state.
fn advance(seed: u32, n: u32) -> u32 {
    var s = seed;
    for (var i = 0u; i < n; i = i + 1u) {
        s = step_state(s);
    }
    return s;
}

@compute @workgroup_size(64)
fn advance_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // State after `n` next_bit shifts, i.e. `.state()` after n next_bit calls.
    dst[idx] = advance(in_seed[idx], in_steps[idx]);
}

@compute @workgroup_size(64)
fn next_bit_at(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // Output bit of the n-th (0-indexed) next_bit call: next_bit reads the LSB
    // before shifting, so the n-th emitted bit is the LSB after n shifts.
    dst[idx] = advance(in_seed[idx], in_steps[idx]) & 1u;
}

@compute @workgroup_size(64)
fn next_u32_at(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // n-th (0-indexed) next_u32 word: advance past the first n words, then pack
    // the next 32 emitted bits MSB-first, exactly as the reference assembles.
    var s = advance(in_seed[idx], in_steps[idx] * STATE_BITS);
    var w = 0u;
    for (var b = 0u; b < STATE_BITS; b = b + 1u) {
        let outbit = s & 1u;
        s = step_state(s);
        w = (w << 1u) | outbit;
    }
    dst[idx] = w;
}
"#;

/// Uniform parameters for one dispatch: the element `count` plus padding to a
/// `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`FIBONACCI_LFSR_WGSL`].
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

/// A compiled, reusable set of `u32` Fibonacci `LFSR` kernels (advance-state,
/// nth output bit and nth `32`-bit word).
pub struct GpuFibonacciLfsr {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_advance_state: ComputePipeline,
    pipeline_next_bit_at: ComputePipeline,
    pipeline_next_u32_at: ComputePipeline,
}

impl GpuFibonacciLfsr {
    /// Compiles the three Fibonacci `LFSR` kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFibonacciLfsr {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fibonacci_lfsr"),
            source: ShaderSource::Wgsl(FIBONACCI_LFSR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fibonacci_lfsr_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fibonacci_lfsr_pipeline_layout"),
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
        let pipeline_advance_state = make(
            "advance_state",
            "prism_volumetric_fibonacci_lfsr_advance_state_pipeline",
        );
        let pipeline_next_bit_at = make(
            "next_bit_at",
            "prism_volumetric_fibonacci_lfsr_next_bit_at_pipeline",
        );
        let pipeline_next_u32_at = make(
            "next_u32_at",
            "prism_volumetric_fibonacci_lfsr_next_u32_at_pipeline",
        );
        GpuFibonacciLfsr {
            module,
            layout,
            pipeline_advance_state,
            pipeline_next_bit_at,
            pipeline_next_u32_at,
        }
    }

    /// Advances each seed register by its paired `steps` count of single-bit
    /// shifts and returns the resulting raw `state`, mirroring seeding with
    /// [`from_state`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::from_state),
    /// running that many
    /// [`next_bit`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_bit)
    /// calls, and reading
    /// [`state`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::state).
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued (a storage buffer cannot be zero-sized).
    ///
    /// # Panics
    ///
    /// Panics if `seeds` and `steps` differ in length.
    #[must_use]
    pub fn advance_state(&self, ctx: &GpuContext, seeds: &[u32], steps: &[u32]) -> Vec<u32> {
        assert_eq!(
            seeds.len(),
            steps.len(),
            "seeds and steps must be the same length"
        );
        self.dispatch(ctx, &self.pipeline_advance_state, seeds, steps)
    }

    /// Returns the output bit emitted by the `n`-th (`0`-indexed)
    /// [`next_bit`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_bit)
    /// call for each seed, where `n` is the paired `indices` entry. Each result
    /// is `0` or `1`.
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `seeds` and `indices` differ in length.
    #[must_use]
    pub fn next_bit_at(&self, ctx: &GpuContext, seeds: &[u32], indices: &[u32]) -> Vec<u32> {
        assert_eq!(
            seeds.len(),
            indices.len(),
            "seeds and indices must be the same length"
        );
        self.dispatch(ctx, &self.pipeline_next_bit_at, seeds, indices)
    }

    /// Returns the `n`-th (`0`-indexed)
    /// [`next_u32`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_u32)
    /// word for each seed, where `n` is the paired `indices` entry. Mapping
    /// `0..N` over this reproduces
    /// [`next_array`](prism_render_architecture::particle::fibonacci_lfsr::FibonacciLfsr::next_array)
    /// element for element.
    ///
    /// Returns one output per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `seeds` and `indices` differ in length.
    #[must_use]
    pub fn next_u32_at(&self, ctx: &GpuContext, seeds: &[u32], indices: &[u32]) -> Vec<u32> {
        assert_eq!(
            seeds.len(),
            indices.len(),
            "seeds and indices must be the same length"
        );
        self.dispatch(ctx, &self.pipeline_next_u32_at, seeds, indices)
    }

    /// Issues one `1-D` dispatch of `pipeline` over the paired `seeds` and
    /// `steps` inputs, reading the `u32` outputs back. Empty inputs
    /// short-circuit without a dispatch because a storage buffer cannot be
    /// zero-sized.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        seeds: &[u32],
        steps: &[u32],
    ) -> Vec<u32> {
        let count = seeds.len();
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
            label: Some("prism_volumetric_fibonacci_lfsr_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let in_seed = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fibonacci_lfsr_in_seed"),
            contents: bytemuck::cast_slice(seeds),
            usage: BufferUsages::STORAGE,
        });
        let in_steps = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fibonacci_lfsr_in_steps"),
            contents: bytemuck::cast_slice(steps),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = size_of_val(seeds) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fibonacci_lfsr_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fibonacci_lfsr_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: in_seed.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: in_steps.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fibonacci_lfsr_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fibonacci_lfsr_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fibonacci_lfsr_pass"),
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
