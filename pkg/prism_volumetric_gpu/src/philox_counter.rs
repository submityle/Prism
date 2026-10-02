//! `wgpu` compute twin of the `Philox4x32-10` counter-based `PRNG`
//! ([`philox_counter`](prism_render_architecture::particle::philox_counter)).
//!
//! `Philox` is a *counter-based* generator: the i-th block of randomness is a
//! pure, stateless function `philox4x32(ctr, key)` of a `128`-bit counter (four
//! `u32` words) and a `64`-bit key (two `u32` words). The `CPU` golden
//! [`philox4x32`](prism_render_architecture::particle::philox_counter::philox4x32)
//! runs ten Feistel-like rounds built on integer multiply-high/multiply-low
//! (`mulhilo`), bumping the key by the two Weyl constants between rounds. This
//! statelessness is exactly what a massively parallel `GPU` particle system
//! wants: any thread can jump straight to the block for a given particle index
//! without replaying the stream.
//!
//! [`GpuPhiloxCounter`] is the on-device twin: one thread computes
//! `philox4x32(ctr, key)` for one query. Because every operation is `32`-bit
//! wrapping integer and bitwise arithmetic with no floating point, `CPU` and
//! `GPU` compute the identical `128`-bit output block, so a passing real-device
//! parity test is direct evidence the ported kernel multiplies, splits and
//! mixes the words exactly as the reference does, not merely that the shader
//! compiles.
//!
//! # What is twinned
//!
//! The pure block function
//! [`philox4x32`](prism_render_architecture::particle::philox_counter::philox4x32)
//! is reproduced for a batch of independent `(ctr, key)` queries, each thread
//! emitting the four output `u32` words. The counter-walking semantics of
//! [`Philox4x32::next_block`](prism_render_architecture::particle::philox_counter::Philox4x32::next_block)
//! are twinned by letting the host pre-generate the little-endian-incremented
//! counter for each element and feeding those counters as queries: the i-th
//! walked block equals `philox4x32` of the i-th counter, which the kernel
//! reproduces element for element (the parity test drives the reference stream
//! and compares).
//!
//! # The `mulhilo` `32x32->64` reconstruction
//!
//! The reference `mulhilo(a, b)` forms the `64`-bit product `(a as u64) * (b as
//! u64)` and splits it into `(hi, lo)`. `WGSL` has no `u64`, so the kernel
//! rebuilds the full `64`-bit product from `32`-bit `schoolbook` arithmetic.
//! Each operand is split into high and low `16`-bit halves, giving four partial
//! products `ll`, `lh`, `hl`, `hh`, each a product of two `16`-bit values and so
//! bounded by `2^32` (it fits a `u32` with no overflow). They are summed with
//! explicit `16`-bit carry propagation: `cross` folds the high half of `ll`
//! with the low halves of the two middle products, `lo` places `cross`'s low
//! `16` bits above `ll`'s low `16` bits, and `hi` accumulates `hh`, the two
//! middle products' high halves and `cross`'s carry-out. The result is the
//! bit-exact `(hi, lo)` the `u64` multiply would produce, verified at the
//! extreme `0xFFFF_FFFF^2` corner where the carry chain is longest.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — the arithmetic `+ *`,
//! the bit operators `^ >> << & |`, and unsigned index comparison — with no
//! transcendental call, no `countOneBits`/`firstTrailingBit` intrinsic, no
//! optional device feature and no `u64`. Every shift amount is a `u32` constant
//! below `32`. The round count is a fixed, constant-bound loop, so the kernel
//! provably terminates and runs unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Every operation is pure `u32` wrapping and bit algebra with no rounding
//! anywhere on the path, so `CPU` and `GPU` compute identical bit patterns. The
//! parity test asserts an exact `==` on every output word with no tolerance:
//! any mismatch is a genuine port bug (a wrong multiplier, a dropped carry, a
//! swapped lane).
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::philox_counter`；无第三方引擎源码或衍生代码。
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

/// The portable core-`WGSL` `Philox4x32-10` kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `solve` mirrors the
/// `CPU` golden
/// [`philox4x32`](prism_render_architecture::particle::philox_counter::philox4x32)
/// round for round; see the module documentation for the `mulhilo`
/// reconstruction.
const PHILOX_COUNTER_WGSL: &str = r#"
// Philox4x32-10 twin: one thread per query reproduces the CPU golden
// `particle::philox_counter::philox4x32`. Ten Feistel-like rounds built on a
// u32 32x32->64 schoolbook multiply (mulhilo), with the key bumped by the two
// Weyl constants between rounds. Pure u32 wrapping add / xor / shift / mask: no
// transcendental, no intrinsic, no u64, every shift amount a u32 constant below
// 32. The round count is a constant-bound loop, so the kernel terminates and is
// portable on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::philox_counter；无第三方
// 引擎源码或衍生代码。

struct Params {
    // Number of valid queries; threads past this short-circuit.
    count: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // 128-bit counter (little-endian across the four words).
    ctr: vec4<u32>,
    // 64-bit key.
    key: vec2<u32>,
    // Padding to a 16-byte-aligned std430 slot.
    pad0: u32,
    pad1: u32,
}

struct Result {
    // The four-word Philox output block.
    block: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// First-lane counter multiplier (odd 32-bit Random123 constant).
const M0: u32 = 0xD2511F53u;
// Second-lane counter multiplier (odd 32-bit Random123 constant).
const M1: u32 = 0xCD9E8D57u;
// First Weyl key-bump constant (fractional bits of the golden ratio).
const W0: u32 = 0x9E3779B9u;
// Second Weyl key-bump constant (fractional bits of sqrt(3)).
const W1: u32 = 0xBB67AE85u;
// Number of mixing rounds in the canonical Philox4x32-10 variant.
const ROUNDS: u32 = 10u;
// Mask for the low 16 bits of a word.
const LOW16: u32 = 0xFFFFu;
// Half-word shift amount (always below 32).
const HALF: u32 = 16u;

// The (hi, lo) halves of a 64-bit product rebuilt from u32 arithmetic.
struct HiLo {
    hi: u32,
    lo: u32,
}

// Multiply two u32 values and return the (high, low) 32-bit halves of the
// 64-bit product, mirroring the reference `mulhilo`. WGSL has no u64, so the
// product is rebuilt from four 16-bit partial products with explicit carry
// propagation. Each partial product multiplies two 16-bit values and so fits a
// u32 with no overflow.
fn mulhilo(a: u32, b: u32) -> HiLo {
    let a_lo = a & LOW16;
    let a_hi = a >> HALF;
    let b_lo = b & LOW16;
    let b_hi = b >> HALF;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    // Fold the carry out of bit 16: the high half of ll plus the low halves of
    // the two middle products. Bounded by 3 * (2^16 - 1), so it fits a u32.
    let cross = (ll >> HALF) + (lh & LOW16) + (hl & LOW16);
    // Low 32 bits: cross's low 16 bits (its high bits wrap away under the <<
    // 16) sit above ll's low 16 bits.
    let lo = (cross << HALF) | (ll & LOW16);
    // High 32 bits: hh plus the middle products' high halves plus cross's
    // carry-out. The true product's high word fits a u32, so this cannot
    // overflow.
    let hi = hh + (lh >> HALF) + (hl >> HALF) + (cross >> HALF);

    return HiLo(hi, lo);
}

// Apply one Philox4x32 round to the counter `c` under the round key `k`. Two
// lanes are multiplied by the fixed multipliers and each product is split via
// mulhilo; the high and low halves cross-combine with the surviving counter
// words and the key through XOR, exactly as the reference `round`.
fn philox_round(c: vec4<u32>, k: vec2<u32>) -> vec4<u32> {
    let m0 = mulhilo(M0, c.x);
    let m1 = mulhilo(M1, c.z);
    return vec4<u32>(m1.hi ^ c.y ^ k.x, m1.lo, m0.hi ^ c.w ^ k.y, m0.lo);
}

// Advance the key by the two Weyl constants (one bump between rounds). u32 add
// wraps modulo 2^32 in WGSL, matching the reference `wrapping_add`.
fn bump_key(k: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(k.x + W0, k.y + W1);
}

// Compute the Philox4x32-10 output block for counter `ctr` under `key`: ten
// rounds, bumping the key after each round, mirroring the reference loop.
fn philox4x32(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var c = ctr;
    var k = key;
    for (var i = 0u; i < ROUNDS; i = i + 1u) {
        c = philox_round(c, k);
        k = bump_key(k);
    }
    return c;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var out: Result;
    out.block = philox4x32(q.ctr, q.key);
    results[idx] = out;
}
"#;

/// Uniform parameters for one dispatch: the query `count` plus padding to a
/// `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`PHILOX_COUNTER_WGSL`].
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
/// The `vec4` counter and `vec2` key are followed by two pad words so the slot
/// stays `16`-byte aligned on device.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// `128`-bit counter (little-endian across the four words).
    ctr: [u32; 4],
    /// `64`-bit key.
    key: [u32; 2],
    /// Pad word after `key`.
    pad0: u32,
    /// Pad word after `key`.
    pad1: u32,
}

/// `repr(C)` `std430` layout of one result, matching the `WGSL` `Result`
/// struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The four-word `Philox` output block.
    block: [u32; 4],
}

/// One `Philox4x32` query: a `128`-bit counter and a `64`-bit key.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::philox_counter`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhiloxCounterQuery {
    /// The `128`-bit counter (four `u32` words, little-endian), matching the
    /// `ctr` argument of
    /// [`philox4x32`](prism_render_architecture::particle::philox_counter::philox4x32).
    pub ctr: [u32; 4],
    /// The `64`-bit key (two `u32` words), matching the `key` argument of
    /// [`philox4x32`](prism_render_architecture::particle::philox_counter::philox4x32).
    pub key: [u32; 2],
}

/// One resolved `Philox4x32` output block.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::philox_counter`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhiloxCounterResult {
    /// The four `u32` output words, matching the return of
    /// [`philox4x32`](prism_render_architecture::particle::philox_counter::philox4x32).
    pub block: [u32; 4],
}

/// Encodes one [`PhiloxCounterQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &PhiloxCounterQuery) -> GpuQuery {
    GpuQuery {
        ctr: q.ctr,
        key: q.key,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one packed [`GpuResult`] into the public [`PhiloxCounterResult`].
fn decode_result(raw: &GpuResult) -> PhiloxCounterResult {
    PhiloxCounterResult { block: raw.block }
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

/// A compiled, reusable `Philox4x32-10` compute pipeline, twinning the `CPU`
/// golden
/// [`philox4x32`](prism_render_architecture::particle::philox_counter::philox4x32).
pub struct GpuPhiloxCounter {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuPhiloxCounter {
    /// Compiles the `Philox4x32-10` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuPhiloxCounter {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_philox_counter"),
            source: ShaderSource::Wgsl(PHILOX_COUNTER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_philox_counter_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_philox_counter_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_philox_counter_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuPhiloxCounter {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes `philox4x32(ctr, key)` for every query and returns one
    /// [`PhiloxCounterResult`] per input, in order.
    ///
    /// Each output block equals the `CPU` golden
    /// [`philox4x32`](prism_render_architecture::particle::philox_counter::philox4x32)
    /// bit for bit. An empty `queries` batch returns an empty vector with no
    /// dispatch issued, since a storage buffer cannot be zero-sized.
    #[must_use]
    pub fn evaluate(
        &self,
        ctx: &GpuContext,
        queries: &[PhiloxCounterQuery],
    ) -> Vec<PhiloxCounterResult> {
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
            label: Some("prism_volumetric_philox_counter_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let encoded: Vec<GpuQuery> = queries.iter().map(encode_query).collect();
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_philox_counter_queries"),
            contents: bytemuck::cast_slice(&encoded),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_philox_counter_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_philox_counter_bind_group"),
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
            label: Some("prism_volumetric_philox_counter_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_philox_counter_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_philox_counter_pass"),
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
