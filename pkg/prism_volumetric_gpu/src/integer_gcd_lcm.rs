//! `wgpu` compute twin of the exact integer least-common-multiple with an
//! explicit `u32` overflow guard
//! ([`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm)).
//!
//! The `CPU` golden standard owns the integer math in the `u64` domain: the
//! least-common-multiple identity
//! [`lcm_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_u64)
//! and its overflow-reporting companion
//! [`lcm_checked_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_checked_u64).
//! [`GpuIntegerGcdLcm`] is the on-device twin that runs one thread per query
//! and reproduces the divide-first identity `lcm(a, b) == a / gcd(a, b) * b`
//! step for step, reporting an explicit overflow flag whenever the true `lcm`
//! does not fit in `u32`.
//!
//! # Deduplication (honest provenance)
//!
//! The greatest-common-divisor primitive itself is already twinned by
//! [`integer_gcd`](crate::integer_gcd); this module does **not** re-export or
//! duplicate that `GPU` surface. `lcm` needs a `gcd` internally, so the kernel
//! inlines a private `Stein` binary-`gcd` helper (the same hand-rolled
//! `trailing_zeros` reduction used by the existing `gcd` twin) purely as a
//! building block. The novel surface twinned here is the
//! least-common-multiple together with the `u32` overflow guard, not the `gcd`
//! reduction.
//!
//! # Domain constraint (honest provenance)
//!
//! The golden standard works in the `u64` domain. `WGSL` has **no** `u64`
//! type, so this twin mirrors the identity in the **`u32` input domain**: for
//! every pair `a <= u32::MAX`, `b <= u32::MAX` the kernel computes
//! `a / gcd(a, b) * b` exactly in `u32`. The divide is exact because the `gcd`
//! divides `a`, and the final multiply is overflow-checked in `u32`: when the
//! true `lcm` exceeds `u32::MAX` the kernel sets the overflow flag to `1` and
//! writes a sentinel value of `0` instead of silently wrapping. This matches
//! [`lcm_checked_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_checked_u64)
//! restricted to the `u32` range: a result that fits is reported with the flag
//! clear, a result that does not is reported with the flag set. This is not a
//! stub; the identity and the guard are faithful, only the width is pinned to
//! `u32`.
//!
//! # What is twinned
//!
//! A single kernel runs one thread per `(a, b)` pair and reproduces the
//! least-common-multiple identity with an overflow guard:
//!
//! * The degenerate convention `lcm(a, 0) == lcm(0, b) == 0` falls out of an
//!   early branch, matching
//!   [`lcm_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_u64).
//! * Otherwise `g = gcd(a, b)` is computed with an inlined `Stein` binary
//!   reduction, `q = a / g` is exact, and `prod = q * b` is the candidate
//!   `lcm`.
//! * The overflow guard reports `1` when the `u32` multiply `q * b` wraps,
//!   detected by the standard unsigned test `prod / q != b`, mirroring the
//!   `checked_mul` in
//!   [`lcm_checked_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_checked_u64).
//!
//! The extended-Euclidean reference routine
//! ([`ext_gcd_i64`](prism_render_architecture::particle::integer_gcd_lcm::ext_gcd_i64))
//! and the signed/`Option` wrappers
//! ([`gcd_i64`](prism_render_architecture::particle::integer_gcd_lcm::gcd_i64),
//! [`coprime_u64`](prism_render_architecture::particle::integer_gcd_lcm::coprime_u64),
//! the `Option`封装 of
//! [`lcm_checked_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_checked_u64))
//! stay on the host: they need `i64`/`i128` arithmetic or an `Option` return
//! that `WGSL`'s `u32`-only integer set cannot represent. The host uses
//! `lcm_checked_u64` to generate the parity oracle.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — the operators
//! `% >> << & | > ==`, `+ - *` and unsigned index arithmetic — with a
//! hand-rolled `trailing_zeros` built from masks and shifts (no
//! `firstTrailingBit` intrinsic), no transcendental call and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The binary
//! reduction loop carries a static upper bound of `64` iterations: the `u32`
//! Euclidean worst case is a `Fibonacci` pair at roughly `47` steps, so `64` is
//! a safe ceiling that keeps the loop statically bounded for every input.
//!
//! # Correctness model
//!
//! Every operation is pure unsigned integer arithmetic with no reordering and
//! no rounding, so `CPU` (restricted to the `u32` domain) and `GPU` compute
//! bit-identical results. The parity test asserts an exact `==` on both the
//! `u32` value and the overflow flag with no tolerance: any mismatch is a
//! genuine port bug.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm)
//! plus `wgpu` compute dispatch; no third-party engine source or derived code.

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
/// shared by the other twins in this crate.
const WORKGROUP_SIZE: u32 = 64;

/// The inlined least-common-multiple kernel. One thread per `(a, b)` pair
/// reproduces the golden
/// [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm)
/// identity with an explicit `u32` overflow guard; see the module documentation
/// for the algorithm.
const INTEGER_GCD_LCM_WGSL: &str = r#"
// Integer-lcm twin: one thread per query computes `a / gcd(a, b) * b` in the
// u32 input domain (WGSL has no u64) with an explicit overflow flag. It mirrors
// the CPU golden `particle::integer_gcd_lcm` lcm identity step for step, uses
// only the portable core-WGSL subset (% >> << & | > ==, + - * and a
// hand-rolled trailing_zeros) and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// The gcd primitive is already twinned elsewhere; here binary_gcd is a private
// building block for the lcm, not a separately exposed path.
//
// Provenance: twinned from this repository's `particle::integer_gcd_lcm`; no
// third-party engine source or derived code.

struct Params {
    // Number of valid queries in `queries`.
    count: u32,
    // Padding to a 16-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One query. 16-byte std430 stride matching the host `GpuQuery`: the two u32
// operands and two pad words.
struct Query {
    a: u32,
    b: u32,
    pad0: u32,
    pad1: u32,
}

// One result: the lcm value (or a 0 sentinel on overflow) and the overflow
// flag (0 or 1). 8-byte std430 stride matching the host `GpuResult`.
struct Result {
    value: u32,
    overflow: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Hand-rolled trailing-zero count (binary search), standing in for
// `u32::trailing_zeros`: no `firstTrailingBit` intrinsic. Every call site below
// passes a value known to be non-zero, so the degenerate x == 0 input never
// occurs on the twinned paths.
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

// Stein's binary gcd, a private building block mirroring the golden
// `binary_gcd_u64`: factor out the common power of two once, reduce `a` to odd,
// then on each round reduce `b` to odd, order the pair and subtract. `a | b`
// and the per-round `b` are non-zero at every `trailing_zeros` call, and the
// final `shift` is < 32.
fn binary_gcd(a0: u32, b0: u32) -> u32 {
    if (a0 == 0u) { return b0; }
    if (b0 == 0u) { return a0; }
    let shift = trailing_zeros(a0 | b0);
    var a = a0 >> trailing_zeros(a0);
    var b = b0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        if (b == 0u) { break; }
        b = b >> trailing_zeros(b);
        if (a > b) {
            let t = a;
            a = b;
            b = t;
        }
        b = b - a;
    }
    return a << shift;
}

// Least-common-multiple with a u32 overflow guard, mirroring the golden
// `lcm_u64` / `lcm_checked_u64`: lcm(a, 0) == lcm(0, b) == 0; otherwise
// q = a / gcd(a, b) is exact and prod = q * b is the candidate. The multiply is
// overflow-checked with the standard unsigned test prod / q != b; on overflow
// the flag is 1 and the value is a 0 sentinel.
fn lcm_guarded(a: u32, b: u32) -> Result {
    var out: Result;
    if (a == 0u || b == 0u) {
        out.value = 0u;
        out.overflow = 0u;
        return out;
    }
    let g = binary_gcd(a, b);
    let q = a / g;
    let prod = q * b;
    // `q` is always non-zero here (g divides a and a > 0), so the guard reduces
    // to the wrap test prod / q != b.
    if ((prod / q) != b) {
        out.value = 0u;
        out.overflow = 1u;
        return out;
    }
    out.value = prod;
    out.overflow = 0u;
    return out;
}

@compute @workgroup_size(64)
fn lcm_dispatch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = lcm_guarded(q.a, q.b);
}
"#;

/// One least-common-multiple query: an unsigned operand pair.
///
/// The twin computes `lcm(a, b)` in the `u32` domain. Derives [`Eq`] because
/// every field is an integer.
///
/// Provenance: twinned from this repository's
/// [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerGcdLcmQuery {
    /// First operand.
    pub a: u32,
    /// Second operand.
    pub b: u32,
}

impl IntegerGcdLcmQuery {
    /// Builds a query from an operand pair.
    #[must_use]
    pub const fn new(a: u32, b: u32) -> IntegerGcdLcmQuery {
        IntegerGcdLcmQuery { a, b }
    }
}

/// The least-common-multiple answer for one query.
///
/// `lcm` holds the least common multiple in the `u32` domain when it fits;
/// `overflow` is `true` when the true `lcm` exceeds `u32::MAX`, in which case
/// `lcm` carries the `0` sentinel. Mirrors the `Some`/`None` distinction of the
/// golden
/// [`lcm_checked_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_checked_u64)
/// restricted to the `u32` range. Derives [`Eq`] because every field is an
/// integer or a `bool`.
///
/// Provenance: twinned from this repository's
/// [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerGcdLcmResult {
    /// Least common multiple in the `u32` domain, or the `0` sentinel on
    /// overflow.
    pub lcm: u32,
    /// `true` when the true `lcm` does not fit in `u32`.
    pub overflow: bool,
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`INTEGER_GCD_LCM_WGSL`]: the query count and three pad words —
/// `16` bytes, each field at the uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of valid queries.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One query as uploaded. `16`-byte `std430` stride matching `Query` in the
/// shader: the two operands and two pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First operand.
    a: u32,
    /// Second operand.
    b: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One result as read back. `8`-byte `std430` stride matching `Result` in the
/// shader: the `lcm` value and the overflow flag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Least common multiple, or the `0` sentinel on overflow.
    value: u32,
    /// Overflow flag (`0` or `1`).
    overflow: u32,
}

/// Packs one [`IntegerGcdLcmQuery`] into its `std430` [`GpuQuery`] slot.
fn encode_query(q: &IntegerGcdLcmQuery) -> GpuQuery {
    GpuQuery {
        a: q.a,
        b: q.b,
        pad0: 0,
        pad1: 0,
    }
}

/// Decodes one `std430` [`GpuResult`] into a host [`IntegerGcdLcmResult`].
fn decode_result(r: &GpuResult) -> IntegerGcdLcmResult {
    IntegerGcdLcmResult {
        lcm: r.value,
        overflow: r.overflow != 0,
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

/// A compiled, reusable integer-`lcm` dispatch pipeline.
pub struct GpuIntegerGcdLcm {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuIntegerGcdLcm {
    /// Compiles the integer-`lcm` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuIntegerGcdLcm {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_integer_gcd_lcm"),
            source: ShaderSource::Wgsl(INTEGER_GCD_LCM_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_integer_gcd_lcm_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_integer_gcd_lcm_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_integer_gcd_lcm_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("lcm_dispatch"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuIntegerGcdLcm {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries`, returning one [`IntegerGcdLcmResult`] per
    /// query.
    ///
    /// Each answer reproduces the golden
    /// [`lcm_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_u64)
    /// evaluated in the `u32` domain, with the overflow flag matching
    /// [`lcm_checked_u64`](prism_render_architecture::particle::integer_gcd_lcm::lcm_checked_u64)
    /// restricted to that range. An empty `queries` slice yields an empty
    /// vector — storage buffers cannot be zero-sized, so it is handled by an
    /// early return before any dispatch.
    #[must_use]
    pub fn run(
        &self,
        ctx: &GpuContext,
        queries: &[IntegerGcdLcmQuery],
    ) -> Vec<IntegerGcdLcmResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_params = GpuParams {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let gpu_queries: Vec<GpuQuery> = queries.iter().map(encode_query).collect();

        let out_bytes = (queries.len() * size_of::<GpuResult>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_integer_gcd_lcm_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_integer_gcd_lcm_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_integer_gcd_lcm_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_integer_gcd_lcm_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_integer_gcd_lcm_bind_group"),
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
            label: Some("prism_volumetric_integer_gcd_lcm_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_integer_gcd_lcm_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());
        gpu_results.iter().map(decode_result).collect()
    }
}
