//! `wgpu` compute twin of the exact integer greatest-common-divisor
//! primitives
//! ([`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm)).
//!
//! The `CPU` golden standard owns the integer math: the iterative Euclidean
//! [`gcd_u64`](prism_render_architecture::particle::integer_gcd_lcm::gcd_u64),
//! `Stein`'s binary
//! [`binary_gcd_u64`](prism_render_architecture::particle::integer_gcd_lcm::binary_gcd_u64)
//! and the divisibility predicate
//! [`coprime_u64`](prism_render_architecture::particle::integer_gcd_lcm::coprime_u64).
//! [`GpuIntegerGcd`] is the on-device twin that runs one thread per query and
//! reproduces those algorithms step for step, so a passing real-device parity
//! test is direct evidence the ported kernel runs the identical integer
//! reduction, not merely that the shader compiles.
//!
//! # Domain constraint (honest provenance)
//!
//! The golden standard works in the `u64` domain. `WGSL` has **no** `u64`
//! type, so this twin mirrors each algorithm in the **`u32` input domain**:
//! every reduction step, swap and shift matches the reference line for line,
//! only the operand bit width is narrowed to `u32`. For every pair of inputs
//! with `a <= u32::MAX` and `b <= u32::MAX` the kernel's `gcd` is bit-identical
//! to [`gcd_u64`](prism_render_architecture::particle::integer_gcd_lcm::gcd_u64)
//! evaluated on the same operands widened to `u64` (the true `gcd` of two `u32`
//! values is itself always `<= u32::MAX`, so no result is ever truncated). This
//! is not a stub: the algorithms are faithful, only the width is pinned to
//! `u32`.
//!
//! # What is twinned
//!
//! A single kernel dispatches on a per-query operation selector and reproduces
//! three reference routines:
//!
//! * [`GpuGcdOp::Gcd`] mirrors
//!   [`gcd_u64`](prism_render_architecture::particle::integer_gcd_lcm::gcd_u64):
//!   the iterative Euclidean remainder loop `(a, b) -> (b, a % b)` until `b`
//!   reaches `0`, with the `gcd(0, 0) == 0` convention.
//! * [`GpuGcdOp::BinaryGcd`] mirrors
//!   [`binary_gcd_u64`](prism_render_architecture::particle::integer_gcd_lcm::binary_gcd_u64):
//!   `Stein`'s method factors out the common power of two once, reduces each
//!   operand to odd with a `trailing_zeros` shift, orders the pair and
//!   subtracts, agreeing with the Euclidean result on every input.
//! * [`GpuGcdOp::Coprime`] mirrors
//!   [`coprime_u64`](prism_render_architecture::particle::integer_gcd_lcm::coprime_u64):
//!   it reports `1` when `gcd(a, b) == 1` and `0` otherwise.
//!
//! The least-common-multiple and extended-Euclidean reference routines
//! (`lcm_u64`, `lcm_checked_u64`, `ext_gcd_i64`) are deliberately **not**
//! twinned: `lcm` needs a `checked_mul` in a width wider than the operands and
//! the extended recurrence accumulates `Bezout` coefficients in the `i128`
//! domain, neither of which `WGSL`'s `u32`-only integer set can represent.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — the operators
//! `% >> << & | > ==`, `+ - *` and unsigned index arithmetic — with a
//! hand-rolled `trailing_zeros` built from masks and shifts (no
//! `firstTrailingBit` intrinsic), no transcendental call and no optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. Both
//! reduction loops carry a static upper bound of `64` iterations: the `u32`
//! Euclidean worst case is a `Fibonacci` pair at roughly `47` steps, so `64` is
//! a safe ceiling that keeps the loop statically bounded for every input.
//!
//! # Correctness model
//!
//! Every operation is pure unsigned integer arithmetic with no reordering and
//! no rounding, so `CPU` (restricted to the `u32` domain) and `GPU` compute
//! bit-identical results. The parity test asserts an exact `==` on every `u32`
//! output with no tolerance: any mismatch is a genuine port bug.
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

/// Stable operation code for the Euclidean `gcd` kernel path, matching
/// `case 0u` in [`INTEGER_GCD_WGSL`].
const OP_GCD: u32 = 0;

/// Stable operation code for the binary (`Stein`) `gcd` kernel path, matching
/// `case 1u` in [`INTEGER_GCD_WGSL`].
const OP_BINARY_GCD: u32 = 1;

/// Stable operation code for the coprime predicate kernel path, matching
/// `case 2u` in [`INTEGER_GCD_WGSL`].
const OP_COPRIME: u32 = 2;

/// The portable core-`WGSL` integer-`gcd` kernel, embedded inline so the twin
/// ships as a single source file. One thread per query dispatches on the stable
/// operation code and mirrors the `CPU` golden
/// [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm)
/// routines step for step; see the module documentation for the algorithms.
const INTEGER_GCD_WGSL: &str = r#"
// Integer-gcd twin: one thread per query reduces a `u32` operand pair with the
// algorithm selected by `op` and writes a single `u32` answer. It mirrors the
// CPU golden `particle::integer_gcd_lcm` step for step in the u32 input domain
// (WGSL has no u64), uses only the portable core-WGSL subset (% >> << & | > ==,
// + - * and a hand-rolled trailing_zeros) and takes no optional feature, so it
// runs unmodified on Metal, Vulkan and DX12.
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
// operands, the operation selector and one pad word.
struct Query {
    a: u32,
    b: u32,
    op: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

// Stable operation codes, mirroring the host `GpuGcdOp`.
const OP_GCD: u32 = 0u;
const OP_BINARY_GCD: u32 = 1u;
const OP_COPRIME: u32 = 2u;

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

// Euclidean gcd, mirroring the golden `gcd_u64`: repeatedly replace (a, b) with
// (b, a % b) until b is zero. The loop carries a static 64-step bound (the u32
// worst case is a Fibonacci pair at ~47 steps). The convention gcd(0, 0) == 0
// and gcd(n, 0) == gcd(0, n) == n falls out of the early break.
fn gcd(a0: u32, b0: u32) -> u32 {
    var a = a0;
    var b = b0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        if (b == 0u) { break; }
        let r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// Stein's binary gcd, mirroring the golden `binary_gcd_u64`: factor out the
// common power of two once, reduce `a` to odd, then on each round reduce `b` to
// odd, order the pair and subtract. `a | b` and the per-round `b` are non-zero
// at every `trailing_zeros` call, and the final `shift` is < 32.
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

// Coprime predicate, mirroring the golden `coprime_u64`: 1 when gcd(a, b) == 1,
// else 0.
fn coprime(a: u32, b: u32) -> u32 {
    if (gcd(a, b) == 1u) {
        return 1u;
    }
    return 0u;
}

@compute @workgroup_size(64)
fn gcd_dispatch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    var out = 0u;
    switch (q.op) {
        case 0u: { out = gcd(q.a, q.b); }           // OP_GCD
        case 1u: { out = binary_gcd(q.a, q.b); }    // OP_BINARY_GCD
        case 2u: { out = coprime(q.a, q.b); }       // OP_COPRIME
        default: { out = 0u; }
    }
    results[idx] = out;
}
"#;

/// The integer-`gcd` operation one query selects, mirroring the twinned subset
/// of the `CPU` golden
/// [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm).
///
/// The host encodes each variant into the device buffer through its stable
/// `u32` code; the kernel's `switch` dispatches on the same code.
///
/// Provenance: twinned from this repository's
/// [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuGcdOp {
    /// Euclidean greatest common divisor, mirroring
    /// [`gcd_u64`](prism_render_architecture::particle::integer_gcd_lcm::gcd_u64).
    Gcd,
    /// `Stein`'s binary greatest common divisor, mirroring
    /// [`binary_gcd_u64`](prism_render_architecture::particle::integer_gcd_lcm::binary_gcd_u64).
    BinaryGcd,
    /// Coprime predicate (`1` when the `gcd` is `1`), mirroring
    /// [`coprime_u64`](prism_render_architecture::particle::integer_gcd_lcm::coprime_u64).
    Coprime,
}

impl GpuGcdOp {
    /// Returns the stable `u32` code the kernel `switch` dispatches on.
    #[must_use]
    pub const fn to_u32(self) -> u32 {
        match self {
            GpuGcdOp::Gcd => OP_GCD,
            GpuGcdOp::BinaryGcd => OP_BINARY_GCD,
            GpuGcdOp::Coprime => OP_COPRIME,
        }
    }
}

/// One integer-`gcd` query: a `u32` operand pair and the operation to run.
///
/// For [`GpuGcdOp::Coprime`] the result is the `0`/`1` flag; for the two `gcd`
/// operations it is the greatest common divisor in the `u32` domain. Derives
/// [`Eq`] because every field is an integer or a discrete code.
///
/// Provenance: twinned from this repository's
/// [`integer_gcd_lcm`](prism_render_architecture::particle::integer_gcd_lcm);
/// no third-party engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuGcdQuery {
    /// First operand.
    pub a: u32,
    /// Second operand.
    pub b: u32,
    /// Operation applied to `a` and `b`.
    pub op: GpuGcdOp,
}

impl GpuGcdQuery {
    /// Builds a query from an operand pair and an operation.
    #[must_use]
    pub const fn new(a: u32, b: u32, op: GpuGcdOp) -> GpuGcdQuery {
        GpuGcdQuery { a, b, op }
    }
}

/// Uniform parameters for one dispatch. `repr(C)` `std430` layout matching
/// `Params` in [`INTEGER_GCD_WGSL`]: the query count and three pad words — `16`
/// bytes, each field at the uniform offset the shader expects.
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
/// shader: the two operands, the operation code and one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// First operand.
    a: u32,
    /// Second operand.
    b: u32,
    /// Stable operation code.
    op: u32,
    /// Padding word.
    pad0: u32,
}

/// Packs one [`GpuGcdQuery`] into its `std430` [`GpuQuery`] slot, encoding the
/// operation through its stable `to_u32` code.
fn encode_query(q: &GpuGcdQuery) -> GpuQuery {
    GpuQuery {
        a: q.a,
        b: q.b,
        op: q.op.to_u32(),
        pad0: 0,
    }
}

/// A compiled, reusable integer-`gcd` dispatch pipeline.
pub struct GpuIntegerGcd {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuIntegerGcd {
    /// Compiles the integer-`gcd` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuIntegerGcd {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_integer_gcd"),
            source: ShaderSource::Wgsl(INTEGER_GCD_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_integer_gcd_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_integer_gcd_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_integer_gcd_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gcd_dispatch"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuIntegerGcd {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs every query in `queries`, returning one `u32` answer per query.
    ///
    /// Each answer reproduces the golden
    /// [`gcd_u64`](prism_render_architecture::particle::integer_gcd_lcm::gcd_u64),
    /// [`binary_gcd_u64`](prism_render_architecture::particle::integer_gcd_lcm::binary_gcd_u64)
    /// or
    /// [`coprime_u64`](prism_render_architecture::particle::integer_gcd_lcm::coprime_u64)
    /// (as a `0`/`1` flag) selected by `query.op`, evaluated in the `u32`
    /// domain. An empty `queries` slice yields an empty vector — storage
    /// buffers cannot be zero-sized, so it is handled by an early return before
    /// any dispatch.
    #[must_use]
    pub fn run(&self, ctx: &GpuContext, queries: &[GpuGcdQuery]) -> Vec<u32> {
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

        let out_bytes = (queries.len() * size_of::<u32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_integer_gcd_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_integer_gcd_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_integer_gcd_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_integer_gcd_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_integer_gcd_bind_group"),
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
            label: Some("prism_volumetric_integer_gcd_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_integer_gcd_pass"),
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
        let out = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(out.len(), queries.len());
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
