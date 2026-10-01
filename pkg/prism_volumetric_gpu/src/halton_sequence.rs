//! `wgpu` compute twin of the low-discrepancy `Halton` / `Hammersley` sequence
//! generator
//! ([`halton_sequence`](prism_render_architecture::particle::halton_sequence)).
//!
//! A low-discrepancy sequence fills a domain more evenly than independent
//! pseudo-random draws do: `TAA` sub-pixel jitter wants successive frames on
//! well-separated offsets inside one pixel, and Monte-Carlo kernels converge
//! faster when their strata are covered. The golden
//! [`halton_sequence`](prism_render_architecture::particle::halton_sequence)
//! builds those point sets out of the *radical inverse* — reflecting an index's
//! base-`b` digits about the radix point — using only integer remainder /
//! division and `f32` multiply / add (plus a single `u32` bit reversal for the
//! base-2 fast path). No transcendental function appears, so the math ports to
//! the portable core-`WGSL` subset verbatim.
//!
//! [`GpuHaltonSequence`] is the on-device twin: one thread computes the point
//! for one index. Each entry point mirrors one golden function tap for tap, so
//! a passing real-device parity test is direct evidence the ported kernels
//! accumulate the same digits in the same order the reference does, not merely
//! that the shaders compile.
//!
//! # What is twinned
//!
//! Every public golden function with a per-index shape is reproduced:
//!
//! - [`GpuHaltonSequence::radical_inverse_base2`] mirrors
//!   [`radical_inverse_base2`](prism_render_architecture::particle::halton_sequence::radical_inverse_base2):
//!   one `u32` `reverseBits` plus a multiply by `2^-32`.
//! - [`GpuHaltonSequence::radical_inverse`] mirrors
//!   [`radical_inverse`](prism_render_architecture::particle::halton_sequence::radical_inverse):
//!   the arbitrary-base digit loop with its rational accumulation and the
//!   degenerate `base < 2` guard returning `0.0`.
//! - [`GpuHaltonSequence::halton_point_2d`] and
//!   [`GpuHaltonSequence::halton_point`] mirror the canonical and explicit
//!   base-pair `Halton` points.
//! - [`GpuHaltonSequence::hammersley_point`] mirrors the `i / n` first axis and
//!   base-2 radical-inverse second axis, including the degenerate `n = 0`
//!   treated as `n = 1`.
//! - [`GpuHaltonSequence::taa_jitter`] mirrors the frame-indexed `TAA` offset
//!   (advance the `Halton` (2, 3) sequence by one, remap into `[-0.5, 0.5)`).
//! - [`GpuHaltonSequence::pixel_jitter`] mirrors the unit-square to
//!   `[-0.5, 0.5)` sub-pixel remap.
//! - [`GpuHaltonSequence::cranley_patterson`] mirrors the toroidal
//!   `Cranley-Patterson` rotation (add offset, keep the fractional part). The
//!   two-dimensional form is componentwise identical, so the scalar twin
//!   covers it.
//!
//! # Degenerate regions
//!
//! Index `0` radical-inverses to `0.0` on both sides (the digit loop makes zero
//! iterations), so the `Halton` and `Hammersley` origin points land exactly on
//! `0.0`. A `base < 2` returns `0.0` and a `Hammersley` count `n = 0` is
//! clamped to `1`, matching the reference guards with no divide by zero.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `reverseBits`,
//! `floor`, `max`, `+ - * /`, integer `%` and `/`, and unsigned index
//! comparison — with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt`, no `u64` and
//! no optional device feature. The digit loop uses a constant upper bound (the
//! `32`-digit worst case of a `u32` in base 2) and breaks when the index
//! reaches zero, so it is never unbounded. The kernels therefore run unmodified
//! on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! `radical_inverse_base2` is a bit reversal times a power of two, so it is
//! bit-exact on both sides. The arbitrary-base path is a fixed, non-reorderable
//! sequence of `f32` multiply / add in the same order as the reference, so
//! `CPU` and `GPU` evaluate the same closed form; they are not bit-exact
//! because a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts `abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! tight enough to fail a genuinely wrong port (a swapped base, a dropped
//! digit, a missing fractional wrap) yet loose enough to admit a legal fused
//! multiply-add.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::halton_sequence`；无第三方引擎源码或衍生代码。
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
/// that divides evenly on every target backend; the per-index outputs are
/// flattened to a single linear index so the dispatch stays one-dimensional.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `Halton` / `Hammersley` kernels, embedded inline so
/// the twin ships as a single source file. Each entry point mirrors one `CPU`
/// golden
/// [`halton_sequence`](prism_render_architecture::particle::halton_sequence)
/// function tap for tap; see the module documentation for the algorithm.
const HALTON_SEQUENCE_WGSL: &str = r#"
// Halton / Hammersley low-discrepancy twin: one thread computes the point for
// one index. Each entry point mirrors the CPU golden
// `particle::halton_sequence`. The core primitive is the radical inverse:
// `radical_inverse_base2` is one u32 reverseBits times 2^-32 (bit-exact), and
// `radical_inverse` walks the base-b digits least-significant first with a
// rational accumulation. Only reverseBits, floor, max, + - * /, integer % and /
// appear: no transcendental, no u64, no optional feature, portable on Metal,
// Vulkan and DX12. The digit loop uses a constant 32-iteration upper bound (the
// base-2 worst case for a u32) and breaks when the index reaches zero.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::halton_sequence；无第三方
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
// Generic u32 input slots: their meaning depends on the entry point (base, i,
// n, base_x / base_y / frame). Unused slots are bound to a 1-element dummy.
@group(0) @binding(1) var<storage, read> in_a: array<u32>;
@group(0) @binding(2) var<storage, read> in_b: array<u32>;
@group(0) @binding(3) var<storage, read> in_c: array<u32>;
// Generic f32 input slots: sample value / offset / point components.
@group(0) @binding(4) var<storage, read> in_x: array<f32>;
@group(0) @binding(5) var<storage, read> in_y: array<f32>;
// Flat output: scalar entries write one f32 per index, vec2 entries write two.
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;

// 2^-32, exact since it is a power of two. Mirrors the reference INV_2_POW_32.
const INV_2_POW_32: f32 = 1.0 / 4294967296.0;

// Upper bound on base-b digits of a u32 (the base-2 worst case); the loop
// breaks early when the running index reaches zero.
const MAX_DIGITS: u32 = 32u;

// Base-2 radical inverse: reflect the binary digits of `i` about the radix
// point via one u32 bit reversal, then scale by 2^-32. Matches the reference.
fn radical_inverse_base2_fn(i: u32) -> f32 {
    return f32(reverseBits(i)) * INV_2_POW_32;
}

// Arbitrary-base radical inverse. Walks the base-`base` digits of `i` from least
// to most significant, accumulating each at the rational weight base^-k. A
// degenerate base < 2 returns 0.0 rather than looping forever. Matches the
// reference accumulation order exactly.
fn radical_inverse_fn(base: u32, i_in: u32) -> f32 {
    if (base < 2u) {
        return 0.0;
    }
    let inv_base = 1.0 / f32(base);
    var inv_base_k = 1.0;
    var result = 0.0;
    var i = i_in;
    for (var k = 0u; k < MAX_DIGITS; k = k + 1u) {
        if (i == 0u) {
            break;
        }
        let digit = i % base;
        i = i / base;
        inv_base_k = inv_base_k * inv_base;
        result = result + f32(digit) * inv_base_k;
    }
    return result;
}

// Fractional part in [0, 1): subtract the floor. Mirrors the reference wrap01.
fn wrap01_fn(v: f32) -> f32 {
    return v - floor(v);
}

// Canonical (2, 3) Halton point: base-2 fast path on x, base-3 digit loop on y.
fn halton_point_2d_fn(i: u32) -> vec2<f32> {
    return vec2<f32>(radical_inverse_base2_fn(i), radical_inverse_fn(3u, i));
}

// Remap a unit-square sample into the [-0.5, 0.5) sub-pixel offset.
fn pixel_jitter_fn(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(p.x - 0.5, p.y - 0.5);
}

@compute @workgroup_size(64)
fn radical_inverse_base2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    dst[idx] = radical_inverse_base2_fn(in_a[idx]);
}

@compute @workgroup_size(64)
fn radical_inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // in_a carries the base, in_b carries the index.
    dst[idx] = radical_inverse_fn(in_a[idx], in_b[idx]);
}

@compute @workgroup_size(64)
fn halton_point_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let p = halton_point_2d_fn(in_a[idx]);
    dst[idx * 2u] = p.x;
    dst[idx * 2u + 1u] = p.y;
}

@compute @workgroup_size(64)
fn halton_point(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // in_a = base_x, in_b = base_y, in_c = index.
    let p = vec2<f32>(
        radical_inverse_fn(in_a[idx], in_c[idx]),
        radical_inverse_fn(in_b[idx], in_c[idx]),
    );
    dst[idx * 2u] = p.x;
    dst[idx * 2u + 1u] = p.y;
}

@compute @workgroup_size(64)
fn hammersley_point(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // in_a = index i, in_b = total count n (clamped up to 1).
    let i = in_a[idx];
    let n = max(in_b[idx], 1u);
    let x = f32(i) / f32(n);
    dst[idx * 2u] = x;
    dst[idx * 2u + 1u] = radical_inverse_base2_fn(i);
}

@compute @workgroup_size(64)
fn taa_jitter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // Advance the Halton (2, 3) sequence by one (u32 wrap on frame + 1), then
    // remap into [-0.5, 0.5), exactly as the reference taa_jitter does.
    let p = pixel_jitter_fn(halton_point_2d_fn(in_a[idx] + 1u));
    dst[idx * 2u] = p.x;
    dst[idx * 2u + 1u] = p.y;
}

@compute @workgroup_size(64)
fn pixel_jitter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let p = pixel_jitter_fn(vec2<f32>(in_x[idx], in_y[idx]));
    dst[idx * 2u] = p.x;
    dst[idx * 2u + 1u] = p.y;
}

@compute @workgroup_size(64)
fn cranley_patterson(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // in_x = sample value, in_y = per-instance offset.
    dst[idx] = wrap01_fn(in_x[idx] + in_y[idx]);
}
"#;

/// Uniform parameters for one dispatch: the element `count` plus padding to a
/// `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`HALTON_SEQUENCE_WGSL`].
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

/// The generic per-call inputs for one dispatch. Each entry point reads only
/// the slots it needs; the rest are bound to a 1-element dummy buffer. `width`
/// is the number of `f32` outputs per index (`1` for a scalar, `2` for a point).
struct DispatchInputs<'a> {
    /// Element count (number of indices processed).
    count: usize,
    /// First generic `u32` input (base / index / frame depending on entry).
    a: &'a [u32],
    /// Second generic `u32` input (index / `base_y` / count depending on entry).
    b: &'a [u32],
    /// Third generic `u32` input (index for the explicit base-pair point).
    c: &'a [u32],
    /// First generic `f32` input (sample value / point x component).
    x: &'a [f32],
    /// Second generic `f32` input (offset / point y component).
    y: &'a [f32],
    /// Number of `f32` outputs written per index.
    width: usize,
}

/// A compiled, reusable set of `Halton` / `Hammersley` sequence kernels, one
/// per golden function, sharing one bind-group layout and one `WGSL` module.
pub struct GpuHaltonSequence {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_radical_inverse_base2: ComputePipeline,
    pipeline_radical_inverse: ComputePipeline,
    pipeline_halton_point_2d: ComputePipeline,
    pipeline_halton_point: ComputePipeline,
    pipeline_hammersley_point: ComputePipeline,
    pipeline_taa_jitter: ComputePipeline,
    pipeline_pixel_jitter: ComputePipeline,
    pipeline_cranley_patterson: ComputePipeline,
}

impl GpuHaltonSequence {
    /// Compiles the `Halton` / `Hammersley` kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHaltonSequence {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_halton_sequence"),
            source: ShaderSource::Wgsl(HALTON_SEQUENCE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_halton_sequence_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: true }),
                buffer_entry(4, BufferBindingType::Storage { read_only: true }),
                buffer_entry(5, BufferBindingType::Storage { read_only: true }),
                buffer_entry(6, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_halton_sequence_pipeline_layout"),
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
        let pipeline_radical_inverse_base2 = make(
            "radical_inverse_base2",
            "prism_volumetric_halton_sequence_radical_inverse_base2_pipeline",
        );
        let pipeline_radical_inverse = make(
            "radical_inverse",
            "prism_volumetric_halton_sequence_radical_inverse_pipeline",
        );
        let pipeline_halton_point_2d = make(
            "halton_point_2d",
            "prism_volumetric_halton_sequence_halton_point_2d_pipeline",
        );
        let pipeline_halton_point = make(
            "halton_point",
            "prism_volumetric_halton_sequence_halton_point_pipeline",
        );
        let pipeline_hammersley_point = make(
            "hammersley_point",
            "prism_volumetric_halton_sequence_hammersley_point_pipeline",
        );
        let pipeline_taa_jitter = make(
            "taa_jitter",
            "prism_volumetric_halton_sequence_taa_jitter_pipeline",
        );
        let pipeline_pixel_jitter = make(
            "pixel_jitter",
            "prism_volumetric_halton_sequence_pixel_jitter_pipeline",
        );
        let pipeline_cranley_patterson = make(
            "cranley_patterson",
            "prism_volumetric_halton_sequence_cranley_patterson_pipeline",
        );
        GpuHaltonSequence {
            module,
            layout,
            pipeline_radical_inverse_base2,
            pipeline_radical_inverse,
            pipeline_halton_point_2d,
            pipeline_halton_point,
            pipeline_hammersley_point,
            pipeline_taa_jitter,
            pipeline_pixel_jitter,
            pipeline_cranley_patterson,
        }
    }

    /// Base-2 radical inverse (van der Corput) of each index, mirroring
    /// [`radical_inverse_base2`](prism_render_architecture::particle::halton_sequence::radical_inverse_base2).
    ///
    /// Returns one `f32` per index, in order. An empty input returns an empty
    /// vector with no dispatch issued (a storage buffer cannot be zero-sized).
    #[must_use]
    pub fn radical_inverse_base2(&self, ctx: &GpuContext, indices: &[u32]) -> Vec<f32> {
        self.dispatch_scalar(
            ctx,
            &self.pipeline_radical_inverse_base2,
            DispatchInputs {
                count: indices.len(),
                a: indices,
                b: &[],
                c: &[],
                x: &[],
                y: &[],
                width: 1,
            },
        )
    }

    /// Arbitrary-base radical inverse of each paired `(base, index)`, mirroring
    /// [`radical_inverse`](prism_render_architecture::particle::halton_sequence::radical_inverse),
    /// including the degenerate `base < 2` guard returning `0.0`.
    ///
    /// Returns one `f32` per index, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `bases` and `indices` differ in length.
    #[must_use]
    pub fn radical_inverse(&self, ctx: &GpuContext, bases: &[u32], indices: &[u32]) -> Vec<f32> {
        assert_eq!(
            bases.len(),
            indices.len(),
            "bases and indices must be the same length"
        );
        self.dispatch_scalar(
            ctx,
            &self.pipeline_radical_inverse,
            DispatchInputs {
                count: indices.len(),
                a: bases,
                b: indices,
                c: &[],
                x: &[],
                y: &[],
                width: 1,
            },
        )
    }

    /// Canonical (2, 3) `Halton` point for each index, mirroring
    /// [`halton_point_2d`](prism_render_architecture::particle::halton_sequence::halton_point_2d).
    ///
    /// Returns one `[f32; 2]` per index, in order. An empty input returns an
    /// empty vector with no dispatch issued.
    #[must_use]
    pub fn halton_point_2d(&self, ctx: &GpuContext, indices: &[u32]) -> Vec<[f32; 2]> {
        self.dispatch_vec2(
            ctx,
            &self.pipeline_halton_point_2d,
            DispatchInputs {
                count: indices.len(),
                a: indices,
                b: &[],
                c: &[],
                x: &[],
                y: &[],
                width: 2,
            },
        )
    }

    /// `Halton` point using an explicit base pair for each `(base_x, base_y,
    /// index)` triple, mirroring
    /// [`halton_point`](prism_render_architecture::particle::halton_sequence::halton_point).
    ///
    /// Returns one `[f32; 2]` per index, in order. An empty input returns an
    /// empty vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `base_x`, `base_y` and `indices` differ in length.
    #[must_use]
    pub fn halton_point(
        &self,
        ctx: &GpuContext,
        base_x: &[u32],
        base_y: &[u32],
        indices: &[u32],
    ) -> Vec<[f32; 2]> {
        assert_eq!(
            base_x.len(),
            indices.len(),
            "base_x and indices must be the same length"
        );
        assert_eq!(
            base_y.len(),
            indices.len(),
            "base_y and indices must be the same length"
        );
        self.dispatch_vec2(
            ctx,
            &self.pipeline_halton_point,
            DispatchInputs {
                count: indices.len(),
                a: base_x,
                b: base_y,
                c: indices,
                x: &[],
                y: &[],
                width: 2,
            },
        )
    }

    /// `Hammersley` point for each `(index, count)` pair, mirroring
    /// [`hammersley_point`](prism_render_architecture::particle::halton_sequence::hammersley_point),
    /// including the degenerate `count = 0` treated as `count = 1`.
    ///
    /// Returns one `[f32; 2]` per index, in order. An empty input returns an
    /// empty vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `indices` and `counts` differ in length.
    #[must_use]
    pub fn hammersley_point(
        &self,
        ctx: &GpuContext,
        indices: &[u32],
        counts: &[u32],
    ) -> Vec<[f32; 2]> {
        assert_eq!(
            indices.len(),
            counts.len(),
            "indices and counts must be the same length"
        );
        self.dispatch_vec2(
            ctx,
            &self.pipeline_hammersley_point,
            DispatchInputs {
                count: indices.len(),
                a: indices,
                b: counts,
                c: &[],
                x: &[],
                y: &[],
                width: 2,
            },
        )
    }

    /// Sub-pixel `TAA` jitter for each frame index, mirroring
    /// [`taa_jitter`](prism_render_architecture::particle::halton_sequence::taa_jitter).
    ///
    /// Returns one `[f32; 2]` per frame, in order. An empty input returns an
    /// empty vector with no dispatch issued.
    #[must_use]
    pub fn taa_jitter(&self, ctx: &GpuContext, frames: &[u32]) -> Vec<[f32; 2]> {
        self.dispatch_vec2(
            ctx,
            &self.pipeline_taa_jitter,
            DispatchInputs {
                count: frames.len(),
                a: frames,
                b: &[],
                c: &[],
                x: &[],
                y: &[],
                width: 2,
            },
        )
    }

    /// Remaps each unit-square point into the `[-0.5, 0.5)` sub-pixel offset,
    /// mirroring
    /// [`pixel_jitter`](prism_render_architecture::particle::halton_sequence::pixel_jitter).
    ///
    /// Returns one `[f32; 2]` per point, in order. An empty input returns an
    /// empty vector with no dispatch issued.
    #[must_use]
    pub fn pixel_jitter(&self, ctx: &GpuContext, points: &[[f32; 2]]) -> Vec<[f32; 2]> {
        let xs: Vec<f32> = points.iter().map(|p| p[0]).collect();
        let ys: Vec<f32> = points.iter().map(|p| p[1]).collect();
        self.dispatch_vec2(
            ctx,
            &self.pipeline_pixel_jitter,
            DispatchInputs {
                count: points.len(),
                a: &[],
                b: &[],
                c: &[],
                x: &xs,
                y: &ys,
                width: 2,
            },
        )
    }

    /// `Cranley-Patterson` rotation of each `(value, offset)` pair, mirroring
    /// [`cranley_patterson`](prism_render_architecture::particle::halton_sequence::cranley_patterson).
    /// The two-dimensional form is componentwise identical, so this scalar twin
    /// covers it.
    ///
    /// Returns one `f32` per pair, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `values` and `offsets` differ in length.
    #[must_use]
    pub fn cranley_patterson(&self, ctx: &GpuContext, values: &[f32], offsets: &[f32]) -> Vec<f32> {
        assert_eq!(
            values.len(),
            offsets.len(),
            "values and offsets must be the same length"
        );
        self.dispatch_scalar(
            ctx,
            &self.pipeline_cranley_patterson,
            DispatchInputs {
                count: values.len(),
                a: &[],
                b: &[],
                c: &[],
                x: values,
                y: offsets,
                width: 1,
            },
        )
    }

    /// Issues one dispatch and reshapes the flat `f32` readback into `[f32; 2]`
    /// points. Empty input short-circuits to an empty vector.
    fn dispatch_vec2(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        inputs: DispatchInputs<'_>,
    ) -> Vec<[f32; 2]> {
        let flat = self.dispatch(ctx, pipeline, inputs);
        flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    }

    /// Issues one dispatch returning one `f32` per index. Empty input
    /// short-circuits to an empty vector.
    fn dispatch_scalar(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        inputs: DispatchInputs<'_>,
    ) -> Vec<f32> {
        self.dispatch(ctx, pipeline, inputs)
    }

    /// Issues one `1-D` dispatch of `pipeline` over `inputs`, reading the flat
    /// `f32` outputs back. Empty inputs short-circuit without a dispatch because
    /// a storage buffer cannot be zero-sized.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        pipeline: &ComputePipeline,
        inputs: DispatchInputs<'_>,
    ) -> Vec<f32> {
        let count = inputs.count;
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
            label: Some("prism_volumetric_halton_sequence_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let in_a = u32_buffer(device, "prism_volumetric_halton_sequence_in_a", inputs.a);
        let in_b = u32_buffer(device, "prism_volumetric_halton_sequence_in_b", inputs.b);
        let in_c = u32_buffer(device, "prism_volumetric_halton_sequence_in_c", inputs.c);
        let in_x = f32_buffer(device, "prism_volumetric_halton_sequence_in_x", inputs.x);
        let in_y = f32_buffer(device, "prism_volumetric_halton_sequence_in_y", inputs.y);

        let out_len = count * inputs.width;
        let out_bytes = (out_len * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_halton_sequence_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_halton_sequence_bind_group"),
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
                    resource: in_c.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: in_x.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: in_y.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_halton_sequence_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_halton_sequence_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_halton_sequence_pass"),
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
        let result = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();
        result
    }
}

/// Builds a `u32` storage buffer from `data`, substituting a 1-element dummy for
/// an empty slot so the generic layout's binding is always satisfied.
fn u32_buffer(device: &wgpu::Device, label: &str, data: &[u32]) -> wgpu::Buffer {
    let dummy = [0u32];
    let src = if data.is_empty() { &dummy[..] } else { data };
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(src),
        usage: BufferUsages::STORAGE,
    })
}

/// Builds an `f32` storage buffer from `data`, substituting a 1-element dummy
/// for an empty slot so the generic layout's binding is always satisfied.
fn f32_buffer(device: &wgpu::Device, label: &str, data: &[f32]) -> wgpu::Buffer {
    let dummy = [0.0f32];
    let src = if data.is_empty() { &dummy[..] } else { data };
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(src),
        usage: BufferUsages::STORAGE,
    })
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
