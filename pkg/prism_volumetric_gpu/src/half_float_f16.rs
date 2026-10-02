//! `wgpu` compute twin of the `IEEE` 754 binary16 (half precision) <-> `f32`
//! bit-exact converters
//! ([`half_float_f16`](prism_render_architecture::particle::half_float_f16),
//! particle design §25, §29).
//!
//! The `CPU` golden
//! [`half_float_f16`](prism_render_architecture::particle::half_float_f16) owns
//! two pure-integer converters:
//! [`f32_to_f16_bits`](prism_render_architecture::particle::half_float_f16::f32_to_f16_bits)
//! rounds an `f32` to the binary16 bit pattern with round-to-nearest-even
//! (`RNE`), and
//! [`f16_bits_to_f32`](prism_render_architecture::particle::half_float_f16::f16_bits_to_f32)
//! widens a binary16 pattern back to an `f32`. Both operate solely on raw
//! integer bit patterns: no `f32` arithmetic or comparison takes place.
//! [`GpuHalfFloatF16`] is the on-device twin: one thread converts one element
//! in both directions, so a passing real-device parity test is direct evidence
//! the ported kernel reproduces the exact same integer rounding, subnormal
//! normalization, `inf`/`NaN` handling and signed-zero logic the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! Each [`HalfFloatQuery`] packs a forward input `value` (the `f32` to narrow)
//! and a reverse input `half_bits` (the binary16 pattern to widen). The kernel
//! returns a [`HalfFloatResult`] holding `forward_bits`, the binary16 pattern
//! of
//! [`f32_to_f16_bits`](prism_render_architecture::particle::half_float_f16::f32_to_f16_bits)
//! applied to `value`, and `reverse_value`, the `f32` of
//! [`f16_bits_to_f32`](prism_render_architecture::particle::half_float_f16::f16_bits_to_f32)
//! applied to `half_bits`. Both directions cover the full edge-case set:
//! `RNE` ties-to-even rounding, gradual underflow to subnormals, flush to a
//! signed zero, overflow to infinity, `NaN` payloads with the quiet bit forced,
//! signed zeros and the 127 <-> 15 exponent-bias change.
//!
//! # Why raw bits, not an `f32`, cross the device boundary
//!
//! `WGSL` has no `u16`, so each binary16 result is carried in the low 16 bits
//! of a `u32` with the high bits cleared. More importantly, both the forward
//! input and the reverse output cross the boundary as raw `u32` bit patterns,
//! never as a device `f32`: a `GPU` may flush an `f32` subnormal to zero or
//! canonicalize a `NaN` payload the moment a float is materialized in a
//! register, so keeping every value in the integer domain is what makes the
//! twin bit-exact across the whole subnormal range and every `NaN` payload. The
//! kernel performs no `f32` arithmetic whatsoever; it manipulates `u32` words
//! with shifts, masks and integer compares, exactly as the reference does. The
//! host rebuilds the reverse `f32` with [`f32::from_bits`] once the exact `u32`
//! pattern is back on the `CPU`.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — unsigned integer
//! shift / and / or / xor, integer compares and a bounded normalization loop —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `sqrt` or optional device
//! feature, so it runs unmodified on Metal, Vulkan and DX12. The subnormal
//! `leading_zeros` of the reference is reproduced by a bounded left-shift loop
//! (at most ten steps for a ten-bit mantissa), avoiding any non-core built-in.
//!
//! # Correctness model
//!
//! Every operation is integer work and `WGSL` unsigned integers shift, mask and
//! wrap exactly like Rust's `>>`, `&`, `|` and `^`, so the device result is
//! bit-identical to the reference in both directions. There are no continuous
//! outputs and therefore no rounding slack: the parity test asserts exact `==`
//! on both the forward binary16 word and the reverse `f32` bit pattern, with no
//! tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: textbook `IEEE` 754 binary16 <-> `f32` bit manipulation plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

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

/// Threads per workgroup; one thread converts one element in both directions.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` half-float kernel, embedded inline so the twin
/// ships as a single compiled artifact with no external `.wgsl` resource.
const HALF_FLOAT_F16_WGSL: &str = r#"
// Half-float twin kernel: a per-element port of `particle::half_float_f16`. It
// narrows an f32 bit pattern to binary16 with round-to-nearest-even and widens
// a binary16 pattern back to an f32 bit pattern, using only unsigned integer
// shifts, masks, compares and a bounded normalization loop. No f32 arithmetic
// is performed and no non-core built-in is used, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: textbook IEEE 754 binary16 <-> f32 bit manipulation; no Unreal
// Engine source or derived code.

// One conversion element. 8-byte std430 stride of 2 scalar words matching the
// host `GpuElem`: the forward f32 input (carried as its raw bit pattern) and
// the reverse binary16 input (carried in the low 16 bits of a u32).
struct Elem {
    value_bits: u32,
    half_bits: u32,
}

// One conversion result. 8-byte std430 stride of 2 scalar words matching the
// host `GpuResult`: the forward binary16 pattern (low 16 bits, high bits clear)
// and the reverse f32 bit pattern.
struct Conv {
    forward_bits: u32,
    reverse_bits: u32,
}

// Dispatch parameters. 16-byte std430/uniform block: the valid element count
// plus three pad words, matching the host `Params`.
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> elems: array<Elem>;
@group(0) @binding(2) var<storage, read_write> results: array<Conv>;

// Narrows an f32 bit pattern to the binary16 bit pattern, rounding to nearest
// with ties to even, mirroring the reference `f32_to_f16_bits`. The result
// occupies the low 16 bits; the high bits are clear.
fn f32_to_f16_bits(bits: u32) -> u32 {
    // Sign bit relocated from f32 bit 31 to binary16 bit 15.
    let sign = (bits >> 16u) & 0x00008000u;
    // Raw f32 exponent field (bias 127), as a signed integer for re-biasing.
    let raw_exp = i32((bits >> 23u) & 0xffu);
    // 23-bit f32 significand (no implicit leading one).
    let mantissa = bits & 0x007fffffu;

    // NaN or infinity: f32 exponent field is all ones.
    if (raw_exp == 0xff) {
        if (mantissa != 0u) {
            // NaN: carry the top significand bits down and force the quiet bit
            // so the result stays a quiet NaN with a non-zero mantissa.
            let quiet = (mantissa >> 13u) | 0x00000200u;
            return sign | 0x00007c00u | quiet;
        }
        return sign | 0x00007c00u;
    }

    // Re-bias the exponent for binary16 (127 -> 15).
    let exp = raw_exp - 112;

    if (exp >= 0x1f) {
        // Overflow: finite magnitude too large for binary16.
        return sign | 0x00007c00u;
    }

    if (exp <= 0) {
        if (exp < -10) {
            // Far below the smallest subnormal: flush to a signed zero.
            return sign;
        }
        // Subnormal result: restore the implicit leading one, then shift the
        // 24-bit significand right into the 10-bit field with RNE rounding.
        let m = mantissa | 0x00800000u;
        let shift = u32(14 - exp);
        let truncated = m >> shift;
        let round_bit = (m >> (shift - 1u)) & 1u;
        let sticky = (m & ((1u << (shift - 1u)) - 1u)) != 0u;
        var result = truncated;
        if (round_bit == 1u && (sticky || (truncated & 1u) == 1u)) {
            result = result + 1u;
        }
        return sign | result;
    }

    // Normalized result: drop the low 13 significand bits with RNE. The least
    // significant retained bit decides ties.
    let truncated = mantissa >> 13u;
    let round_bit = (mantissa >> 12u) & 1u;
    let sticky = (mantissa & 0x00000fffu) != 0u;
    var result = (u32(exp) << 10u) | truncated;
    if (round_bit == 1u && (sticky || (truncated & 1u) == 1u)) {
        // A carry out of the mantissa correctly increments the exponent field.
        result = result + 1u;
    }
    if (result >= 0x00007c00u) {
        // Rounding pushed the magnitude up to infinity.
        return sign | 0x00007c00u;
    }
    return sign | result;
}

// Widens a binary16 bit pattern to the corresponding f32 bit pattern, mirroring
// the reference `f16_bits_to_f32`. Subnormals are normalized with a bounded
// left-shift loop that reproduces the reference integer `leading_zeros` count.
fn f16_bits_to_f32_bits(h: u32) -> u32 {
    // Sign bit relocated from binary16 bit 15 to f32 bit 31.
    let sign = (h & 0x00008000u) << 16u;
    let exp = (h >> 10u) & 0x1fu;
    let mant = h & 0x000003ffu;

    if (exp == 0x1fu) {
        if (mant == 0u) {
            // Infinity.
            return sign | 0x7f800000u;
        }
        // NaN: binary16 mantissa bit 9 maps to f32 mantissa bit 22, the quiet
        // bit, so the quiet status is preserved.
        return sign | 0x7f800000u | (mant << 13u);
    }

    if (exp == 0u) {
        if (mant == 0u) {
            // Signed zero.
            return sign;
        }
        // Subnormal: left-shift the mantissa until the implicit leading one
        // reaches bit 10, counting the shifts. For a mantissa in [1, 0x3ff]
        // this takes between one and ten steps; the shift count `e` relates to
        // the reference `leading_zeros` by `f32_exp = 113 - e`.
        var m = mant;
        var e = 0u;
        loop {
            if ((m & 0x00000400u) != 0u) {
                break;
            }
            m = m << 1u;
            e = e + 1u;
        }
        let f32_exp = 113u - e;
        let f32_mant = (m << 13u) & 0x007fffffu;
        return sign | (f32_exp << 23u) | f32_mant;
    }

    // Normalized: re-bias the exponent (15 -> 127) and widen the mantissa.
    let f32_exp = exp + 112u;
    let f32_mant = mant << 13u;
    return sign | (f32_exp << 23u) | f32_mant;
}

@compute @workgroup_size(64)
fn half_convert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let e = elems[idx];
    var out: Conv;
    out.forward_bits = f32_to_f16_bits(e.value_bits);
    out.reverse_bits = f16_bits_to_f32_bits(e.half_bits & 0x0000ffffu);
    results[idx] = out;
}
"#;

/// One conversion element: a forward input `value` to narrow to binary16 and a
/// reverse input `half_bits` to widen back to an `f32`.
///
/// The forward `value` is uploaded as its raw [`f32::to_bits`] pattern so the
/// device never materializes it as a float, preserving every subnormal and
/// `NaN` payload exactly.
///
/// Provenance: textbook `IEEE` 754 binary16 <-> `f32` conversion; no Unreal
/// Engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfFloatQuery {
    /// The `f32` value to narrow to binary16 with round-to-nearest-even.
    pub value: f32,
    /// The binary16 bit pattern to widen back to an `f32`.
    pub half_bits: u16,
}

/// One conversion result: the forward binary16 pattern and the reverse `f32`.
///
/// `forward_bits` equals
/// [`f32_to_f16_bits`](prism_render_architecture::particle::half_float_f16::f32_to_f16_bits)
/// of the query `value`; `reverse_value` equals
/// [`f16_bits_to_f32`](prism_render_architecture::particle::half_float_f16::f16_bits_to_f32)
/// of the query `half_bits`, each bit-identical to the reference.
///
/// Provenance: textbook `IEEE` 754 binary16 <-> `f32` conversion; no Unreal
/// Engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HalfFloatResult {
    /// The binary16 pattern of the narrowed forward `value`.
    pub forward_bits: u16,
    /// The `f32` of the widened reverse `half_bits`, rebuilt on the host with
    /// [`f32::from_bits`] from the exact device bit pattern.
    pub reverse_value: f32,
}

/// Uniform parameters for one half-float dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`HALF_FLOAT_F16_WGSL`]: the valid element count plus
/// three pad words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of valid elements.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One element as uploaded. `8`-byte `std430` stride of `2` scalar words,
/// matching `Elem` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuElem {
    /// Forward `f32` input carried as its raw bit pattern.
    value_bits: u32,
    /// Reverse binary16 input carried in the low 16 bits.
    half_bits: u32,
}

/// One result as read back. `8`-byte `std430` stride of `2` scalar words,
/// matching `Conv` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// Forward binary16 pattern in the low 16 bits.
    forward_bits: u32,
    /// Reverse `f32` bit pattern.
    reverse_bits: u32,
}

/// A compiled, reusable half-float conversion pipeline.
pub struct GpuHalfFloatF16 {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHalfFloatF16 {
    /// Compiles the half-float kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHalfFloatF16 {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_half_float_f16"),
            source: ShaderSource::Wgsl(HALF_FLOAT_F16_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_half_float_f16_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_half_float_f16_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_half_float_f16_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("half_convert"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHalfFloatF16 {
            module,
            layout,
            pipeline,
        }
    }

    /// Converts every element in `queries`, returning one [`HalfFloatResult`]
    /// per query in input order.
    ///
    /// For query `i`, `forward_bits` equals
    /// [`f32_to_f16_bits(value)`](prism_render_architecture::particle::half_float_f16::f32_to_f16_bits)
    /// and `reverse_value` equals
    /// [`f16_bits_to_f32(half_bits)`](prism_render_architecture::particle::half_float_f16::f16_bits_to_f32),
    /// each bit-identical to the reference. An empty `queries` slice yields an
    /// empty result — storage buffers cannot be zero-sized, so it is handled by
    /// an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[HalfFloatQuery]) -> Vec<HalfFloatResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_elems: Vec<GpuElem> = queries
            .iter()
            .map(|q| GpuElem {
                value_bits: q.value.to_bits(),
                half_bits: u32::from(q.half_bits),
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuResult>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_half_float_f16_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let elems_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_half_float_f16_elems"),
            contents: bytemuck::cast_slice(&gpu_elems),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_half_float_f16_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_half_float_f16_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_half_float_f16_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: elems_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_half_float_f16_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_half_float_f16_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per element, in workgroups of `WORKGROUP_SIZE`.
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

        gpu_results
            .into_iter()
            .map(|r| HalfFloatResult {
                forward_bits: r.forward_bits as u16,
                reverse_value: f32::from_bits(r.reverse_bits),
            })
            .collect()
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
