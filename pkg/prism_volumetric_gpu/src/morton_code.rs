//! `wgpu` compute twin of the `Morton` (`Z-order`) space-filling-curve bit
//! interleave
//! ([`morton_code`](prism_render_architecture::particle::morton_code)).
//!
//! A `Morton` code interleaves the bits of each grid coordinate so points that
//! are close in space stay close along a single sortable key. The `CPU` golden
//! [`morton_code`](prism_render_architecture::particle::morton_code) owns that
//! math as a pure `u16`/`u32` *magic-bits* cascade — a short run of
//! shift-or-and steps that spreads every input bit into its interleaved slot in
//! `O(log bits)` operations, and the inverse that gathers them back.
//! [`GpuMortonCode`] is the on-device twin: one thread per element, every
//! operation a `u32` shift, mask or `or` with no multiply, divide or floating
//! point. `CPU` and `GPU` therefore compute the identical bit pattern, so a
//! passing real-device parity test is direct evidence the ported kernels spread,
//! gather and pack the bits exactly as the reference does, not merely that the
//! shaders compile.
//!
//! # What is twinned
//!
//! Four per-element kernels mirror the reference encode/decode pairs, each
//! sharing one bind-group layout and the four magic-bits helpers (`part1by1`,
//! `compact1by1`, `part1by2`, `compact1by2`). Every kernel reads one `u32` from
//! the `src` slot and writes one `u32` to the `dst` slot; coordinate tuples are
//! packed into a single word so the dispatch stays a clean one-in, one-out map:
//!
//! - [`GpuMortonCode::encode_2d`] mirrors
//!   [`morton_encode_2d`](prism_render_architecture::particle::morton_code::morton_encode_2d):
//!   the `src` word carries `x` in its low `16` bits and `y` in its high `16`
//!   bits; the kernel spreads each with `part1by1` and packs `x` onto the even
//!   bits and `y` onto the odd bits of the output code.
//! - [`GpuMortonCode::decode_2d`] mirrors
//!   [`morton_decode_2d`](prism_render_architecture::particle::morton_code::morton_decode_2d):
//!   the `src` word is a 2D code; the kernel gathers the even and odd bits with
//!   `compact1by1` and packs the recovered `x` (low `16` bits) and `y` (high
//!   `16` bits) into the output word.
//! - [`GpuMortonCode::encode_3d`] mirrors
//!   [`morton_encode_3d`](prism_render_architecture::particle::morton_code::morton_encode_3d):
//!   the `src` word carries `x`, `y`, `z` in three `10`-bit fields (bits `0..10`,
//!   `10..20`, `20..30`); the kernel spreads each with `part1by2` and packs them
//!   onto bits `0`, `1`, `2` of each triple in the low `30` bits of the output.
//! - [`GpuMortonCode::decode_3d`] mirrors
//!   [`morton_decode_3d`](prism_render_architecture::particle::morton_code::morton_decode_3d):
//!   the `src` word is a 3D code; the kernel gathers every third bit with
//!   `compact1by2` and packs the three recovered `10`-bit coordinates into the
//!   output word.
//!
//! The high-bit masking of the reference is reproduced: `part1by1` keeps only
//! the low `16` input bits and `part1by2` only the low `10`, so passing an
//! out-of-range coordinate drops the same bits on both sides. The 2D corner
//! range [`morton_aabb_range_2d`](prism_render_architecture::particle::morton_code::morton_aabb_range_2d)
//! is deliberately out of scope here: it is a `min`/`max` corner normalization
//! composed from two [`GpuMortonCode::encode_2d`] calls, not a distinct bit
//! algorithm, so twinning the four interleave kernels covers all of the
//! bit-level math.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — the bit operators
//! `>> << & |`, unsigned addition in the loop index and unsigned index
//! comparison — with no transcendental call, no `countOneBits`/`firstTrailingBit`
//! intrinsic, no optional device feature and no `u64`. Each magic-bits cascade
//! is a fixed, unrolled sequence of shift-or-and steps, so the kernels run
//! unmodified on `Metal`, `Vulkan` and `DX12`.
//!
//! # Correctness model
//!
//! Every operation is pure `u32` bit algebra with no rounding anywhere on the
//! path, so `CPU` and `GPU` compute identical bit patterns. The parity test
//! asserts an exact `==` on every `u32` output, with no tolerance: any mismatch
//! is a genuine port bug (a wrong mask, a flipped shift direction, a swapped
//! axis). The interleave is reversible, so encode-then-decode is the identity on
//! both sides and the degenerate inputs (zero, a single bit, a fully set
//! coordinate) carry no special case.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::morton_code`；无第三方引擎源码或衍生代码。
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

/// The `Morton` bit-interleave kernels, mirroring the `CPU` golden
/// [`morton_code`](prism_render_architecture::particle::morton_code) magic-bits
/// cascade step for step. One source file hosts four entry points sharing one
/// bind-group layout and the four spread/gather helpers.
const MORTON_CODE_WGSL: &str = r#"
// Morton (Z-order) interleave twin: one thread per element. Four entry points
// mirror the CPU golden `particle::morton_code` encode/decode pairs. Each reads
// one u32 from `src` and writes one u32 to `dst`; coordinate tuples are packed
// into a single word (2D: x in bits 0..16, y in 16..32; 3D: x/y/z in three
// 10-bit fields). Pure u32 shifts / masks / or: no transcendental, no intrinsic,
// no u64, portable on Metal, Vulkan and DX12. Every magic-bits cascade is a
// fixed, unrolled sequence, so no loop bounds depend on the data.
//
// Provenance: 孪生自本仓 prism_render_architecture::particle::morton_code；无第三方
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
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

// Spread the 16 low bits of x so each occupies an even bit with one empty odd
// bit between them (interval 1). Matches the reference `part1by1`.
fn part1by1(x: u32) -> u32 {
    var v = x & 0x0000FFFFu;
    v = (v | (v << 8u)) & 0x00FF00FFu;
    v = (v | (v << 4u)) & 0x0F0F0F0Fu;
    v = (v | (v << 2u)) & 0x33333333u;
    v = (v | (v << 1u)) & 0x55555555u;
    return v;
}

// Inverse of part1by1: gather the even bits back into a dense 16-bit value.
// Matches the reference `compact1by1`.
fn compact1by1(x: u32) -> u32 {
    var v = x & 0x55555555u;
    v = (v | (v >> 1u)) & 0x33333333u;
    v = (v | (v >> 2u)) & 0x0F0F0F0Fu;
    v = (v | (v >> 4u)) & 0x00FF00FFu;
    v = (v | (v >> 8u)) & 0x0000FFFFu;
    return v;
}

// Spread the 10 low bits of x so each occupies every third bit with two empty
// bits between them (interval 2). Matches the reference `part1by2`.
fn part1by2(x: u32) -> u32 {
    var v = x & 0x000003FFu;
    v = (v | (v << 16u)) & 0xFF0000FFu;
    v = (v | (v << 8u)) & 0x0300F00Fu;
    v = (v | (v << 4u)) & 0x030C30C3u;
    v = (v | (v << 2u)) & 0x09249249u;
    return v;
}

// Inverse of part1by2: gather every third bit back into a dense 10-bit value.
// Matches the reference `compact1by2`.
fn compact1by2(x: u32) -> u32 {
    var v = x & 0x09249249u;
    v = (v | (v >> 2u)) & 0x030C30C3u;
    v = (v | (v >> 4u)) & 0x0300F00Fu;
    v = (v | (v >> 8u)) & 0xFF0000FFu;
    v = (v | (v >> 16u)) & 0x000003FFu;
    return v;
}

@compute @workgroup_size(64)
fn encode_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // src packs x in bits 0..16 and y in bits 16..32; x -> even, y -> odd bits.
    let packed = src[idx];
    let x = packed & 0x0000FFFFu;
    let y = packed >> 16u;
    dst[idx] = part1by1(x) | (part1by1(y) << 1u);
}

@compute @workgroup_size(64)
fn decode_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // src is a 2D code; recover x from the even bits, y from the odd bits.
    let code = src[idx];
    let x = compact1by1(code);
    let y = compact1by1(code >> 1u);
    dst[idx] = x | (y << 16u);
}

@compute @workgroup_size(64)
fn encode_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // src packs x/y/z in three 10-bit fields; x -> bit 0, y -> 1, z -> 2.
    let packed = src[idx];
    let x = packed & 0x000003FFu;
    let y = (packed >> 10u) & 0x000003FFu;
    let z = (packed >> 20u) & 0x000003FFu;
    dst[idx] = part1by2(x) | (part1by2(y) << 1u) | (part1by2(z) << 2u);
}

@compute @workgroup_size(64)
fn decode_3d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    // src is a 3D code; recover each 10-bit axis from every third bit.
    let code = src[idx];
    let x = compact1by2(code);
    let y = compact1by2(code >> 1u);
    let z = compact1by2(code >> 2u);
    dst[idx] = x | (y << 10u) | (z << 20u);
}
"#;

/// Uniform parameters for one dispatch: the element `count` plus padding to a
/// `16`-byte, `std140`-aligned uniform struct matching `Params` in
/// [`MORTON_CODE_WGSL`].
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

/// A compiled, reusable set of `Morton` bit-interleave kernels (2D encode/decode
/// and 3D encode/decode).
pub struct GpuMortonCode {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_encode_2d: ComputePipeline,
    pipeline_decode_2d: ComputePipeline,
    pipeline_encode_3d: ComputePipeline,
    pipeline_decode_3d: ComputePipeline,
}

impl GpuMortonCode {
    /// Compiles the four `Morton` interleave kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuMortonCode {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_morton_code"),
            source: ShaderSource::Wgsl(MORTON_CODE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_morton_code_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_morton_code_pipeline_layout"),
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
        let pipeline_encode_2d = make(
            "encode_2d",
            "prism_volumetric_morton_code_encode_2d_pipeline",
        );
        let pipeline_decode_2d = make(
            "decode_2d",
            "prism_volumetric_morton_code_decode_2d_pipeline",
        );
        let pipeline_encode_3d = make(
            "encode_3d",
            "prism_volumetric_morton_code_encode_3d_pipeline",
        );
        let pipeline_decode_3d = make(
            "decode_3d",
            "prism_volumetric_morton_code_decode_3d_pipeline",
        );
        GpuMortonCode {
            module,
            layout,
            pipeline_encode_2d,
            pipeline_decode_2d,
            pipeline_encode_3d,
            pipeline_decode_3d,
        }
    }

    /// Interleaves each paired `(x, y)` 16-bit coordinate into a 32-bit 2D
    /// `Morton` code, with `x` on the even bits and `y` on the odd bits,
    /// mirroring
    /// [`morton_encode_2d`](prism_render_architecture::particle::morton_code::morton_encode_2d).
    ///
    /// Returns one code per input, in order. An empty input returns an empty
    /// vector with no dispatch issued (a storage buffer cannot be zero-sized).
    ///
    /// # Panics
    ///
    /// Panics if `xs` and `ys` differ in length.
    #[must_use]
    pub fn encode_2d(&self, ctx: &GpuContext, xs: &[u16], ys: &[u16]) -> Vec<u32> {
        assert_eq!(xs.len(), ys.len(), "xs and ys must be the same length");
        let packed: Vec<u32> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| u32::from(x) | (u32::from(y) << 16))
            .collect();
        self.dispatch(ctx, &self.pipeline_encode_2d, &packed)
    }

    /// Recovers each `(x, y)` 16-bit coordinate pair from a 2D `Morton` code,
    /// mirroring
    /// [`morton_decode_2d`](prism_render_architecture::particle::morton_code::morton_decode_2d).
    ///
    /// Returns one pair per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    #[must_use]
    pub fn decode_2d(&self, ctx: &GpuContext, codes: &[u32]) -> Vec<(u16, u16)> {
        let packed = self.dispatch(ctx, &self.pipeline_decode_2d, codes);
        packed
            .iter()
            .map(|&w| {
                let x = u16::try_from(w & 0x0000FFFF).unwrap_or(0);
                let y = u16::try_from(w >> 16).unwrap_or(0);
                (x, y)
            })
            .collect()
    }

    /// Interleaves each paired `(x, y, z)` 10-bit coordinate into the low 30
    /// bits of a 32-bit 3D `Morton` code (`x` on bit 0, `y` on bit 1, `z` on bit
    /// 2 of each triple), mirroring
    /// [`morton_encode_3d`](prism_render_architecture::particle::morton_code::morton_encode_3d).
    /// Bits above bit 9 of any coordinate are ignored, exactly as the reference
    /// drops them.
    ///
    /// Returns one code per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    ///
    /// # Panics
    ///
    /// Panics if `xs`, `ys` and `zs` do not all share the same length.
    #[must_use]
    pub fn encode_3d(&self, ctx: &GpuContext, xs: &[u32], ys: &[u32], zs: &[u32]) -> Vec<u32> {
        assert_eq!(xs.len(), ys.len(), "xs and ys must be the same length");
        assert_eq!(xs.len(), zs.len(), "xs and zs must be the same length");
        let packed: Vec<u32> = xs
            .iter()
            .zip(ys.iter())
            .zip(zs.iter())
            .map(|((&x, &y), &z)| (x & 0x3FF) | ((y & 0x3FF) << 10) | ((z & 0x3FF) << 20))
            .collect();
        self.dispatch(ctx, &self.pipeline_encode_3d, &packed)
    }

    /// Recovers each `(x, y, z)` 10-bit coordinate triple from a 3D `Morton`
    /// code, mirroring
    /// [`morton_decode_3d`](prism_render_architecture::particle::morton_code::morton_decode_3d).
    ///
    /// Returns one triple per input, in order. An empty input returns an empty
    /// vector with no dispatch issued.
    #[must_use]
    pub fn decode_3d(&self, ctx: &GpuContext, codes: &[u32]) -> Vec<(u32, u32, u32)> {
        let packed = self.dispatch(ctx, &self.pipeline_decode_3d, codes);
        packed
            .iter()
            .map(|&w| (w & 0x3FF, (w >> 10) & 0x3FF, (w >> 20) & 0x3FF))
            .collect()
    }

    /// Issues one `1-D` dispatch of `pipeline` over the `input` words, reading
    /// the `u32` outputs back. Empty inputs short-circuit without a dispatch
    /// because a storage buffer cannot be zero-sized.
    fn dispatch(&self, ctx: &GpuContext, pipeline: &ComputePipeline, input: &[u32]) -> Vec<u32> {
        let count = input.len();
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
            label: Some("prism_volumetric_morton_code_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let in_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_morton_code_in"),
            contents: bytemuck::cast_slice(input),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = size_of_val(input) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_morton_code_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_morton_code_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: in_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_morton_code_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_morton_code_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_morton_code_pass"),
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
