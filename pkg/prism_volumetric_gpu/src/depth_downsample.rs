//! `wgpu` compute twin of the generic `2x2` depth / data down-sample
//! `mip`-chain builder
//! ([`depth_downsample`](prism_render_architecture::particle::depth_downsample),
//! design §16-§21).
//!
//! Many screen-space effects want a *pyramid* of an input buffer: a chain of
//! ever-coarser `mip` levels where each texel summarizes a `2x2` block of the
//! finer level. A screen-space-reflection ray-march reads coarse depth `mip`s
//! to skip empty space; a screen-space-ambient-occlusion kernel reads a coarse
//! depth pyramid to bound its sampling radius; a min/max depth pyramid feeds
//! tile classification. All of them fold a base image down `2x2` at a time
//! under a chosen reduction operator until a single `1x1` texel remains.
//!
//! The `CPU` golden
//! [`depth_downsample`](prism_render_architecture::particle::depth_downsample)
//! owns that primitive — the conservative `div_ceil` size rule, the single
//! level [`downsample_2x2`](prism_render_architecture::particle::depth_downsample::downsample_2x2),
//! and the full-chain
//! [`DepthMipChain::build`](prism_render_architecture::particle::depth_downsample::DepthMipChain::build);
//! [`GpuDepthDownsample`] is the on-device twin. One compute dispatch per level
//! transition reduces the finer level into the next coarser one with **one
//! thread per destination texel**, each gathering the up-to-four source texels
//! of its clamped `2x2` footprint; the dispatches are chained finest-to-coarsest
//! and the per-level outputs are packed back-to-back row-major, reproducing the
//! exact byte layout
//! [`DepthMipChain`](prism_render_architecture::particle::depth_downsample::DepthMipChain)
//! produces. A passing real-device parity test is therefore direct evidence the
//! ported kernel builds the same pyramid the reference does, not merely that its
//! shader compiles.
//!
//! # Conservative odd-dimension handling
//!
//! Each level extent is `div_ceil(dim, 2)` of the previous, so an odd dimension
//! rounds *up* and keeps its lone boundary row or column instead of dropping it.
//! That boundary destination texel reduces a *partial* block (one or two source
//! texels), so no source texel is ever discarded and an extreme value on an odd
//! edge survives into the coarser level unchanged. The kernel reproduces that
//! rule by clamping every `2x2` sample to the source bounds (`sx < src_w`,
//! `sy < src_h`) and reducing only the gathered samples, matching the
//! reference's partial-block gather exactly.
//!
//! # Operator coverage
//!
//! All four reference operators are twinned lane for lane: [`ReduceOp::Min`]
//! (nearest-depth pyramid), [`ReduceOp::Max`] (farthest-depth pyramid),
//! [`ReduceOp::Average`] (arithmetic mean of the gathered samples) and
//! [`ReduceOp::CheckerboardMinMax`], whose destination texel takes the block
//! minimum on even `(dx + dy)` and the maximum on odd — the same
//! `(dx + dy) & 1 == 1` parity the reference uses.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `+ - * /` and unsigned integer arithmetic / bit `and` — with no `sin`,
//! `cos`, `exp`, `log`, `pow` or optional device feature, so it runs unmodified
//! on Metal, Vulkan and DX12. There is no transcendental call at all (not even
//! `sqrt`): the reduction is comparisons, one running sum and a single guarded
//! divide by the gathered sample count, which is at least `1` for every
//! destination texel (each covers at least its own top-left source texel).
//!
//! # Correctness model
//!
//! `Min`, `Max` and `CheckerboardMinMax` are pure comparisons and exact `f32`
//! copies, so they are bit-identical on `CPU` and `GPU`: the running `min` /
//! `max` folds the gathered samples in the same row-major order the reference
//! does. `Average` sums the same samples in the same order and divides by the
//! same exact integer count, so it too matches to the last bit on hardware that
//! does not contract the sum; the parity test allows an `abs`/`rel` `1e-5`
//! slack to admit a legal fused multiply-add on the running sum while still
//! failing a genuinely wrong port (a dropped boundary sample, a wrong operator,
//! a mis-sized level). Chaining the dispatches finest-to-coarsest performs the
//! identical per-level operation the reference's
//! [`DepthMipChain::build`](prism_render_architecture::particle::depth_downsample::DepthMipChain::build)
//! chains, so the whole packed pyramid matches level for level.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `2x2` reduction `mip`-pyramid build (min/max/average
//! depth pyramids feeding `SSR`/`SSAO`/tile classification) plus `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::depth_downsample::ReduceOp;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` `2x2` reduction kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden
/// [`downsample_2x2`](prism_render_architecture::particle::depth_downsample::downsample_2x2)
/// texel for texel; see the module documentation for the algorithm.
///
/// The `Params` struct is a `32`-byte `std430` block of eight `u32` words
/// matching the host [`Params`]: the source and destination extents, the
/// operator code, the destination texel count and two pad words.
const DEPTH_DOWNSAMPLE_WGSL: &str = r#"
// 2x2 reduction twin: one thread per destination texel gathers the up-to-four
// source texels of its clamped 2x2 footprint and folds them under the selected
// operator (0 = Min, 1 = Max, 2 = Average, 3 = CheckerboardMinMax). It mirrors
// the CPU golden `particle::depth_downsample::downsample_2x2`, uses only the
// portable core-WGSL subset (min/max and + - * / plus unsigned integer / bit
// ops, no transcendental), and takes no optional feature, so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: standard 2x2 reduction mip pyramid; no Unreal Engine source or
// derived code.

struct Params {
    // Source (finer) level extents in texels.
    src_w: u32,
    src_h: u32,
    // Destination (coarser) level extents, each `div_ceil(src, 2)`.
    dst_w: u32,
    dst_h: u32,
    // Reduction operator code: 0 Min, 1 Max, 2 Average, 3 CheckerboardMinMax.
    op: u32,
    // Number of destination texels (`dst_w * dst_h`), one thread each.
    dst_count: u32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.dst_count) {
        return;
    }
    // Row-major destination coordinate from the linear thread index.
    let dx = idx % params.dst_w;
    let dy = idx / params.dst_w;
    let x0 = dx * 2u;
    let y0 = dy * 2u;

    // Running reduction over the gathered samples, folded in the same row-major
    // (oy outer, ox inner) order the reference gathers them.
    var vmin = 0.0;
    var vmax = 0.0;
    var vsum = 0.0;
    var count = 0u;
    for (var oy = 0u; oy < 2u; oy = oy + 1u) {
        let sy = y0 + oy;
        if (sy >= params.src_h) {
            continue;
        }
        for (var ox = 0u; ox < 2u; ox = ox + 1u) {
            let sx = x0 + ox;
            if (sx >= params.src_w) {
                continue;
            }
            let v = src[sy * params.src_w + sx];
            if (count == 0u) {
                vmin = v;
                vmax = v;
            } else {
                vmin = min(vmin, v);
                vmax = max(vmax, v);
            }
            vsum = vsum + v;
            count = count + 1u;
        }
    }

    var out = vmin;
    if (params.op == 1u) {
        out = vmax;
    } else if (params.op == 2u) {
        // Every destination texel covers at least its top-left source texel, so
        // `count` is at least 1 and the divide never hits zero.
        out = vsum / f32(count);
    } else if (params.op == 3u) {
        // Checkerboard: block max on odd `(dx + dy)`, block min on even.
        if (((dx + dy) & 1u) == 1u) {
            out = vmax;
        } else {
            out = vmin;
        }
    }
    dst[idx] = out;
}
"#;

/// One depth-down-sample build request: the twin of a single
/// [`DepthMipChain::build`](prism_render_architecture::particle::depth_downsample::DepthMipChain::build)
/// call.
///
/// `base` is the `mip` 0 image in row-major order; it must hold exactly
/// `width * height` texels. Derives only [`PartialEq`] (no `Eq`/`Hash`) because
/// it borrows `f32` depths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthDownsampleQuery<'base> {
    /// Base-level width in texels.
    pub width: usize,
    /// Base-level height in texels.
    pub height: usize,
    /// Row-major base image (`mip` 0), exactly `width * height` texels.
    pub base: &'base [f32],
    /// Reduction operator applied to each `2x2` block.
    pub op: ReduceOp,
}

/// Uniform parameters for one `2x2` reduction dispatch. `repr(C)` `std430`
/// layout matching `Params` in [`DEPTH_DOWNSAMPLE_WGSL`]: the source and
/// destination extents, the operator code, the destination texel count and two
/// pad words — `32` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    op: u32,
    dst_count: u32,
    pad0: u32,
    pad1: u32,
}

/// Maps a [`ReduceOp`] to the `u32` operator code the kernel branches on.
fn op_code(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Min => 0,
        ReduceOp::Max => 1,
        ReduceOp::Average => 2,
        ReduceOp::CheckerboardMinMax => 3,
    }
}

/// A compiled, reusable `2x2` depth-down-sample pipeline.
pub struct GpuDepthDownsample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuDepthDownsample {
    /// Compiles the `2x2` reduction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDepthDownsample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_depth_downsample"),
            source: ShaderSource::Wgsl(DEPTH_DOWNSAMPLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_depth_downsample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_depth_downsample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_depth_downsample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("reduce"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDepthDownsample {
            module,
            layout,
            pipeline,
        }
    }

    /// Builds the full `2x2` reduction `mip` chain for `query`, returning the
    /// whole chain packed back-to-back row-major (`mip` 0 first, then every
    /// coarser level down to `1x1`).
    ///
    /// The returned buffer equals the concatenation of
    /// [`DepthMipChain::mip`](prism_render_architecture::particle::depth_downsample::DepthMipChain::mip)
    /// over every level of
    /// [`DepthMipChain::build`](prism_render_architecture::particle::depth_downsample::DepthMipChain::build)`(query.width, query.height, query.base, query.op)`
    /// to within the tolerance documented on this module. Returns [`None`] on a
    /// degenerate base — a zero `width` or `height`, a `width * height`
    /// overflow, or a `base` length that does not equal `width * height` —
    /// exactly as the reference `build` returns [`None`], so a degenerate input
    /// never panics.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &DepthDownsampleQuery<'_>) -> Option<Vec<f32>> {
        let DepthDownsampleQuery {
            width,
            height,
            base,
            op,
        } = *query;
        if width == 0 || height == 0 {
            return None;
        }
        let base_texels = width.checked_mul(height)?;
        if base.len() != base_texels {
            return None;
        }

        // `mip` 0 is the base copied verbatim, matching the reference, which
        // packs it before any reduction. Every coarser level is produced by one
        // GPU dispatch reading the previous level.
        let mut packed: Vec<f32> = Vec::with_capacity(base_texels);
        packed.extend_from_slice(base);

        let mut current: Vec<f32> = base.to_vec();
        let (mut w, mut h) = (width, height);
        while w > 1 || h > 1 {
            let next = self.reduce_level(ctx, &current, w, h, op);
            packed.extend_from_slice(&next);
            current = next;
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        Some(packed)
    }

    /// Reduces one `(src_w, src_h)` level into its conservative
    /// `(div_ceil(src_w, 2), div_ceil(src_h, 2))` coarser level under `op` with
    /// a single dispatch, returning the coarse texels in row-major order.
    ///
    /// `src` must hold exactly `src_w * src_h` texels; the caller only invokes
    /// this when `src_w > 1 || src_h > 1`, so both the source buffer and the
    /// coarser destination are non-empty and bindable.
    fn reduce_level(
        &self,
        ctx: &GpuContext,
        src: &[f32],
        src_w: usize,
        src_h: usize,
        op: ReduceOp,
    ) -> Vec<f32> {
        let device = ctx.device();
        // The coarser extent is the conservative `div_ceil` halving, matching
        // the reference level-size rule.
        let dst_w = src_w.div_ceil(2);
        let dst_h = src_h.div_ceil(2);
        let dst_count = dst_w * dst_h;

        let gpu_params = Params {
            src_w: src_w as u32,
            src_h: src_h as u32,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            op: op_code(op),
            dst_count: dst_count as u32,
            pad0: 0,
            pad1: 0,
        };

        let out_bytes = (dst_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_depth_downsample_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let src_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_depth_downsample_src"),
            contents: bytemuck::cast_slice(src),
            usage: BufferUsages::STORAGE,
        });
        let dst_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_depth_downsample_dst"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dst_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_depth_downsample_dst_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_depth_downsample_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dst_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_depth_downsample_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_depth_downsample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per destination texel, in workgroups of 64.
            let groups = (dst_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&dst_buf, 0, &dst_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        dst_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = dst_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let level = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        dst_stage.unmap();
        debug_assert_eq!(level.len(), dst_count);

        level
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
