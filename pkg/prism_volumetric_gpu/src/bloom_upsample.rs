//! `wgpu` compute twin of the `bloom` progressive-upsample composite
//! ([`bloom_upsample`](prism_render_architecture::particle::bloom_upsample),
//! design sections 16-21, the wide-halo second half of the bloom pyramid).
//!
//! Once a bright-pass prefilter has isolated the emissive pixels and a
//! downsample chain has built a pyramid of ever-coarser `mip` levels, the glow
//! is synthesized by walking that pyramid back *up*: each coarse level is
//! upsampled with a small `tent` kernel and *added* onto the next-finer level
//! with a per-`mip` weight. Summing progressively wider blurs is what turns a
//! handful of cheap blurs into the smooth, wide-domain halo the eye reads as a
//! surface emitting more light than the display can show.
//!
//! The `CPU` golden
//! [`composite_chain`](prism_render_architecture::particle::bloom_upsample::composite_chain)
//! owns that math; [`GpuBloomUpsample`] is the on-device twin that runs one
//! thread per *output* `texel` per upsample step and reproduces the same image
//! the batch form produces. A passing real-device parity test is therefore
//! direct evidence the ported kernel samples the same `tent` taps, clamps the
//! same edges, blends the same scatter and adds the same per-`mip` weight the
//! reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The whole upsample-and-composite contract is reproduced:
//!
//! * the 3x3 separable `[1, 2, 1]` `tent`
//!   ([`tent_filter_9tap`](prism_render_architecture::particle::bloom_upsample::tent_filter_9tap),
//!   outer product summing to `16`, normalized so a constant level is
//!   reproduced unchanged), tap for tap, in the reference accumulation order
//!   (`y` outer, `x` inner) with the same clamp-to-edge addressing;
//! * the `radius` scatter blend
//!   ([`scatter`] -> `smoothstep(0, 1, radius)`) between a crisp nearest sample
//!   and the full `tent` spread, implemented with the identical denominator
//!   guard and Hermite polynomial the reference uses;
//! * the weighted additive composite
//!   ([`upsample_add`](prism_render_architecture::particle::bloom_upsample::upsample_add)):
//!   each finer level is `base + weight * lerp(sharp, wide, spread)`, in that
//!   evaluation order; and
//! * the coarse-to-fine chain
//!   ([`composite_chain`](prism_render_architecture::particle::bloom_upsample::composite_chain)):
//!   starting from the coarsest level, every step upsamples the running
//!   accumulator onto the next-finer level with that level's
//!   [`MipWeights`](prism_render_architecture::particle::bloom_upsample::MipWeights)
//!   entry. The host records each level's extent and chains the storage buffers
//!   so each pass reads the previous pass's output.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `clamp`,
//! `abs`, unsigned integer index arithmetic and `+ - * /` on scalars and
//! vectors — with no `sin`, `cos`, `exp`, `log`, `pow` or `sqrt` and no optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. There
//! is no transcendental call on this path: the `tent` taps are multiplies and
//! adds divided by the compile-time constant `16`, and the scatter blend is a
//! clamped cubic.
//!
//! # Correctness model
//!
//! Each output `texel` is a fixed, non-reorderable sequence of weighted `tent`
//! taps, one scatter lerp and one weighted add, so `CPU` and `GPU` evaluate the
//! same closed form in the same order. They are not bit-exact: a `GPU` may fuse
//! a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, and that perturbation
//! compounds across the pyramid. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a swapped `tent` tap, a dropped scatter blend, a wrong
//! normalizer, a missing edge clamp, a mis-ordered chain) yet loose enough to
//! admit legal fused multiply-add contraction summed over the pyramid.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard progressive dual-filter `bloom` upsample (`Unreal`,
//! `Frostbite`, Call-of-Duty-style wide-halo composite) plus `wgpu` compute
//! dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::bloom_upsample::{MipImage, MipWeights};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// used across this crate's kernels; the output `texels` of one upsample step
/// are flattened to a single linear index so the dispatch stays one-dimensional.
const WORKGROUP_SIZE: u32 = 64;

/// Scalar components per pixel: the linear `HDR` `RGB` triple the reference
/// [`MipImage`] stores. The device buffers pack each level as a flat
/// `width * height * 3` `f32` array so a storage buffer needs no `vec3`
/// alignment padding.
const CHANNELS: usize = 3;

/// The portable core-`WGSL` upsample-add kernel, embedded inline so the twin
/// ships as a single source file. The entry point `upsample_add` mirrors one
/// step of the `CPU` golden
/// [`upsample_add`](prism_render_architecture::particle::bloom_upsample::upsample_add)
/// tap for tap; see the module documentation for the algorithm.
const BLOOM_UPSAMPLE_WGSL: &str = r#"
// Bloom upsample-add twin: one thread per output texel runs one composite step,
// computing `base + weight * lerp(sharp, wide, scatter(radius))` where `sharp`
// is the nearest low-level texel and `wide` is its 3x3 [1,2,1] tent. It mirrors
// the CPU golden `particle::bloom_upsample::upsample_add`, uses only the
// portable core-WGSL subset (min/clamp/abs and + - * / on scalars and vectors
// plus unsigned index math) and takes no optional feature, so it runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard progressive dual-filter bloom upsample; no third-party
// engine source or derived code.

struct Params {
    // Low (coarse) source level extents in texels.
    low_w: u32,
    low_h: u32,
    // High (fine) base level extents in texels; also the output extents, with
    // one thread per output texel.
    high_w: u32,
    high_h: u32,
    // Per-mip additive weight for this step.
    weight: f32,
    // Glow-width scatter control, blended via smoothstep toward the full tent.
    radius: f32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> low: array<f32>;
@group(0) @binding(2) var<storage, read> high: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

// Normalization divisor for the 3x3 tent: the [1,2,1] outer product sums to 16.
const TENT_NORM: f32 = 16.0;
// Denominators smaller than this collapse to a defined fallback so the
// smoothstep normalization never divides by zero, matching the reference guard.
const MIN_DENOM: f32 = 1.0e-6;

// Reads coarse texel `(x, y)` as an RGB triple from the flat 3-per-pixel buffer.
// Coordinates are pre-clamped, so the index is always in range.
fn load_low(x: u32, y: u32) -> vec3<f32> {
    let base = (y * params.low_w + x) * 3u;
    return vec3<f32>(low[base], low[base + 1u], low[base + 2u]);
}

// Reads fine base texel `(x, y)` as an RGB triple from the flat buffer.
fn load_high(x: u32, y: u32) -> vec3<f32> {
    let base = (y * params.high_w + x) * 3u;
    return vec3<f32>(high[base], high[base + 1u], high[base + 2u]);
}

// Writes an RGB triple to output texel `(x, y)` in the flat buffer.
fn store_dst(x: u32, y: u32, c: vec3<f32>) {
    let base = (y * params.high_w + x) * 3u;
    dst[base] = c.x;
    dst[base + 1u] = c.y;
    dst[base + 2u] = c.z;
}

// Clamp-to-edge tent tap, mirroring the reference `clamp_tap`: tap 0 -> the
// texel left/above (center - 1, saturating at 0), tap 2 -> the texel
// right/below (center + 1, clamped to extent - 1), tap 1 -> the center itself
// (clamped to extent - 1). `extent` is always positive (empty levels are guarded
// on the host).
fn clamp_tap(center: u32, tap: u32, extent: u32) -> u32 {
    let hi = extent - 1u;
    if (tap == 0u) {
        if (center == 0u) {
            return 0u;
        }
        return center - 1u;
    }
    if (tap == 2u) {
        return min(center + 1u, hi);
    }
    return min(center, hi);
}

// Samples the coarse level with the 3x3 separable [1,2,1] tent centered on
// `(cx, cy)`, normalized by 16. The nine taps are accumulated y-outer, x-inner
// with weight `wx * wy` applied before the add, exactly as the reference
// `tent_filter_9tap` does, so the fused-multiply-add structure matches.
fn tent9(cx: u32, cy: u32) -> vec3<f32> {
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    // TENT_1D = [1, 2, 1]: the center tap carries weight 2, the edges weight 1.
    for (var ty = 0u; ty < 3u; ty = ty + 1u) {
        var wy = 1.0;
        if (ty == 1u) {
            wy = 2.0;
        }
        let sy = clamp_tap(cy, ty, params.low_h);
        for (var tx = 0u; tx < 3u; tx = tx + 1u) {
            var wx = 1.0;
            if (tx == 1u) {
                wx = 2.0;
            }
            let sx = clamp_tap(cx, tx, params.low_w);
            let w = wx * wy;
            acc = acc + load_low(sx, sy) * w;
        }
    }
    return acc / TENT_NORM;
}

// The [0, 1] scatter blend for the glow `radius`: smoothstep(0, 1, radius) with
// the reference's denominator guard. `0` keeps the upsample crisp (nearest
// sample), `1` uses the full tent, and values between blend the two.
fn scatter(radius: f32) -> f32 {
    let edge0 = 0.0;
    let edge1 = 1.0;
    let denom = edge1 - edge0;
    var safe = denom;
    if (abs(denom) < MIN_DENOM) {
        safe = MIN_DENOM;
    }
    let t = clamp((radius - edge0) / safe, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

@compute @workgroup_size(64)
fn upsample_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.high_w * params.high_h;
    if (idx >= total) {
        return;
    }
    let ox = idx % params.high_w;
    let oy = idx / params.high_w;
    // Halve the output coordinate to find the matching coarse texel, clamped to
    // the last column/row (reference: `(x / 2).min(low.width - 1)`).
    let lx = min(ox / 2u, params.low_w - 1u);
    let ly = min(oy / 2u, params.low_h - 1u);
    let spread = scatter(params.radius);
    let sharp = load_low(lx, ly);
    let wide = tent9(lx, ly);
    // lerp3(sharp, wide, spread) = sharp + (wide - sharp) * spread.
    let up = sharp + (wide - sharp) * spread;
    let base = load_high(ox, oy);
    // Additive composite: a larger weight can only brighten, never darken.
    store_dst(ox, oy, base + up * params.weight);
}
"#;

/// One `bloom` upsample-composite request: the `mip` pyramid, its per-level
/// weight chain and the glow `radius`.
///
/// Mirrors the `(mips, weights, radius)` triple the reference
/// [`composite_chain`](prism_render_architecture::particle::bloom_upsample::composite_chain)
/// consumes. `mips` is ordered finest-first (index `0`) to coarsest-last.
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because the levels hold `f32`
/// pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct BloomUpsampleQuery {
    /// The `mip` pyramid, finest level first and coarsest last.
    pub mips: Vec<MipImage>,
    /// The per-`mip` additive weight chain, finest level first.
    pub weights: MipWeights,
    /// The glow-width scatter control blended toward the full 3x3 `tent`.
    pub radius: f32,
}

/// Uniform parameters for one upsample-add step dispatch. `repr(C)` layout
/// matching `Params` in [`BLOOM_UPSAMPLE_WGSL`]: the coarse and fine extents,
/// the per-`mip` `weight`, the glow `radius` and two pad words — `32` bytes,
/// each field at the `std140` uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Low (coarse) source width in `texels`.
    low_w: u32,
    /// Low (coarse) source height in `texels`.
    low_h: u32,
    /// High (fine) base and output width in `texels`.
    high_w: u32,
    /// High (fine) base and output height in `texels`.
    high_h: u32,
    /// Per-`mip` additive weight for this step.
    weight: f32,
    /// Glow-width scatter control.
    radius: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

impl GpuParams {
    /// Packs one step's coarse/fine extents, additive `weight` and `radius`.
    fn new(
        low_w: usize,
        low_h: usize,
        high_w: usize,
        high_h: usize,
        weight: f32,
        radius: f32,
    ) -> GpuParams {
        GpuParams {
            low_w: low_w as u32,
            low_h: low_h as u32,
            high_w: high_w as u32,
            high_h: high_h as u32,
            weight,
            radius,
            pad0: 0,
            pad1: 0,
        }
    }
}

/// One step in the recorded coarse-to-fine chain: the low (accumulator), high
/// (base `mip`) and destination buffer indices into the host-owned buffer list,
/// plus the packed uniform for the step.
struct PassSpec {
    /// Index of the coarse accumulator source buffer in the host buffer list.
    low: usize,
    /// Index of the fine base-`mip` source buffer in the host buffer list.
    high: usize,
    /// Index of the destination buffer in the host buffer list.
    dst: usize,
    /// The packed uniform parameters for this step.
    params: GpuParams,
}

/// A compiled, reusable `bloom` upsample-composite pipeline.
pub struct GpuBloomUpsample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBloomUpsample {
    /// Compiles the `bloom` upsample-add kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuBloomUpsample {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bloom_upsample"),
            source: ShaderSource::Wgsl(BLOOM_UPSAMPLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bloom_upsample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bloom_upsample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bloom_upsample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("upsample_add"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBloomUpsample {
            module,
            layout,
            pipeline,
        }
    }

    /// Composites `query.mips` coarsest-to-finest, returning the glow at the
    /// finest resolution.
    ///
    /// The result equals
    /// [`composite_chain(&query.mips, &query.weights, query.radius)`](prism_render_architecture::particle::bloom_upsample::composite_chain)
    /// to within the tolerance documented on this module. An empty pyramid
    /// returns [`None`] exactly as the reference does. A single-level pyramid
    /// returns a clone of that level (the chain's base case), with no dispatch
    /// issued. For a well-formed multi-level pyramid one dispatch is issued per
    /// composite step, chaining the storage buffers so each step reads the
    /// previous step's accumulator.
    ///
    /// Every level is required to be non-empty: a real `bloom` pyramid never
    /// holds an empty level, and a zero-`texel` level cannot back a storage
    /// buffer, so a degenerate pyramid of two or more levels returns [`None`]
    /// rather than panicking.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &BloomUpsampleQuery) -> Option<MipImage> {
        let mips = &query.mips;
        let n = mips.len();
        // Empty pyramid: the reference `composite_chain` returns `None`.
        if n == 0 {
            return None;
        }
        // Single level: the chain's base case clones the lone (coarsest ==
        // finest) level unchanged; no upsample step runs.
        if n == 1 {
            return Some(mips[0].clone());
        }
        // A zero-texel level cannot back a storage buffer; such a degenerate
        // pyramid is outside the twin's domain.
        if mips.iter().any(MipImage::is_empty) {
            return None;
        }
        let device = ctx.device();

        // Upload every level as a read-only source buffer: indices `0..n`, with
        // the coarsest (index `n - 1`) serving as the initial accumulator.
        let mut buffers: Vec<wgpu::Buffer> = Vec::with_capacity(2 * n - 1);
        for mip in mips {
            buffers.push(device.create_buffer_init(&BufferInitDescriptor {
                label: Some("prism_volumetric_bloom_upsample_level"),
                contents: bytemuck::cast_slice(&flatten_pixels(mip)),
                usage: BufferUsages::STORAGE,
            }));
        }

        // Walk coarsest-to-finest. For the step producing level `level` the
        // accumulator (`low`) is at level `level + 1`'s resolution and the base
        // (`high`) is `mips[level]`; the output is a fresh buffer at that
        // level's resolution and becomes the next step's accumulator.
        let mut low_idx = n - 1;
        let (mut low_w, mut low_h) = (mips[n - 1].width, mips[n - 1].height);
        let mut specs: Vec<PassSpec> = Vec::with_capacity(n - 1);
        for level in (0..n - 1).rev() {
            let (hw, hh) = (mips[level].width, mips[level].height);
            buffers.push(empty_image_buffer(device, hw, hh));
            let dst = buffers.len() - 1;
            // `unwrap_or(0.0)` mirrors the reference when the weight chain is
            // shorter than the pyramid.
            let weight = query.weights.weight(level).unwrap_or(0.0);
            specs.push(PassSpec {
                low: low_idx,
                high: level,
                dst,
                params: GpuParams::new(low_w, low_h, hw, hh, weight, query.radius),
            });
            low_idx = dst;
            low_w = hw;
            low_h = hh;
        }
        let final_idx = low_idx;

        // Build the per-step uniform buffers and bind groups up front so the
        // encoder can record every step back to back.
        let mut param_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(specs.len());
        let mut bind_groups: Vec<wgpu::BindGroup> = Vec::with_capacity(specs.len());
        for spec in &specs {
            let params_buf = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("prism_volumetric_bloom_upsample_params"),
                contents: bytemuck::bytes_of(&spec.params),
                usage: BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_volumetric_bloom_upsample_bind_group"),
                layout: &self.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: buffers[spec.low].as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: buffers[spec.high].as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: buffers[spec.dst].as_entire_binding(),
                    },
                ],
            });
            param_buffers.push(params_buf);
            bind_groups.push(bind_group);
        }

        let final_texels = mips[0].width.saturating_mul(mips[0].height);
        let out_bytes = (final_texels * CHANNELS * size_of::<f32>()) as u64;
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bloom_upsample_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bloom_upsample_encoder"),
        });
        for (spec, bind_group) in specs.iter().zip(bind_groups.iter()) {
            let total = spec.params.high_w * spec.params.high_h;
            let groups = total.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bloom_upsample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            // One thread per output `texel`, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buffers[final_idx], 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut pixels: Vec<[f32; 3]> = Vec::with_capacity(final_texels);
        for chunk in flat.chunks_exact(CHANNELS) {
            pixels.push([chunk[0], chunk[1], chunk[2]]);
        }
        debug_assert_eq!(pixels.len(), final_texels);

        Some(MipImage::new(mips[0].width, mips[0].height, pixels))
    }
}

/// Flattens a [`MipImage`] into the row-major, 3-per-pixel `f32` layout the
/// device storage buffers expect.
fn flatten_pixels(image: &MipImage) -> Vec<f32> {
    let mut data = Vec::with_capacity(image.pixels.len() * CHANNELS);
    for pixel in &image.pixels {
        data.push(pixel[0]);
        data.push(pixel[1]);
        data.push(pixel[2]);
    }
    data
}

/// Allocates an uninitialized storage buffer sized for a `width * height` `RGB`
/// level, usable as a step output and readable as the next step's accumulator.
fn empty_image_buffer(device: &wgpu::Device, width: usize, height: usize) -> wgpu::Buffer {
    let texels = width.saturating_mul(height);
    device.create_buffer(&BufferDescriptor {
        label: Some("prism_volumetric_bloom_upsample_output"),
        size: (texels * CHANNELS * size_of::<f32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
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
