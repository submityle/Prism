//! `wgpu` compute twin of the dual-`Kawase` blur down/up filter chain
//! ([`kawase_dual_blur`](prism_render_architecture::particle::kawase_dual_blur),
//! design sections 16-21, "双 `Kawase` 模糊").
//!
//! A dual-`Kawase` blur synthesizes a wide, gaussian-like blur out of a handful
//! of cheap bilinear taps: it walks an image *down* a resolution pyramid with a
//! five-tap "center plus four diagonals" filter, then walks it back *up* with an
//! eight-tap `tent`, each pass sampling at a fixed sub-`texel` diagonal offset.
//! Hardware bilinear filtering does most of the averaging, so a small number of
//! taps produces a blur whose spatial support doubles with every down pass. The
//! scheme is Marius `Bjorge`'s "Bandwidth-Efficient Rendering" (`SIGGRAPH`
//! 2015).
//!
//! The `CPU` golden
//! [`kawase_dual_blur`](prism_render_architecture::particle::kawase_dual_blur)
//! owns that math; [`GpuKawaseBlur`] is the on-device twin that runs one thread
//! per *output* `texel` per pass and reproduces the same image the batch form
//! [`dual_blur`](prism_render_architecture::particle::kawase_dual_blur::KawaseImage::dual_blur)
//! produces. A passing real-device parity test is therefore direct evidence the
//! ported kernels sample the same offsets, clamp the same edges and weight the
//! same taps the reference does, not merely that the shaders compile.
//!
//! # What is twinned
//!
//! Both halves of the chain are reproduced exactly: (1) the five-tap
//! [`downsample`](prism_render_architecture::particle::kawase_dual_blur::KawaseImage::downsample)
//! kernel (center weighted by four plus four diagonal taps at `+/- offset`,
//! normalized by eight) and (2) the eight-tap
//! [`upsample`](prism_render_architecture::particle::kawase_dual_blur::KawaseImage::upsample)
//! kernel (four axis-aligned edge taps at `+/- 2 * offset` plus four
//! double-weighted diagonal taps at `+/- offset`, normalized by twelve). The
//! hand-rolled bilinear sampler is reproduced tap for tap: the same
//! `(i + 0.5) / extent` `texel`-center convention, the same `floor`-based
//! integer split, the same clamp-to-edge addressing and the same
//! `a + (b - a) * t` lerp order the reference uses, so the device sampler reads
//! the identical neighborhood. The full
//! [`dual_blur`](prism_render_architecture::particle::kawase_dual_blur::KawaseImage::dual_blur)
//! chain records the extent of every level before each down pass and retraces
//! those recorded extents on the way up (a `div_ceil` halving is not reversible
//! by another `div_ceil`), so the host issues one dispatch per recorded pass and
//! chains the storage buffers so each pass reads the previous pass's output.
//!
//! # Portability
//!
//! The kernels use only the portable core-`WGSL` subset — `min`, `max`,
//! `floor`, `+ - * /` and unsigned integer index arithmetic — with no `sin`,
//! `cos`, `exp`, `log`, `pow`, `sqrt` or optional device feature, so they run
//! unmodified on `Metal`, `Vulkan` and `DX12`. There is no transcendental call
//! at all on this path: the taps are multiplies, adds and two divides by the
//! compile-time constant normalizers whose divisors (`8` and `12`) are never
//! zero.
//!
//! # Correctness model
//!
//! Each output `texel` is a fixed, non-reorderable sequence of bilinear lerps,
//! weighted adds and one normalizing multiply, so `CPU` and `GPU` evaluate the
//! same closed form in the same order. They are not bit-exact: a `GPU` may fuse
//! a multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place, and that perturbation
//! compounds across the down/up pyramid. The parity test therefore asserts a
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a
//! genuinely wrong port (a swapped tap sign, a dropped diagonal, a wrong
//! normalizer, a missing edge clamp) yet loose enough to admit legal fused
//! multiply-add contraction summed over the pyramid.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard dual-`Kawase` blur (`Bjorge`, "Bandwidth-Efficient
//! Rendering", `SIGGRAPH` 2015) plus `wgpu` compute dispatch; no third-party
//! engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::kawase_dual_blur::{KawaseBlurParams, KawaseImage};
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
/// used across this crate's kernels; the output `texels` of one pass are
/// flattened to a single linear index so the dispatch stays one-dimensional.
const WORKGROUP_SIZE: u32 = 64;

/// Scalar components per pixel: the linear `HDR` `RGB` triple the reference
/// [`KawaseImage`] stores. The device buffers pack the image as a flat
/// `width * height * 3` `f32` array so a storage buffer needs no `vec3`
/// alignment padding.
const CHANNELS: usize = 3;

/// The portable core-`WGSL` dual-`Kawase` kernels, embedded inline so the twin
/// ships as a single source file. The two entry points `downsample` and
/// `upsample` mirror the `CPU` golden
/// [`KawaseImage::downsample`](prism_render_architecture::particle::kawase_dual_blur::KawaseImage::downsample)
/// and
/// [`KawaseImage::upsample`](prism_render_architecture::particle::kawase_dual_blur::KawaseImage::upsample)
/// tap for tap; see the module documentation for the algorithm.
const KAWASE_BLUR_WGSL: &str = r#"
// Dual-Kawase blur twin: one thread per output texel runs one down- or
// up-sample pass. The downsample kernel is a 5-tap "center plus four diagonals"
// filter normalized by 8; the upsample kernel is an 8-tap tent of four edge
// taps plus four double-weighted diagonal taps normalized by 12. Both share a
// hand-rolled bilinear sampler with clamp-to-edge addressing. They mirror the
// CPU golden `particle::kawase_dual_blur`, use only the portable core-WGSL
// subset (min/floor and + - * / on vectors plus unsigned index math), and take
// no optional feature, so they run unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard dual-Kawase blur (Bjorge, SIGGRAPH 2015); no third-party
// engine source or derived code.

struct Params {
    // Source image extents in texels (the pass input).
    src_w: u32,
    src_h: u32,
    // Destination image extents in texels (the pass output, one thread each).
    dst_w: u32,
    dst_h: u32,
    // Diagonal tap distance in source texels (classically 0.5).
    offset: f32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

// Normalization divisor for the down kernel: weights 4 + 1 + 1 + 1 + 1 = 8.
const DOWN_NORM: f32 = 8.0;
// Weight of the center tap in the down kernel.
const DOWN_CENTER_WEIGHT: f32 = 4.0;
// Normalization divisor for the up kernel: 4 edge taps of weight 1 plus 4
// diagonal taps of weight 2 sum to 12.
const UP_NORM: f32 = 12.0;
// Weight of each diagonal tap in the up kernel (edge taps carry weight 1).
const UP_DIAGONAL_WEIGHT: f32 = 2.0;

// Clamps a (possibly negative) integer sample coordinate into `[0, extent)`,
// the clamp-to-edge addressing the reference `clamp_index` uses. The host never
// dispatches with an empty source, so `extent` is always positive here, but the
// zero guard is kept to mirror the reference exactly.
fn clamp_index(coord: i32, extent: u32) -> u32 {
    if (extent == 0u) {
        return 0u;
    }
    if (coord < 0) {
        return 0u;
    }
    let c = u32(coord);
    let hi = extent - 1u;
    return min(c, hi);
}

// Reads source texel `(x, y)` as an RGB triple from the flat 3-per-pixel buffer.
// Coordinates are pre-clamped by `clamp_index`, so the index is always in range.
fn load_texel(x: u32, y: u32) -> vec3<f32> {
    let base = (y * params.src_w + x) * 3u;
    return vec3<f32>(src[base], src[base + 1u], src[base + 2u]);
}

// Writes an RGB triple to destination texel `(x, y)` in the flat buffer.
fn store_texel(x: u32, y: u32, c: vec3<f32>) {
    let base = (y * params.dst_w + x) * 3u;
    dst[base] = c.x;
    dst[base + 1u] = c.y;
    dst[base + 2u] = c.z;
}

// Linear interpolation `a + (b - a) * t`, matching the reference `lerp3` order
// exactly (not the `a*(1-t) + b*t` form `mix` would use), so the low mantissa
// bits agree.
fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// Hand-rolled bilinear sample at normalized coordinates `(u, v)` with
// clamp-to-edge addressing, mirroring the reference `sample_bilinear`. Texel
// centers sit at `(i + 0.5) / extent`; taps outside the grid clamp to the
// nearest edge texel.
fn sample_bilinear(u: f32, v: f32) -> vec3<f32> {
    let fx = u * f32(params.src_w) - 0.5;
    let fy = v * f32(params.src_h) - 0.5;
    let fx0 = floor(fx);
    let fy0 = floor(fy);
    let tx = fx - fx0;
    let ty = fy - fy0;
    let bx = i32(fx0);
    let by = i32(fy0);
    let x0 = clamp_index(bx, params.src_w);
    let x1 = clamp_index(bx + 1, params.src_w);
    let y0 = clamp_index(by, params.src_h);
    let y1 = clamp_index(by + 1, params.src_h);
    let p00 = load_texel(x0, y0);
    let p10 = load_texel(x1, y0);
    let p01 = load_texel(x0, y1);
    let p11 = load_texel(x1, y1);
    let top = lerp3(p00, p10, tx);
    let bot = lerp3(p01, p11, tx);
    return lerp3(top, bot, ty);
}

@compute @workgroup_size(64)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.dst_w * params.dst_h;
    if (idx >= total) {
        return;
    }
    let ox = idx % params.dst_w;
    let oy = idx / params.dst_w;
    let hp_u = params.offset / f32(params.src_w);
    let hp_v = params.offset / f32(params.src_h);
    let cu = (f32(ox) + 0.5) / f32(params.dst_w);
    let cv = (f32(oy) + 0.5) / f32(params.dst_h);
    // Reference accumulation order: center * 4, then the four diagonal taps.
    var acc = sample_bilinear(cu, cv) * DOWN_CENTER_WEIGHT;
    acc = acc + sample_bilinear(cu - hp_u, cv - hp_v);
    acc = acc + sample_bilinear(cu + hp_u, cv + hp_v);
    acc = acc + sample_bilinear(cu + hp_u, cv - hp_v);
    acc = acc + sample_bilinear(cu - hp_u, cv + hp_v);
    store_texel(ox, oy, acc * (1.0 / DOWN_NORM));
}

@compute @workgroup_size(64)
fn upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.dst_w * params.dst_h;
    if (idx >= total) {
        return;
    }
    let ox = idx % params.dst_w;
    let oy = idx / params.dst_w;
    let hp_u = params.offset / f32(params.src_w);
    let hp_v = params.offset / f32(params.src_h);
    let cu = (f32(ox) + 0.5) / f32(params.dst_w);
    let cv = (f32(oy) + 0.5) / f32(params.dst_h);
    // Reference accumulation order: the four axis-aligned edge taps at
    // +/- 2*offset, then the four double-weighted diagonal taps at +/- offset.
    var acc = sample_bilinear(cu - 2.0 * hp_u, cv);
    acc = acc + sample_bilinear(cu + 2.0 * hp_u, cv);
    acc = acc + sample_bilinear(cu, cv - 2.0 * hp_v);
    acc = acc + sample_bilinear(cu, cv + 2.0 * hp_v);
    acc = acc + sample_bilinear(cu - hp_u, cv - hp_v) * UP_DIAGONAL_WEIGHT;
    acc = acc + sample_bilinear(cu + hp_u, cv - hp_v) * UP_DIAGONAL_WEIGHT;
    acc = acc + sample_bilinear(cu - hp_u, cv + hp_v) * UP_DIAGONAL_WEIGHT;
    acc = acc + sample_bilinear(cu + hp_u, cv + hp_v) * UP_DIAGONAL_WEIGHT;
    store_texel(ox, oy, acc * (1.0 / UP_NORM));
}
"#;

/// One dual-`Kawase` blur request: the source image and the pyramid parameters.
///
/// Mirrors the `(image, passes, offset)` triple the reference
/// [`dual_blur`](prism_render_architecture::particle::kawase_dual_blur::KawaseImage::dual_blur)
/// consumes, carrying the sub-`texel` diagonal `offset` and the `passes` count
/// in the reference
/// [`KawaseBlurParams`](prism_render_architecture::particle::kawase_dual_blur::KawaseBlurParams)
/// block. Derives only [`PartialEq`] (no `Eq`/`Hash`) because the image holds
/// `f32` pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct KawaseBlurQuery {
    /// The source linear `HDR` `RGB` image to blur.
    pub image: KawaseImage,
    /// The pyramid parameters: the diagonal tap `offset` and `passes` count.
    pub params: KawaseBlurParams,
}

/// Uniform parameters for one down/up pass dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`KAWASE_BLUR_WGSL`]: the source and destination
/// extents, the diagonal `offset` and three pad words — `32` bytes, each field
/// at the `std140` uniform offset the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Source image width in `texels`.
    src_w: u32,
    /// Source image height in `texels`.
    src_h: u32,
    /// Destination image width in `texels`.
    dst_w: u32,
    /// Destination image height in `texels`.
    dst_h: u32,
    /// Diagonal tap distance in source `texels`.
    offset: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

impl GpuParams {
    /// Packs one pass's source/destination extents and diagonal `offset`.
    fn new(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize, offset: f32) -> GpuParams {
        GpuParams {
            src_w: src_w as u32,
            src_h: src_h as u32,
            dst_w: dst_w as u32,
            dst_h: dst_h as u32,
            offset,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }
}

/// One pass in the recorded down/up chain: the source and destination buffer
/// indices into the host-owned buffer list, the packed uniform and whether it
/// is a down pass (otherwise an up pass).
struct PassSpec {
    /// Index of the source buffer in the host buffer list.
    src: usize,
    /// Index of the destination buffer in the host buffer list.
    dst: usize,
    /// The packed uniform parameters for this pass.
    params: GpuParams,
    /// Whether this is a down pass (`true`) or an up pass (`false`).
    down: bool,
}

/// A compiled, reusable dual-`Kawase` blur pipeline pair (down and up).
pub struct GpuKawaseBlur {
    #[expect(
        dead_code,
        reason = "kept alive so the pipelines it produced stay valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline_down: ComputePipeline,
    pipeline_up: ComputePipeline,
}

impl GpuKawaseBlur {
    /// Compiles the dual-`Kawase` down and up kernels on `ctx`.
    ///
    /// The kernels use only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuKawaseBlur {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_kawase_blur"),
            source: ShaderSource::Wgsl(KAWASE_BLUR_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_kawase_blur_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_kawase_blur_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline_down = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_kawase_blur_downsample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("downsample"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_up = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_kawase_blur_upsample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("upsample"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuKawaseBlur {
            module,
            layout,
            pipeline_down,
            pipeline_up,
        }
    }

    /// Runs the full dual-`Kawase` chain on `query.image`, returning the blurred
    /// image at the original resolution.
    ///
    /// The result equals
    /// [`query.image.dual_blur(query.params.passes, query.params.offset)`](prism_render_architecture::particle::kawase_dual_blur::KawaseImage::dual_blur)
    /// to within the tolerance documented on this module. With `passes == 0` or
    /// an empty image the input is returned unchanged, exactly as the reference
    /// does (and no dispatch is issued, since a storage buffer cannot be
    /// zero-sized). Otherwise one dispatch is issued per recorded down pass and
    /// one per retraced up pass, chaining the storage buffers so each pass reads
    /// the previous pass's output.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &KawaseBlurQuery) -> KawaseImage {
        let image = &query.image;
        let passes = query.params.passes as usize;
        let offset = query.params.offset;
        if passes == 0 || image.is_empty() {
            return image.clone();
        }
        let device = ctx.device();

        // Record the extent of every level before each down pass, exactly as the
        // reference `dual_blur` does, so the up chain can retrace them (a
        // `div_ceil` halving is not reversible by another `div_ceil`).
        let mut dims: Vec<(usize, usize)> = Vec::with_capacity(passes + 1);
        let mut cw = image.width;
        let mut ch = image.height;
        dims.push((cw, ch));
        for _ in 0..passes {
            cw = cw.div_ceil(2);
            ch = ch.div_ceil(2);
            dims.push((cw, ch));
        }

        // Buffer 0 holds the uploaded source image; every pass appends a fresh
        // output buffer and the next pass reads it.
        let mut buffers: Vec<wgpu::Buffer> = Vec::with_capacity(2 * passes + 1);
        buffers.push(device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_kawase_blur_source"),
            contents: bytemuck::cast_slice(&flatten_pixels(image)),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        }));

        let mut specs: Vec<PassSpec> = Vec::with_capacity(2 * passes);
        let mut cursor = 0usize;
        // Down chain: level `k` -> level `k + 1`.
        for k in 0..passes {
            let (sw, sh) = dims[k];
            let (dw, dh) = dims[k + 1];
            buffers.push(empty_image_buffer(device, dw, dh));
            let dst = buffers.len() - 1;
            specs.push(PassSpec {
                src: cursor,
                dst,
                params: GpuParams::new(sw, sh, dw, dh, offset),
                down: true,
            });
            cursor = dst;
        }
        // Up chain: retrace the recorded extents from the smallest level back to
        // the original resolution.
        for k in (0..passes).rev() {
            let (sw, sh) = dims[k + 1];
            let (dw, dh) = dims[k];
            buffers.push(empty_image_buffer(device, dw, dh));
            let dst = buffers.len() - 1;
            specs.push(PassSpec {
                src: cursor,
                dst,
                params: GpuParams::new(sw, sh, dw, dh, offset),
                down: false,
            });
            cursor = dst;
        }
        let final_idx = cursor;

        // Build the per-pass uniform buffers and bind groups up front so the
        // encoder can record every pass back to back.
        let mut param_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(specs.len());
        let mut bind_groups: Vec<wgpu::BindGroup> = Vec::with_capacity(specs.len());
        for spec in &specs {
            let params_buf = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("prism_volumetric_kawase_blur_params"),
                contents: bytemuck::bytes_of(&spec.params),
                usage: BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("prism_volumetric_kawase_blur_bind_group"),
                layout: &self.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: buffers[spec.src].as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: buffers[spec.dst].as_entire_binding(),
                    },
                ],
            });
            param_buffers.push(params_buf);
            bind_groups.push(bind_group);
        }

        let final_texels = image.width.saturating_mul(image.height);
        let out_bytes = (final_texels * CHANNELS * size_of::<f32>()) as u64;
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_kawase_blur_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_kawase_blur_encoder"),
        });
        for (spec, bind_group) in specs.iter().zip(bind_groups.iter()) {
            let pipeline = if spec.down {
                &self.pipeline_down
            } else {
                &self.pipeline_up
            };
            let total = spec.params.dst_w * spec.params.dst_h;
            let groups = total.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_kawase_blur_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
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

        KawaseImage::new(image.width, image.height, pixels)
    }
}

/// Flattens a [`KawaseImage`] into the row-major, 3-per-pixel `f32` layout the
/// device storage buffers expect.
fn flatten_pixels(image: &KawaseImage) -> Vec<f32> {
    let mut data = Vec::with_capacity(image.pixels.len() * CHANNELS);
    for pixel in &image.pixels {
        data.push(pixel[0]);
        data.push(pixel[1]);
        data.push(pixel[2]);
    }
    data
}

/// Allocates an uninitialized storage buffer sized for a `width * height` `RGB`
/// image, usable as a pass output and readable as the next pass's source.
fn empty_image_buffer(device: &wgpu::Device, width: usize, height: usize) -> wgpu::Buffer {
    let texels = width.saturating_mul(height);
    device.create_buffer(&BufferDescriptor {
        label: Some("prism_volumetric_kawase_blur_level"),
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
