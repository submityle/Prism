//! `wgpu` compute twin of the temporal-dither threshold source
//! ([`temporal_dither`](prism_render_architecture::particle::temporal_dither),
//! design §16, "时域抖动阈值").
//!
//! Translucent particles resolved with a hard alpha test leave a crawling,
//! aliased edge. Stochastic (dithered) transparency keeps or discards each
//! fragment against a spatially varying threshold, then lets the temporal
//! accumulation pass average the pattern back into a smooth gradient. The `CPU`
//! golden
//! [`temporal_dither`](prism_render_architecture::particle::temporal_dither)
//! owns the contract for that threshold source; [`GpuTemporalDither`] is the
//! on-device twin that evaluates one thread per pixel and reproduces the same
//! threshold, raw rank and keep/discard decision the reference produces. A
//! passing real-device parity test is therefore direct evidence the ported
//! kernel interleaves the same `Bayer` bits, mixes the same `blue-noise` hash
//! constants and rotates by the same golden-ratio step the reference does, not
//! merely that the shader compiles.
//!
//! # What is twinned
//!
//! A single kernel reproduces, per pixel and per frame, the whole threshold
//! stack: (1) the recursive `Bayer` rank
//! [`bayer4_raw`](prism_render_architecture::particle::temporal_dither::bayer4_raw)
//! /
//! [`bayer8_raw`](prism_render_architecture::particle::temporal_dither::bayer8_raw)
//! with the integer recurrence `M_{2n} = [[4M, 4M+2], [4M+3, 4M+1]]`, unrolled
//! into a `Horner` loop over bit positions because `WGSL` has no recursion; (2)
//! the normalized `Bayer` thresholds
//! [`bayer4`](prism_render_architecture::particle::temporal_dither::bayer4) /
//! [`bayer8`](prism_render_architecture::particle::temporal_dither::bayer8)
//! (rank divided by `16` / `64`); (3) the pure-integer avalanche
//! [`blue_noise01`](prism_render_architecture::particle::temporal_dither::blue_noise01),
//! reproduced bit-for-bit (`u32` wrapping multiplies and `xor`-shifts, top 24
//! bits normalized by `2^24`); (4) the golden-ratio
//! [`temporal_offset`](prism_render_architecture::particle::temporal_dither::temporal_offset)
//! rotation folded into the `Bayer` thresholds via
//! [`dither_threshold`](prism_render_architecture::particle::temporal_dither::dither_threshold);
//! and (5) the strict keep/discard test
//! [`should_discard`](prism_render_architecture::particle::temporal_dither::should_discard)
//! (`alpha < threshold`). The `DitherMode` selecting between the three sources
//! travels in the uniform; the per-pixel `(x, y, frame, alpha)` tuple travels
//! in a storage input buffer; the raw rank, keep/discard mask and normalized
//! threshold come back in a storage output buffer.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — integer `+ - *`,
//! bit shifts and masks, `xor`, `floor` and `f32` division by compile-time
//! constants — with no `sin`, `cos`, `exp`, `log`, `pow`, `tan` or optional
//! device feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
//! `WGSL` `u32` multiplication wraps on overflow, matching the reference's
//! `wrapping_mul`, so the avalanche hash and the golden-ratio rotation agree
//! bit-for-bit.
//!
//! # Correctness model
//!
//! Every integer step — the `Bayer` recurrence, the avalanche hash and the
//! golden-ratio multiply — is exact `u32` arithmetic, so the raw ranks and
//! hashes are bit-for-bit identical and the parity test compares them with `==`.
//! The only floating-point steps are the widening `u32`-to-`f32` cast of a
//! value below `2^24` (exact in an `f32` mantissa), a division by a
//! compile-time power-of-two or small constant, and a `floor`-based `wrap01`.
//! `CPU` and `GPU` evaluate that identical closed form, so the normalized
//! thresholds agree to within a few units in the last place; the parity test
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` for the threshold while
//! keeping the raw rank and the keep/discard mask exact.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: classic ordered `Bayer` dither plus an integer `blue-noise`
//! avalanche and a golden-ratio temporal rotation (`Prism` design §16) ported
//! to a `wgpu` compute dispatch; no third-party engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::temporal_dither::DitherMode;
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
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// The dither kernel source. One thread per pixel reads its `(x, y, frame,
/// alpha)` tuple, evaluates the `DitherMode` the uniform selects and writes the
/// raw rank, keep/discard mask and normalized threshold. It mirrors the `CPU`
/// golden `particle::temporal_dither` step for step and uses only the portable
/// core-`WGSL` subset, so it runs unmodified on `Metal`, `Vulkan` and `DX12`.
const TEMPORAL_DITHER_WGSL: &str = r#"
// Temporal-dither threshold twin: one thread per pixel reproduces the CPU
// golden `particle::temporal_dither`. The Bayer recurrence is unrolled into a
// Horner loop (WGSL has no recursion), the blue-noise avalanche is reproduced
// with wrapping u32 arithmetic, and the golden-ratio temporal rotation folds
// into the Bayer thresholds. Only the portable core-WGSL subset is used
// (integer + - *, shifts, masks, xor, floor and f32 divides by constants), so
// the kernel takes no optional feature and runs on Metal, Vulkan and DX12.
//
// Provenance: classic ordered Bayer dither plus an integer blue-noise avalanche
// and a golden-ratio temporal rotation (Prism design §16); no third-party
// engine source or derived code.

struct Params {
    // Threshold source: 0 = Bayer4, 1 = Bayer8, 2 = BlueNoise.
    mode: u32,
    // Number of valid pixels in the input buffer (bounds the dispatch tail).
    count: u32,
    // Padding to a 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
}

// One input pixel: integer coordinates, frame index and the fragment opacity.
struct DitherInput {
    x: u32,
    y: u32,
    frame: u32,
    alpha: f32,
}

// One output sample: the raw integer rank (Bayer rank or the 24-bit blue-noise
// hash), the keep/discard mask (1 = discard) and the normalized threshold.
struct DitherOutput {
    raw: u32,
    mask: u32,
    threshold: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<DitherInput>;
@group(0) @binding(2) var<storage, read_write> outputs: array<DitherOutput>;

// The 0x9E3779B9 golden-ratio integer step (floor(2^32 / phi)) rotating the
// Bayer threshold from one frame to the next.
const GOLDEN_U32: u32 = 0x9E3779B9u;
// 2^24, the largest power of two exactly representable in an f32 mantissa, used
// as the normalization divisor so every hashed threshold is exact.
const NORM_24: f32 = 16777216.0;

const MODE_BAYER4: u32 = 0u;
const MODE_BAYER8: u32 = 1u;

// Integer avalanche hash (a bit-mix finalizer) with good bit diffusion. Pure
// u32 wrapping arithmetic and shifts, mirroring the golden `hash_u32`.
fn hash_u32(h0: u32) -> u32 {
    var h = h0;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

// The recursive Bayer base term for one bit level, mirroring the golden
// `match (y >= half, x >= half)` mapping with rows indexed by y, columns by x.
fn bayer_base(bx: u32, by: u32) -> u32 {
    if (by == 0u) {
        if (bx == 0u) {
            return 0u;
        }
        return 2u;
    }
    if (bx == 0u) {
        return 3u;
    }
    return 1u;
}

// The recursive Bayer rank unrolled into a Horner loop over `levels` bit
// positions. Bit 0 (the deepest recursion) carries the highest 4^k weight and
// bit `levels - 1` (the coarsest) carries weight 1, exactly reproducing
// `M_{2n} = [[4M, 4M+2], [4M+3, 4M+1]]`.
fn bayer_rank(x: u32, y: u32, levels: u32) -> u32 {
    var value = 0u;
    for (var i = 0u; i < levels; i = i + 1u) {
        let bx = (x >> i) & 1u;
        let by = (y >> i) & 1u;
        value = value * 4u + bayer_base(bx, by);
    }
    return value;
}

// Wraps a value into [0, 1) by subtracting its floor, mirroring `wrap01`.
fn wrap01(v: f32) -> f32 {
    return v - floor(v);
}

// The frame's threshold rotation as a fraction in [0, 1): the golden-ratio
// multiply, top 24 bits normalized, mirroring `temporal_fraction`.
fn temporal_fraction(frame: u32) -> f32 {
    let offset = frame * GOLDEN_U32;
    return f32(offset >> 8u) / NORM_24;
}

@compute @workgroup_size(64)
fn dither(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let inp = inputs[idx];
    let x = inp.x;
    let y = inp.y;
    let frame = inp.frame;
    let alpha = inp.alpha;

    var raw = 0u;
    var threshold = 0.0;
    if (params.mode == MODE_BAYER4) {
        // 4x4 tile: coordinates reduced modulo 4, two bit levels.
        raw = bayer_rank(x & 3u, y & 3u, 2u);
        threshold = wrap01(f32(raw) / 16.0 + temporal_fraction(frame));
    } else if (params.mode == MODE_BAYER8) {
        // 8x8 tile: coordinates reduced modulo 8, three bit levels.
        raw = bayer_rank(x & 7u, y & 7u, 3u);
        threshold = wrap01(f32(raw) / 64.0 + temporal_fraction(frame));
    } else {
        // BlueNoise: decorrelate the coordinates and frame by odd-prime
        // multipliers, mix, avalanche, then normalize the top 24 bits.
        let a = x * 0x9E3779B1u;
        let b = y * 0x85EBCA77u;
        let c = frame * 0xC2B2AE3Du;
        let h = hash_u32(a ^ b ^ c);
        raw = h >> 8u;
        threshold = f32(raw) / NORM_24;
    }

    var mask = 0u;
    // Strict test: an equal alpha and threshold is kept, not discarded.
    if (alpha < threshold) {
        mask = 1u;
    }

    outputs[idx].raw = raw;
    outputs[idx].mask = mask;
    outputs[idx].threshold = threshold;
    outputs[idx].pad = 0u;
}
"#;

/// One pixel to evaluate: integer coordinates, the frame index and the fragment
/// opacity tested against the threshold.
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because `alpha` is an `f32`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitherPixel {
    /// Pixel `x` coordinate (reduced modulo the tile for the `Bayer` modes).
    pub x: u32,
    /// Pixel `y` coordinate (reduced modulo the tile for the `Bayer` modes).
    pub y: u32,
    /// Frame index driving the temporal rotation / `blue-noise` decorrelation.
    pub frame: u32,
    /// Fragment opacity in `[0, 1]` tested against the dither threshold.
    pub alpha: f32,
}

impl DitherPixel {
    /// Builds a pixel sample at `(x, y)` on `frame` with opacity `alpha`.
    #[must_use]
    pub fn new(x: u32, y: u32, frame: u32, alpha: f32) -> DitherPixel {
        DitherPixel { x, y, frame, alpha }
    }
}

/// One batch dither request: the shared [`DitherMode`] and the pixels to
/// evaluate.
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because the pixels carry `f32`
/// opacities.
#[derive(Clone, Debug, PartialEq)]
pub struct DitherQuery {
    /// The threshold source every pixel in the batch samples.
    pub mode: DitherMode,
    /// The pixels to evaluate, one output sample produced per pixel.
    pub pixels: Vec<DitherPixel>,
}

/// One evaluated pixel: the raw integer rank, the keep/discard decision and the
/// normalized threshold.
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because `threshold` is an `f32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitherSample {
    /// The raw integer source: the `Bayer` rank (`0..16` / `0..64`) for the
    /// `Bayer` modes, or the 24-bit avalanche hash for `blue-noise`.
    pub raw: u32,
    /// Whether the fragment is discarded (`alpha < threshold`).
    pub discard: bool,
    /// The dither threshold in `[0, 1)`.
    pub threshold: f32,
}

/// Uniform parameters for the dispatch. `repr(C)` `std430`/`std140` layout
/// matching `Params` in [`TEMPORAL_DITHER_WGSL`]: the `DitherMode` code, the
/// pixel count and two pad words — `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Threshold source code (`0` = `Bayer4`, `1` = `Bayer8`, `2` = `BlueNoise`).
    mode: u32,
    /// Number of valid pixels in the input buffer.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
}

/// One input pixel in the layout the storage buffer expects, matching
/// `DitherInput` in [`TEMPORAL_DITHER_WGSL`] — `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuInput {
    /// Pixel `x` coordinate.
    x: u32,
    /// Pixel `y` coordinate.
    y: u32,
    /// Frame index.
    frame: u32,
    /// Fragment opacity.
    alpha: f32,
}

/// One output sample in the layout the storage buffer produces, matching
/// `DitherOutput` in [`TEMPORAL_DITHER_WGSL`] — `16` bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuOutput {
    /// The raw integer rank or hash.
    raw: u32,
    /// Keep/discard mask (`1` = discard).
    mask: u32,
    /// The normalized threshold in `[0, 1)`.
    threshold: f32,
    /// Padding word.
    pad: u32,
}

/// Maps a [`DitherMode`] onto the integer code the kernel's uniform carries.
fn mode_code(mode: DitherMode) -> u32 {
    match mode {
        DitherMode::Bayer4 => 0,
        DitherMode::Bayer8 => 1,
        DitherMode::BlueNoise => 2,
    }
}

/// A compiled, reusable temporal-dither evaluation pipeline.
pub struct GpuTemporalDither {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuTemporalDither {
    /// Compiles the temporal-dither kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuTemporalDither {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_temporal_dither"),
            source: ShaderSource::Wgsl(TEMPORAL_DITHER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_temporal_dither_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_temporal_dither_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_temporal_dither_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("dither"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuTemporalDither {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates every pixel in `query`, returning one [`DitherSample`] per
    /// pixel in input order.
    ///
    /// Each result equals the `CPU` golden
    /// [`dither_threshold`](prism_render_architecture::particle::temporal_dither::dither_threshold)
    /// /
    /// [`should_discard`](prism_render_architecture::particle::temporal_dither::should_discard)
    /// evaluated with the same `(x, y, frame, alpha)` and [`DitherMode`]: the
    /// raw rank and keep/discard mask bit-for-bit, the normalized threshold to
    /// within the tolerance documented on this module. An empty batch issues
    /// **no dispatch** — a storage buffer may not be zero-sized — and returns an
    /// empty vector.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &DitherQuery) -> Vec<DitherSample> {
        if query.pixels.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = query.pixels.len();

        let inputs: Vec<GpuInput> = query
            .pixels
            .iter()
            .map(|p| GpuInput {
                x: p.x,
                y: p.y,
                frame: p.frame,
                alpha: p.alpha,
            })
            .collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_temporal_dither_input"),
            contents: bytemuck::cast_slice(&inputs),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuOutput>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_temporal_dither_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            mode: mode_code(query.mode),
            count: count as u32,
            pad0: 0,
            pad1: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_temporal_dither_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_temporal_dither_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_temporal_dither_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_temporal_dither_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_temporal_dither_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per pixel, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw: Vec<GpuOutput> = bytemuck::cast_slice::<u8, GpuOutput>(&view).to_vec();
        drop(view);
        stage.unmap();

        raw.into_iter()
            .map(|o| DitherSample {
                raw: o.raw,
                discard: o.mask != 0,
                threshold: o.threshold,
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
