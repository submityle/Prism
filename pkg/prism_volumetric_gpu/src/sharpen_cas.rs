//! `wgpu` compute twin of the `AMD` `FidelityFX` Contrast-Adaptive Sharpening
//! (`CAS`) kernel
//! ([`sharpen_cas`](prism_render_architecture::particle::sharpen_cas), design
//! §16-§21, "`CAS` 锐化").
//!
//! `CAS` is a single-pass, locally adaptive sharpen that raises apparent detail
//! after an upscale, temporal reprojection or alpha-coverage blur without the
//! ringing halos of an unsharp mask and without over-sharpening already
//! high-contrast regions. The `CPU` golden
//! [`CasParams`](prism_render_architecture::particle::sharpen_cas::CasParams)
//! owns that math; [`GpuSharpenCas`] is the on-device twin that runs one thread
//! per *output* `texel` and reproduces the image the batch form
//! [`CasParams::apply`](prism_render_architecture::particle::sharpen_cas::CasParams::apply)
//! produces, `texel` for `texel`. A passing real-device parity test is
//! therefore direct evidence the ported kernel gathers the same `3x3`
//! clamp-to-edge neighborhood, forms the same per-channel `min`/`max`, derives
//! the same adaptive amplitude and blends with the same energy-normalized
//! denominator the reference does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The whole per-`texel` sharpen contract is reproduced in one kernel: the
//! `3x3` clamp-to-edge
//! [`gather`](prism_render_architecture::particle::sharpen_cas::CasParams::apply)
//! that replicates the border texel (only the center and the four cross
//! neighbors `up`/`down`/`left`/`right` participate, as in `AMD` `CAS`), the
//! sharpness-driven negative
//! [`peak_weight`](prism_render_architecture::particle::sharpen_cas::CasParams)
//! `-1 / mix(8, 5, sharpness)`, and the per-channel
//! [`sharpen_channel`](prism_render_architecture::particle::sharpen_cas::CasParams::sharpen_taps)
//! blend: the local `min`/`max` over the five taps, the near-black/near-white
//! amplitude `sqrt(clamp(min(mn, 2 - mx) / mx, 0, 1))`, the `w = amp * peak`
//! scale, the energy-normalizing denominator `1 + 4 w`, and the final clamp to a
//! non-negative result. Every scalar is evaluated in the reference's own order
//! so the low mantissa bits agree.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `abs`, `sqrt` and `+ - * /` plus unsigned index arithmetic — with
//! no `sin`, `cos`, `exp`, `log`, `pow`, `tan` or the built-in `smoothstep`, and
//! no optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The single transcendental is the one `sqrt` the amplitude needs,
//! which is a core-`WGSL` built-in; every divide is by a reference-guarded
//! denominator (`mx` guarded by the near-black test, `1 + 4 w` guarded by the
//! near-zero test, and the compile-time `mix(8, 5, sharpness)` which is never
//! zero), each written as the `y / x` form the reference uses.
//!
//! # Correctness model
//!
//! Each output `texel` channel is a fixed, non-reorderable sequence of
//! `min`/`max`, one `clamp`, one `sqrt`, multiplies, adds and two guarded
//! divides, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, and both the hardware `sqrt` and the `reciprocal` implied by
//! the divide carry a few units in the last place of rounding slack, perturbing
//! the low mantissa bits. The parity test therefore asserts a tolerance
//! (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough to catch a genuinely
//! wrong port (a dropped tap, a swapped `min`/`max`, a wrong amplitude, a lost
//! denominator guard) yet loose enough to admit legal fused multiply-add
//! contraction plus the `sqrt`/`reciprocal` `ULP` slack.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `AMD` `FidelityFX` Contrast-Adaptive Sharpening (`CAS`)
//! plus `wgpu` compute dispatch; mirrors the `CPU` golden
//! `prism_render_architecture::particle::sharpen_cas`; no third-party engine
//! source or derived code.
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

/// Number of threads per workgroup. `64` is the portable, warp-friendly size
/// the sibling twins use; a one-dimensional dispatch covers the flat `texel`
/// grid.
const WORKGROUP_SIZE: u32 = 64;

/// `f32` channels stored per `texel`. The host flattens each `RGB` triple into a
/// `width * height * 3` `f32` array so a storage buffer needs no `vec3`
/// alignment padding, exactly as the reference keeps its pixels.
const CHANNELS: usize = 3;

/// The portable core-`WGSL` `CAS` sharpen kernel, embedded inline so the twin
/// ships as a single source file. The single entry point `sharpen` mirrors the
/// `CPU` golden
/// [`sharpen_channel`](prism_render_architecture::particle::sharpen_cas::CasParams::sharpen_taps)
/// scalar for scalar; see the module documentation for the algorithm.
const SHARPEN_CAS_WGSL: &str = r#"
// AMD FidelityFX Contrast-Adaptive Sharpening (CAS) twin: one thread per output
// texel gathers the 3x3 clamp-to-edge neighborhood, keeps the center and the
// four cross neighbors, forms the per-channel min/max, derives the adaptive
// amplitude sqrt(clamp(min(mn, 2 - mx) / mx, 0, 1)), scales it by the negative
// peak weight -1 / mix(8, 5, sharpness) and blends with the energy-normalizing
// denominator 1 + 4 w, clamping the result to be non-negative. It mirrors the
// CPU golden `particle::sharpen_cas`, uses only the portable core-WGSL subset
// (min/max/clamp/abs/sqrt and + - * / plus unsigned index math) with no
// optional feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard AMD FidelityFX CAS sharpen; no third-party engine source
// or derived code.

struct Params {
    // Sharpen strength in 0..=1 (clamped again in the shader, as the reference
    // `peak_weight` does, so a directly-supplied out-of-range value is safe).
    sharpness: f32,
    // Image width in texels.
    width: u32,
    // Image height in texels (one thread per texel over width * height).
    height: u32,
    // Padding to a 16-byte, std140-aligned uniform struct.
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

// Denominators with magnitude below this are treated as (near) zero so the
// guarded divides fall back to a defined result instead of dividing by zero,
// mirroring the reference `MIN_DENOM`. It doubles as the near-black guard on the
// local maximum.
const MIN_DENOM: f32 = 1e-6;
// Low endpoint of the CAS peak-weight interpolation (softest sharpen).
const PEAK_SOFT: f32 = 8.0;
// High endpoint of the CAS peak-weight interpolation (sharpest).
const PEAK_HARD: f32 = 5.0;

// Linear interpolation `a + (b - a) * t`, the hand-rolled `mix` the reference
// peak weight uses (not the built-in `mix`), so the low mantissa bits agree.
fn mix_scalar(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// The negative peak blend weight for the clamped sharpness:
// `-1 / mix(8, 5, sharpness)`, matching the reference `peak_weight` order.
fn peak_weight(sharpness: f32) -> f32 {
    let sharp = clamp(sharpness, 0.0, 1.0);
    return -1.0 / mix_scalar(PEAK_SOFT, PEAK_HARD, sharp);
}

// Clamps a coordinate stepped by `delta` (only -1, 0 or +1) into `[0, dim)`,
// replicating the edge texel. `dim` is guaranteed non-zero by the host guard,
// mirroring the reference `clamp_coord` (including the saturating decrement).
fn clamp_coord(coord: u32, delta: i32, dim: u32) -> u32 {
    let last = dim - 1u;
    var shifted = coord;
    if (delta < 0) {
        if (coord > 0u) {
            shifted = coord - 1u;
        } else {
            shifted = 0u;
        }
    } else if (delta > 0) {
        shifted = coord + 1u;
    }
    return min(shifted, last);
}

// Reads texel `(x, y)` as an RGB triple from the flat 3-per-pixel buffer.
// Coordinates are pre-clamped by `clamp_coord`, so the index is always in range.
fn load_texel(x: u32, y: u32) -> vec3<f32> {
    let base = (y * params.width + x) * 3u;
    return vec3<f32>(src[base], src[base + 1u], src[base + 2u]);
}

// The per-channel CAS blend, mirroring the reference `sharpen_channel` term for
// term and in its exact order.
fn sharpen_channel(peak: f32, up: f32, down: f32, left: f32, right: f32, center: f32) -> f32 {
    let mn = min(center, min(min(up, down), min(left, right)));
    let mx = max(center, max(max(up, down), max(left, right)));
    var amp: f32 = 0.0;
    if (mx >= MIN_DENOM) {
        let ratio = min(mn, 2.0 - mx) / mx;
        amp = sqrt(clamp(ratio, 0.0, 1.0));
    }
    let w = amp * peak;
    let denom = 1.0 + 4.0 * w;
    var result: f32 = center;
    if (abs(denom) >= MIN_DENOM) {
        result = (w * (up + down + left + right) + center) / denom;
    }
    return max(result, 0.0);
}

@compute @workgroup_size(64)
fn sharpen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if (idx >= total) {
        return;
    }
    let x = idx % params.width;
    let y = idx / params.width;
    // The 3x3 clamp-to-edge cross neighborhood: only up/down/left/right/center
    // participate, exactly the taps the reference `sharpen_taps` reads.
    let xm = clamp_coord(x, -1, params.width);
    let xp = clamp_coord(x, 1, params.width);
    let ym = clamp_coord(y, -1, params.height);
    let yp = clamp_coord(y, 1, params.height);
    let center = load_texel(x, y);
    let up = load_texel(x, ym);
    let down = load_texel(x, yp);
    let left = load_texel(xm, y);
    let right = load_texel(xp, y);
    let peak = peak_weight(params.sharpness);
    let base = idx * 3u;
    dst[base] = sharpen_channel(peak, up.x, down.x, left.x, right.x, center.x);
    dst[base + 1u] = sharpen_channel(peak, up.y, down.y, left.y, right.y, center.y);
    dst[base + 2u] = sharpen_channel(peak, up.z, down.z, left.z, right.z, center.z);
}
"#;

/// One `CAS` sharpen request: the source `RGB` image, its extents and the
/// sharpness parameter.
///
/// Mirrors the `(img, width, height)` triple plus the `sharpness` the reference
/// [`CasParams::apply`](prism_render_architecture::particle::sharpen_cas::CasParams::apply)
/// consumes. Derives only [`PartialEq`] (no `Eq`/`Hash`) because the image holds
/// `f32` pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct SharpenCasQuery {
    /// The source linear `RGB` image in row-major order (`width * height`
    /// `texels`).
    pub img: Vec<[f32; 3]>,
    /// Image width in `texels`.
    pub width: usize,
    /// Image height in `texels`.
    pub height: usize,
    /// Sharpen strength in `0..=1`; the shader clamps it, matching the reference
    /// [`peak_weight`](prism_render_architecture::particle::sharpen_cas::CasParams).
    pub sharpness: f32,
}

/// Uniform parameters for the sharpen dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`SHARPEN_CAS_WGSL`]: the `sharpness` scalar, the image
/// extents and one pad word — `16` bytes, a single `vec4` slot at the `std140`
/// uniform offsets the shader expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Sharpen strength in `0..=1`.
    sharpness: f32,
    /// Image width in `texels`.
    width: u32,
    /// Image height in `texels`.
    height: u32,
    /// Padding word to a `16`-byte, `std140`-aligned uniform struct.
    pad0: u32,
}

impl GpuParams {
    /// Packs the sharpness and the image extents for one dispatch.
    fn new(sharpness: f32, width: usize, height: usize) -> GpuParams {
        GpuParams {
            sharpness,
            width: width as u32,
            height: height as u32,
            pad0: 0,
        }
    }
}

/// A compiled, reusable `CAS` sharpen pipeline.
pub struct GpuSharpenCas {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSharpenCas {
    /// Compiles the `CAS` sharpen kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus the core
    /// `sqrt` built-in, so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSharpenCas {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_sharpen_cas"),
            source: ShaderSource::Wgsl(SHARPEN_CAS_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_sharpen_cas_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_sharpen_cas_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_sharpen_cas_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sharpen"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSharpenCas {
            module,
            layout,
            pipeline,
        }
    }

    /// Runs the `CAS` sharpen over `query.img`, returning the sharpened `RGB`
    /// image in row-major order.
    ///
    /// The result equals
    /// [`CasParams::new(query.sharpness).apply(&query.img, query.width, query.height)`](prism_render_architecture::particle::sharpen_cas::CasParams::apply)
    /// to within the tolerance documented on this module. When either dimension
    /// is zero or `query.img` holds fewer than `width * height` `texels`, an
    /// empty vector is returned and no dispatch is issued (a storage buffer
    /// cannot be zero-sized), exactly as the reference returns an empty vector.
    /// Otherwise a single one-dimensional dispatch covers the flat `texel` grid,
    /// one thread per output `texel`.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &SharpenCasQuery) -> Vec<[f32; 3]> {
        let count = query.width.saturating_mul(query.height);
        if query.width == 0 || query.height == 0 || query.img.len() < count {
            return Vec::new();
        }
        let device = ctx.device();

        let flat = flatten_pixels(&query.img[..count]);
        let src = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sharpen_cas_source"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        let out_bytes = (count * CHANNELS * size_of::<f32>()) as u64;
        let dst = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sharpen_cas_dst"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams::new(query.sharpness, query.width, query.height);
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_sharpen_cas_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_sharpen_cas_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: src.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: dst.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_sharpen_cas_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_sharpen_cas_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_sharpen_cas_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output `texel`, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&dst, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut result: Vec<[f32; 3]> = Vec::with_capacity(count);
        for chunk in out.chunks_exact(CHANNELS) {
            result.push([chunk[0], chunk[1], chunk[2]]);
        }
        debug_assert_eq!(result.len(), count);
        result
    }
}

/// Flattens a slice of `RGB` pixels into the row-major, 3-per-pixel `f32`
/// layout the device storage buffer expects.
fn flatten_pixels(pixels: &[[f32; 3]]) -> Vec<f32> {
    let mut data = Vec::with_capacity(pixels.len() * CHANNELS);
    for pixel in pixels {
        data.push(pixel[0]);
        data.push(pixel[1]);
        data.push(pixel[2]);
    }
    data
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
