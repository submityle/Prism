//! `wgpu` compute twin of the `FXAA` luma-edge resolve
//! ([`fxaa`](prism_render_architecture::particle::fxaa), design sections 16-21,
//! "`FXAA` 3.11-style luma-adaptive antialiasing").
//!
//! `FXAA` (Fast Approximate Anti-Aliasing) is a single-pass, shader-only edge
//! smoother: for every output texel it inspects a `3x3` window of perceived
//! brightness (`luma`), decides whether a contrasty edge runs through it and
//! returns a blend weight a later resolve pass uses to pull the texel toward
//! its neighbors. The `CPU` golden
//! [`resolve_luma_grid`](prism_render_architecture::particle::fxaa::resolve_luma_grid)
//! owns that math; [`GpuFxaa`] is the on-device twin that runs one thread per
//! output texel and reproduces the same per-texel weight the reference does, so
//! a passing real-device parity test is direct evidence the ported kernel
//! clamps the same borders, gates the same edge test and smooths the same
//! subpixel term, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The kernel reproduces the reference evaluation chain texel for texel: the
//! clamp-to-edge `3x3`
//! [`sample_neighborhood`](prism_render_architecture::particle::fxaa) (border
//! texels replicate the edge, exactly as the reference `saturating_sub` / `min`
//! addressing does), the cross-window `max - min` edge test gated by
//! `max(edge_threshold_min, luma_max * edge_threshold)`, the
//! brightness-normalized edge strength `contrast / luma_max`, the subpixel term
//! (the `3x3` low-pass mean's departure from the center, normalized by the
//! full-window contrast, run through a hand-rolled Hermite `smoothstep` and
//! scaled by `subpix_quality`) and the final `max(edge_blend, subpixel)`
//! combination, each clamped to `0..=1` in the same order. The near-zero
//! denominator guard uses the same `1e-6` tolerance so the division fallbacks
//! agree. The edge-direction classifier is not part of the resolve weight the
//! reference emits, so it is intentionally not twinned.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `min`, `max`,
//! `clamp`, `floor`, `abs` and `+ - * /` on scalars plus unsigned index math —
//! with no `sin`, `cos`, `exp`, `log`, `pow`, `smoothstep` or optional device
//! feature, so it runs unmodified on `Metal`, `Vulkan` and `DX12`. The
//! `smoothstep` is expanded by hand to `t * t * (3 - 2t)` on a clamped unit
//! argument, mirroring the reference `smoothstep_unit`.
//!
//! # Correctness model
//!
//! Each output texel is a fixed, non-reorderable sequence of `min`/`max`
//! reductions, guarded divides, a Hermite polynomial and clamps, so `CPU` and
//! `GPU` evaluate the same closed form in the same order. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The parity test therefore asserts a tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`), tight enough to catch a genuinely wrong port (a swapped
//! neighbor, a dropped clamp, a wrong threshold) yet loose enough to admit a
//! legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `FXAA` 3.11 luma-adaptive antialiasing (Lottes,
//! `NVIDIA`, 2011) plus `wgpu` compute dispatch; no third-party engine source
//! or derived code.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::fxaa::FxaaParams;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly width
/// the sibling twins use, so one 1-D dispatch of `div_ceil(count, 64)` groups
/// covers every output texel.
const WORKGROUP_SIZE: u32 = 64;

/// The portable core-`WGSL` `FXAA` resolve kernel, embedded inline so the twin
/// ships as a single source file. The entry point `resolve` mirrors the `CPU`
/// golden
/// [`resolve_luma_grid`](prism_render_architecture::particle::fxaa::resolve_luma_grid)
/// texel for texel; see the module documentation for the algorithm.
const FXAA_WGSL: &str = r#"
// FXAA luma-edge resolve twin: one thread per output texel reproduces the CPU
// golden `particle::fxaa::resolve_luma_grid`. It samples a clamp-to-edge 3x3
// luma window, runs the cross-window edge test, and returns the greater of the
// brightness-normalized edge strength and the Hermite-smoothed subpixel term,
// all clamped to 0..=1. It uses only the portable core-WGSL subset
// (min/max/clamp/abs and + - * / on scalars plus unsigned index math), expands
// smoothstep by hand, takes no optional feature, and so runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: standard FXAA 3.11 (Lottes, NVIDIA, 2011); no third-party engine
// source or derived code.

struct Params {
    // Image extents in texels (one output thread per texel).
    width: u32,
    height: u32,
    // Brightness-relative contrast fraction required to flag an edge.
    edge_threshold: f32,
    // Absolute contrast floor for near-black regions.
    edge_threshold_min: f32,
    // 0..=1 scale applied to the subpixel-aliasing blend term.
    subpix_quality: f32,
    // Padding to a 32-byte, 16-byte-aligned uniform struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

// Near-zero denominator guard, matching the reference `CMP_EPS`. Divisions by a
// magnitude at or below this fall back to a defined zero instead of dividing by
// zero or propagating NaN.
const CMP_EPS: f32 = 1e-6;

// Clamp-to-edge decrement: step one texel toward 0, replicating the edge when
// already at 0, like the reference `saturating_sub(1)`.
fn clamp_dec(c: u32) -> u32 {
    if (c == 0u) {
        return 0u;
    }
    return c - 1u;
}

// Clamp-to-edge increment: step one texel toward `extent - 1`, replicating the
// far edge, like the reference `(c + 1).min(extent - 1)`. The host never
// dispatches with an empty image, so `extent` is always positive here.
fn clamp_inc(c: u32, extent: u32) -> u32 {
    let hi = extent - 1u;
    return min(c + 1u, hi);
}

// Hermite smoothstep on an already-unit input: `t*t*(3 - 2t)`, with the
// argument clamped to 0..=1 first, mirroring the reference `smoothstep_unit`.
// No transcendental is used.
fn smoothstep_unit(t: f32) -> f32 {
    let u = clamp(t, 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
}

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if (idx >= total) {
        return;
    }
    let x = idx % params.width;
    let y = idx / params.width;

    // Clamp-to-edge neighbor coordinates, matching the reference sampler.
    let xm = clamp_dec(x);
    let xp = clamp_inc(x, params.width);
    let ym = clamp_dec(y);
    let yp = clamp_inc(y, params.height);

    // The 3x3 luma window. `n` is the row above (y - 1), `s` the row below.
    let m = src[y * params.width + x];
    let n = src[ym * params.width + x];
    let s = src[yp * params.width + x];
    let e = src[y * params.width + xp];
    let w = src[y * params.width + xm];
    let ne = src[ym * params.width + xp];
    let nw = src[ym * params.width + xm];
    let se = src[yp * params.width + xp];
    let sw = src[yp * params.width + xm];

    // Cross-window extent over the center and four cross neighbors.
    let cross_lo = min(m, min(n, min(s, min(e, w))));
    let cross_hi = max(m, max(n, max(s, max(e, w))));
    let cross_range = cross_hi - cross_lo;
    let threshold = max(params.edge_threshold_min, cross_hi * params.edge_threshold);

    var out = 0.0;
    if (cross_range >= threshold) {
        // Brightness-normalized edge strength.
        var edge_blend = 0.0;
        if (cross_hi > CMP_EPS) {
            edge_blend = clamp(cross_range / cross_hi, 0.0, 1.0);
        }

        // Subpixel-aliasing term: departure of the 3x3 low-pass mean from the
        // center, normalized by the full-window contrast, smoothed and scaled.
        let sum = m + n + s + e + w + ne + nw + se + sw;
        let avg = sum / 9.0;
        let full_lo = min(cross_lo, min(ne, min(nw, min(se, sw))));
        let full_hi = max(cross_hi, max(ne, max(nw, max(se, sw))));
        let full_range = full_hi - full_lo;
        let contrast = abs(avg - m);
        var ratio = 0.0;
        if (full_range > CMP_EPS) {
            ratio = clamp(contrast / full_range, 0.0, 1.0);
        }
        let subpix = clamp(smoothstep_unit(ratio) * params.subpix_quality, 0.0, 1.0);

        out = clamp(max(edge_blend, subpix), 0.0, 1.0);
    }

    dst[idx] = out;
}
"#;

/// One `FXAA` resolve request: the row-major `luma` image, its extents and the
/// tuning parameters.
///
/// Mirrors the `(luma, width, height, params)` arguments the reference
/// [`resolve_luma_grid`](prism_render_architecture::particle::fxaa::resolve_luma_grid)
/// consumes. Derives only [`PartialEq`] (no [`Eq`]) because the image holds
/// `f32` samples.
#[derive(Clone, Debug, PartialEq)]
pub struct FxaaQuery {
    /// The row-major perceived-brightness (`luma`) image to resolve.
    pub luma: Vec<f32>,
    /// Image width in texels.
    pub width: usize,
    /// Image height in texels.
    pub height: usize,
    /// The `FXAA` edge/subpixel tuning scalars.
    pub params: FxaaParams,
}

/// Uniform parameters for one resolve dispatch. `repr(C)` layout matching
/// `Params` in [`FXAA_WGSL`]: the image extents, the three tuning scalars and
/// three pad words — `32` bytes, each field at the uniform offset the shader
/// expects.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Image width in texels.
    width: u32,
    /// Image height in texels.
    height: u32,
    /// Brightness-relative contrast fraction required to flag an edge.
    edge_threshold: f32,
    /// Absolute contrast floor for near-black regions.
    edge_threshold_min: f32,
    /// `0..=1` scale applied to the subpixel-aliasing blend term.
    subpix_quality: f32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

impl GpuParams {
    /// Packs one resolve request's extents and tuning scalars.
    fn new(width: usize, height: usize, params: &FxaaParams) -> GpuParams {
        GpuParams {
            width: width as u32,
            height: height as u32,
            edge_threshold: params.edge_threshold,
            edge_threshold_min: params.edge_threshold_min,
            subpix_quality: params.subpix_quality,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }
}

/// A compiled, reusable `FXAA` resolve compute pipeline.
pub struct GpuFxaa {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuFxaa {
    /// Compiles the `FXAA` resolve kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuFxaa {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_fxaa"),
            source: ShaderSource::Wgsl(FXAA_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_fxaa_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_fxaa_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_fxaa_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("resolve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuFxaa {
            module,
            layout,
            pipeline,
        }
    }

    /// Resolves `query.luma`, returning one blend weight per input texel in
    /// row-major order.
    ///
    /// The result equals
    /// [`resolve_luma_grid`](prism_render_architecture::particle::fxaa::resolve_luma_grid)
    /// to within the tolerance documented on this module. Returns an empty
    /// [`Vec`] when either dimension is zero or the buffer is shorter than
    /// `width * height`, exactly as the reference does (and no dispatch is
    /// issued, since a storage buffer cannot be zero-sized).
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, query: &FxaaQuery) -> Vec<f32> {
        let width = query.width;
        let height = query.height;
        let count = width.saturating_mul(height);
        if width == 0 || height == 0 || query.luma.len() < count {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams::new(width, height, &query.params);
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fxaa_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let src_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_fxaa_src"),
            contents: bytemuck::cast_slice(&query.luma[..count]),
            usage: BufferUsages::STORAGE,
        });
        let out_bytes = (count * size_of::<f32>()) as u64;
        let dst_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fxaa_dst"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_fxaa_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.bind_group(device, &params_buf, &src_buf, &dst_buf);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_fxaa_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_fxaa_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per output texel, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&dst_buf, 0, &stage, 0, out_bytes);
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
        out
    }

    /// Builds the three-entry bind group wiring the uniform, source and
    /// destination buffers to the resolve pipeline.
    fn bind_group(
        &self,
        device: &wgpu::Device,
        params_buf: &wgpu::Buffer,
        src_buf: &wgpu::Buffer,
        dst_buf: &wgpu::Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_fxaa_bind_group"),
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
        })
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
