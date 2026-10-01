//! `wgpu` compute twin of the bicubic texture-filtering reference
//! ([`bicubic_sample`](prism_render_architecture::particle::bicubic_sample),
//! design §8.4).
//!
//! Flipbook frames, decal footprints and streak trails are sampled at
//! fractional texel coordinates; a cubic reconstruction filter keeps that sprite
//! detail crisp without the ringing a naive sinc window introduces. The `CPU`
//! golden
//! [`SampleImage::sample_bicubic`](prism_render_architecture::particle::bicubic_sample::SampleImage::sample_bicubic)
//! owns that math — the generalized `Mitchell-Netravali` family whose
//! `Catmull-Rom` special case is `b = 0, c = 0.5` — and [`GpuBicubicSample`] is
//! the on-device twin that runs one thread per `(image, uv)` sample and
//! reproduces the same `4x4` point-tap footprint.
//!
//! # What is twinned
//!
//! The kernel reproduces the point-sampled `4x4` reconstruction path tap for
//! tap: the same two-piece cubic
//! [`cubic_weights`](prism_render_architecture::particle::bicubic_sample::cubic_weights)
//! polynomial, the same `floor`-based base-texel split, the same
//! `[-1, 0, 1, 2]` tap offsets and the same integer `clamp-to-edge` addressing
//! the reference
//! [`SampleImage::fetch`](prism_render_architecture::particle::bicubic_sample::SampleImage::fetch)
//! uses. [`GpuBicubicSample::sample_catmull_rom`] is the `b = 0, c = 0.5`
//! specialization derived on the host with no second kernel. The hardware
//! `bilinear` five-tap optimization
//! [`catmull_rom_5tap`](prism_render_architecture::particle::bicubic_sample::catmull_rom_5tap)
//! is intentionally out of scope: it relies on hardware `bilinear` fetches that
//! a storage-buffer compute path cannot reproduce bit for bit.
//!
//! # Correctness model
//!
//! Every weight is a piecewise cubic polynomial, so evaluation is nothing but
//! `+`, `-`, `*` and a division by the constant `6`; addressing is pure integer
//! `clamp`. `CPU` and `GPU` therefore evaluate the same closed form. They are
//! not bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few `ULP`. The parity test
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`), tight enough
//! to catch a genuinely wrong port (a swapped tap, a dropped offset, a missing
//! edge clamp) yet loose enough to admit legal fused multiply-add contraction.
//!
//! # Degenerate inputs
//!
//! An empty image (`width == 0` or `height == 0`) samples as transparent black,
//! matching the reference; the storage buffer still holds one placeholder texel
//! that the kernel never reads because the empty short-circuit fires first. An
//! empty coordinate batch short-circuits on the host with no dispatch (a storage
//! buffer cannot be zero-sized).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `clamp`, `floor`,
//! `+ - * /` and unsigned/`i32` index arithmetic — with no `sin`, `cos`, `exp`,
//! `pow`, optional device feature or `u64`, so it runs unmodified on `Metal`,
//! `Vulkan` and `DX12`. The two nested tap loops are fixed `4x4` iterations, so
//! the kernel provably terminates with no runaway loop.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::bicubic_sample`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::bicubic_sample::SampleImage;
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

/// Channels per `RGBA` texel in the flat storage-buffer layout.
const CHANNELS: usize = 4;

/// The bicubic reconstruction kernel, mirroring the `CPU` golden
/// [`SampleImage::sample_bicubic`](prism_render_architecture::particle::bicubic_sample::SampleImage::sample_bicubic).
/// One thread samples one `(image, uv)` pair: it builds the four separable
/// cubic tap weights per axis, gathers the `4x4` point footprint with integer
/// `clamp-to-edge` addressing and writes one `RGBA` sample. `image`, `coords`
/// and `dst` are flat `f32` arrays (`4`, `2` and `4` channels per element
/// respectively). Pure polynomial weights and integer addressing: no
/// transcendental, no intrinsic, no `u64`, portable on `Metal`, `Vulkan` and
/// `DX12`.
///
/// Provenance: 孪生自本仓 `prism_render_architecture::particle::bicubic_sample`；无第三方引擎源码或衍生代码。
const BICUBIC_SAMPLE_WGSL: &str = r#"
struct Params {
    // Image width in texels; zero marks an empty image sampled as black.
    width: u32,
    // Image height in texels; zero marks an empty image sampled as black.
    height: u32,
    // Number of valid (u, v) samples; threads past this short-circuit.
    sample_count: u32,
    // Padding to keep the following f32 pair 16-byte aligned.
    pad0: u32,
    // Mitchell-Netravali B parameter (Catmull-Rom is 0.0).
    b: f32,
    // Mitchell-Netravali C parameter (Catmull-Rom is 0.5).
    c: f32,
    // Padding to a 16-byte-aligned uniform struct.
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read> coords: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

// Generalized Mitchell-Netravali cubic kernel at a non-negative distance x (in
// texels). Two-piece cubic with support [0, 2) split at x = 1; pure polynomial
// in x divided by the constant 6, matching the golden `cubic_kernel`.
fn cubic_kernel(x: f32, b: f32, c: f32) -> f32 {
    if (x < 1.0) {
        let x2 = x * x;
        let x3 = x2 * x;
        return ((12.0 - 9.0 * b - 6.0 * c) * x3
            + (-18.0 + 12.0 * b + 6.0 * c) * x2
            + (6.0 - 2.0 * b)) / 6.0;
    } else if (x < 2.0) {
        let x2 = x * x;
        let x3 = x2 * x;
        return ((-b - 6.0 * c) * x3
            + (6.0 * b + 30.0 * c) * x2
            + (-12.0 * b - 48.0 * c) * x
            + (8.0 * b + 24.0 * c)) / 6.0;
    }
    return 0.0;
}

// Four separable cubic tap weights for a fractional position t in [0, 1],
// ordered far-left tap to far-right tap, matching the golden `cubic_weights`.
// Inputs outside [0, 1] are clamped onto the kernel support.
fn cubic_weights(t: f32, b: f32, c: f32) -> array<f32, 4> {
    let tc = clamp(t, 0.0, 1.0);
    return array<f32, 4>(
        cubic_kernel(1.0 + tc, b, c),
        cubic_kernel(tc, b, c),
        cubic_kernel(1.0 - tc, b, c),
        cubic_kernel(2.0 - tc, b, c),
    );
}

// Fetch texel (x, y) with integer clamp-to-edge addressing, matching the golden
// `SampleImage::fetch`. The empty-image case is handled by the caller before
// this is ever reached, so width and height are at least one here.
fn fetch(x: i32, y: i32) -> vec4<f32> {
    let w = i32(params.width);
    let h = i32(params.height);
    let cx = clamp(x, 0, w - 1);
    let cy = clamp(y, 0, h - 1);
    let base = u32((cy * w + cx) * 4);
    return vec4<f32>(image[base], image[base + 1u], image[base + 2u], image[base + 3u]);
}

@compute @workgroup_size(64)
fn sample_bicubic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.sample_count) {
        return;
    }
    let out_base = idx * 4u;
    if (params.width == 0u || params.height == 0u) {
        dst[out_base] = 0.0;
        dst[out_base + 1u] = 0.0;
        dst[out_base + 2u] = 0.0;
        dst[out_base + 3u] = 0.0;
        return;
    }
    let u = coords[idx * 2u];
    let v = coords[idx * 2u + 1u];
    let fu = floor(u);
    let fv = floor(v);
    let ix0 = i32(fu);
    let iy0 = i32(fv);
    let tx = u - fu;
    let ty = v - fv;
    var wx = cubic_weights(tx, params.b, params.c);
    var wy = cubic_weights(ty, params.b, params.c);
    // Tap offsets relative to the base texel, matching `cubic_weights`: index 0
    // weights the texel one step before the base, so offset = i - 1.
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var j: i32 = 0; j < 4; j = j + 1) {
        let sy = iy0 + (j - 1);
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let texel = fetch(ix0 + (i - 1), sy);
            let weight = wx[i] * wy[j];
            acc = acc + texel * weight;
        }
    }
    dst[out_base] = acc.x;
    dst[out_base + 1u] = acc.y;
    dst[out_base + 2u] = acc.z;
    dst[out_base + 3u] = acc.w;
}
"#;

/// Uniform parameters for one dispatch: the image dimensions, the sample count
/// and the `Mitchell-Netravali` `b`/`c` parameters, padded to a `16`-byte,
/// `std140`-aligned uniform struct matching `Params` in [`BICUBIC_SAMPLE_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Image width in texels.
    width: u32,
    /// Image height in texels.
    height: u32,
    /// Number of valid `(u, v)` samples in the input and output buffers.
    sample_count: u32,
    /// Padding word.
    pad0: u32,
    /// `Mitchell-Netravali` `B` parameter.
    b: f32,
    /// `Mitchell-Netravali` `C` parameter.
    c: f32,
    /// Padding word.
    pad1: f32,
    /// Padding word.
    pad2: f32,
}

/// A compiled, reusable bicubic reconstruction kernel, twinning the `CPU` golden
/// [`bicubic_sample`](prism_render_architecture::particle::bicubic_sample)
/// `4x4` point-sampled reconstruction path.
pub struct GpuBicubicSample {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuBicubicSample {
    /// Compiles the bicubic reconstruction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is requested and compilation succeeds on any `Metal`,
    /// `Vulkan` or `DX12` backend.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_bicubic_sample_module"),
            source: ShaderSource::Wgsl(BICUBIC_SAMPLE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_bicubic_sample_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: true }),
                buffer_entry(3, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_bicubic_sample_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_bicubic_sample_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sample_bicubic"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuBicubicSample {
            module,
            layout,
            pipeline,
        }
    }

    /// Samples `image` at every `(u, v)` texel-space coordinate in `coords` with
    /// the generalized `Mitchell-Netravali` cubic filter of parameters `b` and
    /// `c`, mirroring the golden
    /// [`SampleImage::sample_bicubic`](prism_render_architecture::particle::bicubic_sample::SampleImage::sample_bicubic).
    ///
    /// Returns one `RGBA` sample per coordinate, in order. An empty coordinate
    /// batch returns an empty vector with no dispatch issued (a storage buffer
    /// cannot be zero-sized). An empty image yields all transparent-black
    /// samples.
    #[must_use]
    pub fn sample_bicubic(
        &self,
        ctx: &GpuContext,
        image: &SampleImage,
        coords: &[[f32; 2]],
        b: f32,
        c: f32,
    ) -> Vec<[f32; 4]> {
        self.dispatch(ctx, image, coords, b, c)
    }

    /// Samples `image` at every `(u, v)` texel-space coordinate in `coords` with
    /// the interpolating `Catmull-Rom` filter, mirroring the golden
    /// [`SampleImage::sample_catmull_rom`](prism_render_architecture::particle::bicubic_sample::SampleImage::sample_catmull_rom).
    ///
    /// This is the convenience specialization of [`GpuBicubicSample::sample_bicubic`]
    /// at `b = 0, c = 0.5`; it issues the same single kernel. Returns one `RGBA`
    /// sample per coordinate, in order, and an empty vector for an empty batch.
    #[must_use]
    pub fn sample_catmull_rom(
        &self,
        ctx: &GpuContext,
        image: &SampleImage,
        coords: &[[f32; 2]],
    ) -> Vec<[f32; 4]> {
        self.dispatch(ctx, image, coords, 0.0, 0.5)
    }

    /// Issues one `1-D` dispatch of the kernel, sampling `image` at every
    /// coordinate in `coords` and reading the `RGBA` samples back. Empty
    /// coordinate batches short-circuit without a dispatch because a storage
    /// buffer cannot be zero-sized; an empty image is uploaded as a single
    /// placeholder texel with `width`/`height` pinned to `0` so the kernel never
    /// reads it.
    fn dispatch(
        &self,
        ctx: &GpuContext,
        image: &SampleImage,
        coords: &[[f32; 2]],
        b: f32,
        c: f32,
    ) -> Vec<[f32; 4]> {
        let sample_count = coords.len();
        if sample_count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let params = GpuParams {
            width: image.width,
            height: image.height,
            sample_count: sample_count as u32,
            pad0: 0,
            b,
            c,
            pad1: 0.0,
            pad2: 0.0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bicubic_sample_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        // Flatten the image to a row-major, 4-per-texel f32 array. A storage
        // buffer cannot be zero-sized; for an empty image upload one placeholder
        // texel the kernel never reads because width/height are 0.
        let flat_image = flatten_image(image);
        let placeholder = [0.0f32; CHANNELS];
        let image_contents: &[f32] = if flat_image.is_empty() {
            &placeholder
        } else {
            &flat_image
        };
        let image_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bicubic_sample_image"),
            contents: bytemuck::cast_slice(image_contents),
            usage: BufferUsages::STORAGE,
        });

        let mut flat_coords = Vec::with_capacity(sample_count * 2);
        for uv in coords {
            flat_coords.extend_from_slice(uv);
        }
        let coords_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_bicubic_sample_coords"),
            contents: bytemuck::cast_slice(&flat_coords),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (sample_count * CHANNELS * size_of::<f32>()) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bicubic_sample_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_bicubic_sample_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: image_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: coords_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_bicubic_sample_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_bicubic_sample_encoder"),
        });
        {
            let groups = (sample_count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_bicubic_sample_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per sample, flattened to a 1-D dispatch.
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
        let flat = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        stage.unmap();

        let mut out = Vec::with_capacity(sample_count);
        for chunk in flat.chunks_exact(CHANNELS) {
            out.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        out
    }
}

/// Flattens a [`SampleImage`] into the row-major, `4`-per-texel `f32` layout the
/// kernel's `image` storage buffer expects.
fn flatten_image(image: &SampleImage) -> Vec<f32> {
    let mut data = Vec::with_capacity(image.data.len() * CHANNELS);
    for texel in &image.data {
        data.extend_from_slice(texel);
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
