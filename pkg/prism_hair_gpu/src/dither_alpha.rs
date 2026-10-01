//! `wgpu` compute twin of Prism's blue-noise dither-`alpha` kernel
//! ([`dither_alpha_map`](prism_render_architecture::hair::reactive_mask::dither_alpha_map),
//! built from
//! [`dither_threshold`](prism_render_architecture::hair::reactive_mask::dither_threshold)
//! and
//! [`dither_alpha`](prism_render_architecture::hair::reactive_mask::dither_alpha)).
//!
//! A hair fibre thinner than the analytic `line_coverage` ramp can resolve is
//! instead drawn stochastically: each pixel is given a deterministic blue-noise
//! threshold in `[0, 1)` and the fibre is drawn (`alpha = 1`) exactly when its
//! sub-pixel coverage reaches that threshold, otherwise skipped (`alpha = 0`).
//! Spreading the hard draw decisions over a spatially varying, temporally
//! rotated threshold field reconstructs the correct average coverage without the
//! shimmering of a point-sampled on/off test — the ordered-dither / blue-noise
//! fallback used for the thinnest fibres (`§8.5` item 12 of the design).
//!
//! # What the kernel evaluates
//!
//! [`GpuHairDitherAlpha::eval`] takes a span of sub-pixel `coverages` plus the
//! tile origin `(base_x, base_y)` and the `frame`, and returns one hard draw
//! decision per pixel, preserving input order. Pixel `i` derives its threshold
//! from `(base_x + i, base_y, frame)`, so the batch is the array-in/array-out
//! form used to dither one scanline of a tile against its own threshold field.
//! The pixel index is the invocation id (`@compute @workgroup_size(64)`,
//! one-dimensional dispatch over `global_invocation_id.x`); invocations past the
//! element count early-return.
//!
//! # The tile origin lives on the host, the coverage varies per thread
//!
//! The origin and `frame` are passed as uniform parameters; the per-thread
//! variation is the coverage read from the storage input. The coverage is
//! sanitised in-shader bit-faithfully to the golden, so the host uploads the raw
//! authored value unchanged.
//!
//! # Portability
//!
//! The threshold is pure 32-bit integer arithmetic (`wrapping` multiply,
//! xor-shift) plus a single `u32`->`f32` conversion and an exact divide by
//! `2^32`; the draw is a `clamp` and a compare. No `exp`, `pow`, `sin` or
//! optional device feature is used, so the twin runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Correctness model
//!
//! Unlike every prior twin this kernel is checked **bit-exact**, not within an
//! fma tolerance: `WGSL` unsigned integer arithmetic wraps on overflow exactly
//! like Rust's `wrapping_mul` / `wrapping_add`, the `u32`->`f32` conversion and
//! the divide by `2^32` are exact, and the `alpha` is a hard `0.0`/`1.0`
//! decision. The parity test therefore compares the raw bit patterns
//! ([`f32::to_bits`]) rather than an approximate difference, and asserts every
//! output is exactly `0.0` or `1.0`. The coverage/threshold sanitisers mirror
//! the golden (`is_finite` -> `clamp [0, 1]`; a non-finite coverage collapses to
//! `0`, a non-finite threshold to `1` so a corrupt threshold suppresses the
//! fibre), so non-finite inputs produce the identical decision and the result
//! stays in `{0.0, 1.0}` for every input.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: ordered-dither / blue-noise sub-pixel fallback plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::reactive_mask::dither_alpha_map;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one dither dispatch. Layout matches `Params` in
/// `shaders/dither_alpha.wesl`: the tile origin, the `frame` and the element
/// count packed into a single `16`-byte uniform slot (four `u32`s, no padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    base_x: u32,
    base_y: u32,
    frame: u32,
    element_count: u32,
}

/// A compiled, reusable per-pixel dither-`alpha` pipeline.
pub struct GpuHairDitherAlpha {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairDitherAlpha {
    /// Compiles the per-pixel dither-`alpha` kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset (integer arithmetic
    /// plus `clamp`), so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairDitherAlpha {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_dither_alpha"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dither_alpha.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_dither_alpha_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_dither_alpha_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_dither_alpha_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairDitherAlpha {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes the hard dither draw decision for each coverage in `coverages`,
    /// returning one `alpha` in `{0.0, 1.0}` per pixel in input order.
    ///
    /// The decision for pixel `i` equals the `CPU` golden
    /// [`dither_alpha_map`](prism_render_architecture::hair::reactive_mask::dither_alpha_map)
    /// of `coverages` at `(base_x, base_y, frame)` **bit-for-bit** (the threshold
    /// is pure integer arithmetic), with non-finite coverages sanitised to the
    /// same decision. An empty batch yields an empty vector without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        coverages: &[f32],
        base_x: u32,
        base_y: u32,
        frame: u32,
    ) -> Vec<f32> {
        let element_count = coverages.len();
        if element_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            base_x,
            base_y,
            frame,
            element_count: element_count as u32,
        };

        // Output is one f32 (4 bytes) per pixel.
        let out_bytes = (element_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_dither_alpha_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let coverages_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_dither_alpha_coverages"),
            // Scalar f32 array; the shader sanitises each value bit-faithfully to
            // the golden, so upload the raw authored coverages unchanged.
            contents: bytemuck::cast_slice(coverages),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_dither_alpha_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_dither_alpha_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_dither_alpha_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: coverages_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_dither_alpha_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_dither_alpha_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (element_count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = out_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        drop(view);
        out_stage.unmap();

        out
    }
}

/// The `CPU` golden dither map, re-exported so the parity test can assert the
/// device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_dither_alpha_map(
    coverages: &[f32],
    base_x: u32,
    base_y: u32,
    frame: u32,
) -> Vec<f32> {
    dither_alpha_map(coverages, base_x, base_y, frame)
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
