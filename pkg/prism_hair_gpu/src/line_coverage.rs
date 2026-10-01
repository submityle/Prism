//! `wgpu` compute twin of Prism's analytic sub-pixel line-coverage kernel
//! ([`pixel_coverage`](prism_render_architecture::hair::line_coverage::pixel_coverage),
//! batch form
//! [`coverage_map`](prism_render_architecture::hair::line_coverage::coverage_map)).
//!
//! A hair fibre projected to screen is almost always thinner than one pixel, so
//! a point-sampled rasteriser either misses it or snaps it on and off between
//! frames (the classic shimmering-hair artefact). The film-grade fix (as used by
//! `UE5` Groom's hair rasteriser and the `Weta`/`Pixar` line-AA literature) is to
//! treat each strand segment as a width-carrying line and compute, in closed
//! form, the fraction of each pixel it covers; that coverage is the fibre's
//! anti-aliasing `alpha`. The reference models the pixel as a round coverage
//! kernel of radius `0.5` and the fibre as a capsule of half-width
//! `width * 0.5`, giving the monotone trapezoidal ramp
//! `clamp(width*0.5 + 0.5 - d, 0, 1)` where `d` is the point-to-segment distance
//! from the pixel centre.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairLineCoverage::eval`] takes one width-carrying [`ScreenSegment`] and a
//! batch of pixel centres and returns one coverage ratio per centre, preserving
//! input order — the array-in/array-out form used to shade a tile of pixels
//! against one fibre segment. The pixel index is the invocation id
//! (`@compute @workgroup_size(64)`, one-dimensional dispatch over
//! `global_invocation_id.x`); invocations past the pixel count early-return.
//!
//! # The segment lives on the host, the pixels vary per thread
//!
//! The segment endpoints and stroke width are passed as uniform parameters; the
//! per-thread variation is the pixel centre read from the storage input. The
//! width is sanitized in-shader bit-faithfully to the golden's
//! `sanitized_width`, so the host uploads the raw authored width unchanged.
//!
//! # Portability
//!
//! The kernel uses only multiply/add, `clamp`, `dot` and `length` (one `sqrt`),
//! with no `exp`, `pow`, `sin` or optional device feature, so the twin runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The distance is a `dot`/`sqrt` chain and the ramp a multiply-add a `GPU` may
//! fuse, so `CPU` and `GPU` agree to within the documented fma tolerance
//! (`abs_diff < 1e-4` or `rel_diff < 1e-3`) rather than bit-for-bit. The width
//! guard mirrors the golden's `sanitized_width` exactly (`w == w && w > 0.0 &&
//! w <= MAX_FINITE_F32` rejects NaN, non-positive and `+inf`), and the degenerate
//! zero-length segment collapses to the point distance just like the reference,
//! so negative/non-finite widths produce the same clamped coverage and the
//! result stays finite and in `[0, 1]` for every input.
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: analytic line-AA coverage (`UE5` Groom / `Weta` / `Pixar` line-AA
//! literature) plus `wgpu` compute dispatch; no Unreal Engine source or derived
//! code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::line_coverage::{pixel_coverage, ScreenSegment};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one coverage dispatch. Layout matches `Params` in
/// `shaders/line_coverage.wesl`: the segment packed as `(a.x, a.y, b.x, b.y)`,
/// the stroke width and the pixel count, in two `16`-byte uniform slots.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    segment: [f32; 4],
    width: f32,
    pixel_count: u32,
    pad0: u32,
    pad1: u32,
}

/// A compiled, reusable per-pixel line-coverage pipeline.
pub struct GpuHairLineCoverage {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairLineCoverage {
    /// Compiles the per-pixel line-coverage kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairLineCoverage {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_line_coverage"),
            source: ShaderSource::Wgsl(include_str!("../shaders/line_coverage.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_line_coverage_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_line_coverage_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_line_coverage_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairLineCoverage {
            module,
            layout,
            pipeline,
        }
    }

    /// Computes the analytic coverage of each pixel centre against `seg`,
    /// returning one ratio in `[0, 1]` per centre in input order.
    ///
    /// The coverage for pixel `i` equals the `CPU` golden
    /// [`pixel_coverage`](prism_render_architecture::hair::line_coverage::pixel_coverage)
    /// of `seg` at `centers[i]` to within the module's documented fma tolerance,
    /// with negative/non-finite stroke widths collapsing to the same clamped
    /// result. An empty batch yields an empty vector without a dispatch —
    /// storage buffers cannot be zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, seg: ScreenSegment, centers: &[[f32; 2]]) -> Vec<f32> {
        let pixel_count = centers.len();
        if pixel_count == 0 {
            return Vec::new();
        }

        let device = ctx.device();

        let uniforms = Params {
            segment: [seg.a[0], seg.a[1], seg.b[0], seg.b[1]],
            // The shader sanitizes the width itself (bit-faithfully to the
            // golden), so upload the raw authored value unchanged.
            width: seg.width,
            pixel_count: pixel_count as u32,
            pad0: 0,
            pad1: 0,
        };

        // Output is one f32 (4 bytes) per pixel.
        let out_bytes = (pixel_count as u64) * (size_of::<f32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_line_coverage_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let centers_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_line_coverage_centers"),
            contents: bytemuck::cast_slice(centers),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_line_coverage_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_line_coverage_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_line_coverage_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: centers_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_line_coverage_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_line_coverage_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (pixel_count as u32).div_ceil(64);
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

/// The `CPU` golden coverage for one pixel centre, re-exported so the parity
/// test can assert the device twin against the identical reference it mirrors.
#[must_use]
pub fn reference_coverage(seg: ScreenSegment, center: [f32; 2]) -> f32 {
    pixel_coverage(seg, center)
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
