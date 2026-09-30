//! `wgpu` compute twin of the hair sub-pixel raster-path classifier
//! ([`classify_hair_raster`](prism_render_architecture::hair::raster::classify_hair_raster)).
//!
//! A render strand is usually a fraction of a pixel wide, so production hair
//! engines (UE5 Groom, AMD `TressFX`) rasterize thin strands in a compute
//! visibility pass and reserve the hardware triangle path for the rare near,
//! thick strand (design 6.2). The per-segment routing decision the backend
//! needs is `classify_hair_raster` in
//! [`prism_render_architecture::hair::raster`]; this crate is the on-device
//! twin that evaluates the same classification, one thread per segment, so a
//! passing real-device parity test is direct evidence the ported kernel routes
//! segments identically to the reference — not merely that its shader compiles.
//!
//! # What the kernel evaluates
//!
//! [`GpuHairRaster::eval`] classifies a batch of projected strand-segment
//! statistics under one set of [`HairRasterParams`]. Each [`RasterQuery`]
//! carries the segment's screen width, screen length and analytic coverage
//! (built from a golden
//! [`HairStrandSegmentStats`](prism_render_architecture::hair::raster::HairStrandSegmentStats)
//! with [`RasterQuery::from_stats`]). The kernel mirrors the reference's
//! branches exactly:
//!
//! * a non-positive screen length or a coverage below the clamped
//!   `min_coverage` is [`HairRasterPath::Culled`];
//! * a surviving segment at or below the clamped `software_width_threshold`
//!   takes [`HairRasterPath::SubpixelSoftware`];
//! * anything wider takes [`HairRasterPath::ThickHardware`].
//!
//! Both threshold clamps reproduce the reference's `.max(0.0)` guard, so a
//! negative tunable routes identically on both sides. The kernel emits the
//! resolved [`HairRasterPath`] as its integer discriminant; [`path_code`] maps
//! the golden enum to the same code for comparison.
//!
//! # Portability
//!
//! The classifier uses only `max` and floating-point comparisons in the
//! portable core-`WGSL` subset — no `exp`, `pow` or optional device feature —
//! so it runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! Unlike the geometry twins, the routing decision is pure comparison
//! arithmetic with no fused multiply-add and no transcendental call: the `CPU`
//! and `GPU` compare the identical raw `f32` inputs, so the emitted path code
//! is **bit-identical** and the parity test asserts exact integer equality
//! rather than a floating tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard analytic thin-strand raster routing plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::hair::raster::{
    HairRasterParams, HairRasterPath, HairStrandSegmentStats,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Path code for [`HairRasterPath::SubpixelSoftware`], matching
/// `PATH_SUBPIXEL_SOFTWARE` in `shaders/raster.wesl`.
pub const PATH_SUBPIXEL_SOFTWARE: u32 = 0;
/// Path code for [`HairRasterPath::ThickHardware`], matching
/// `PATH_THICK_HARDWARE` in `shaders/raster.wesl`.
pub const PATH_THICK_HARDWARE: u32 = 1;
/// Path code for [`HairRasterPath::Culled`], matching `PATH_CULLED` in
/// `shaders/raster.wesl`.
pub const PATH_CULLED: u32 = 2;

/// Maps a golden [`HairRasterPath`] to the integer discriminant the kernel
/// emits, so the CPU reference and the GPU readback compare directly.
#[must_use]
pub fn path_code(path: HairRasterPath) -> u32 {
    match path {
        HairRasterPath::SubpixelSoftware => PATH_SUBPIXEL_SOFTWARE,
        HairRasterPath::ThickHardware => PATH_THICK_HARDWARE,
        HairRasterPath::Culled => PATH_CULLED,
    }
}

/// One projected strand-segment statistic to classify.
///
/// The fields encode exactly the inputs the `CPU` golden
/// [`classify_hair_raster`](prism_render_architecture::hair::raster::classify_hair_raster)
/// consumes. The struct is `16`-byte `repr(C)` matching `Query` in
/// `shaders/raster.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct RasterQuery {
    /// Projected screen width of the strand at this segment, in pixels.
    screen_width_px: f32,
    /// Projected screen length of the segment along the strand, in pixels.
    screen_length_px: f32,
    /// Analytic sub-pixel coverage fraction contributed by the segment.
    coverage: f32,
    /// Padding to the `16`-byte stride.
    pad: f32,
}

impl RasterQuery {
    /// Builds a query from a golden
    /// [`HairStrandSegmentStats`](prism_render_architecture::hair::raster::HairStrandSegmentStats)
    /// so the twin classifies the exact statistics the `CPU` solver projects.
    #[must_use]
    pub fn from_stats(stats: HairStrandSegmentStats) -> Self {
        Self {
            screen_width_px: stats.screen_width_px,
            screen_length_px: stats.screen_length_px,
            coverage: stats.coverage,
            pad: 0.0,
        }
    }
}

/// Uniform parameters for one classification dispatch. Layout matches `Params`
/// in `shaders/raster.wesl`: the query count, one pad word, then the two
/// clamped thresholds filling the `16`-byte uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    min_coverage: f32,
    width_threshold: f32,
}

/// A compiled, reusable raster-path classification pipeline.
pub struct GpuHairRaster {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairRaster {
    /// Compiles the raster-path classification kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairRaster {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_raster"),
            source: ShaderSource::Wgsl(include_str!("../shaders/raster.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_raster_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_raster_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_raster_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairRaster {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies every segment under `params`, returning one path code per
    /// query in input order.
    ///
    /// The returned code for query `q` equals [`path_code`] applied to
    /// [`classify_hair_raster`](prism_render_architecture::hair::raster::classify_hair_raster)
    /// on the same statistics and `params` — exactly, since the routing is pure
    /// comparison arithmetic. An empty `queries` slice yields an empty result —
    /// storage buffers cannot be zero-sized, so it is handled by an early
    /// return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[RasterQuery],
        params: HairRasterParams,
    ) -> Vec<u32> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let uniforms = Params {
            count: queries.len() as u32,
            pad0: 0,
            min_coverage: params.min_coverage,
            width_threshold: params.software_width_threshold,
        };

        // One u32 path code per query.
        let out_bytes = (queries.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_raster_params"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_raster_queries"),
            contents: bytemuck::cast_slice(queries),
            usage: BufferUsages::STORAGE,
        });
        let values_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_raster_values"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let values_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_raster_values_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_raster_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: queries_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: values_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_raster_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_raster_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&values_buf, 0, &values_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        values_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = values_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let codes = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        values_stage.unmap();
        debug_assert_eq!(codes.len(), queries.len());
        codes
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
