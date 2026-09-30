//! `wgpu` compute twin of the low-resolution raymarch scheduling predicate
//! ([`active_pixel`](prism_render_architecture::volumetric::temporal::active_pixel)).
//!
//! Volumetric clouds `raymarch` at a fraction of the output resolution and the
//! full image is reconstructed over several frames (design section 10). This
//! predicate decides, deterministically, whether pixel `(x, y)` is `raymarch`ed
//! on frame `frame_index` under one of three
//! [`UpscaleMode`](prism_render_architecture::volumetric::temporal::UpscaleMode)
//! patterns:
//!
//! ```text
//! Full         : always active (period 1)
//! Checkerboard : ((x ^ y ^ frame_index) & 1) == 0 (period 2)
//! QuarterRes   : (x & 1) == (frame_index & 1)
//!                && (y & 1) == ((frame_index >> 1) & 1) (period 4)
//! ```
//!
//! Over one period the per-frame active sets partition, then union to, the
//! whole grid, so no pixel is ever starved. The `CPU` golden
//! [`active_pixel`](prism_render_architecture::volumetric::temporal::active_pixel)
//! owns that logic; [`GpuActivePixel`] is the on-device twin that runs one
//! thread per query and reproduces the same decision.
//!
//! # Correctness model
//!
//! The predicate is integer bit ops and comparisons, so the decision is exact —
//! there is no floating-point slack. The parity test asserts every decision
//! matches the `CPU` golden bit for bit and, over one period, that the active
//! sets cover the whole grid exactly once, so a degenerate kernel could not
//! pass.
//!
//! # Portability
//!
//! The kernel is integer bit ops and comparisons in the portable core-`WGSL`
//! subset — no `exp`, `pow` or optional device feature — so it runs unmodified
//! on Metal, Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard checkerboard / quarter-res upsampling schedule plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::volumetric::temporal::UpscaleMode;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// One scheduling query: the frame index, pixel coordinates and update mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivePixelQuery {
    /// The frame index whose active set is being evaluated.
    pub frame_index: u32,
    /// The pixel `x` coordinate.
    pub x: u32,
    /// The pixel `y` coordinate.
    pub y: u32,
    /// The low-resolution update pattern.
    pub mode: UpscaleMode,
}

/// Maps an [`UpscaleMode`] to the ordinal the shader expects.
fn mode_ordinal(mode: UpscaleMode) -> u32 {
    match mode {
        UpscaleMode::Full => 0,
        UpscaleMode::Checkerboard => 1,
        UpscaleMode::QuarterRes => 2,
    }
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/active_pixel.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    frame_index: u32,
    x: u32,
    y: u32,
    mode: u32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/active_pixel.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable active-pixel pipeline.
pub struct GpuActivePixel {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuActivePixel {
    /// Compiles the active-pixel kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuActivePixel {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_active_pixel"),
            source: ShaderSource::Wgsl(include_str!("../shaders/active_pixel.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_active_pixel_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_active_pixel_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_active_pixel_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("active_pixel_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuActivePixel {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the scheduling predicate for every query in `queries`,
    /// returning one boolean per query in input order (`true` = the pixel is
    /// `raymarch`ed on that frame).
    ///
    /// The returned decision for query `q` equals
    /// [`active_pixel`](prism_render_architecture::volumetric::temporal::active_pixel)`(q.frame_index, q.x, q.y, q.mode)`
    /// exactly. An empty `queries` slice yields an empty result — storage
    /// buffers cannot be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[ActivePixelQuery]) -> Vec<bool> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                frame_index: q.frame_index,
                x: q.x,
                y: q.y,
                mode: mode_ordinal(q.mode),
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<u32>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_active_pixel_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_active_pixel_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_active_pixel_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_active_pixel_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_active_pixel_bind_group"),
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
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_active_pixel_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_active_pixel_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (queries.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let flags = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(flags.len(), queries.len());
        flags.into_iter().map(|f| f != 0).collect()
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
