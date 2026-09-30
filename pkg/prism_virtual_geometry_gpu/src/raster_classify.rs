//! `wgpu` compute twin of the virtual-geometry raster-path selector
//! ([`select_raster_path`](prism_render_architecture::virtual_geometry::select_raster_path)).
//!
//! Before a GPU-driven virtual-geometry pipeline bins its draws it must decide,
//! per cluster, *how* to rasterize it: sub-pixel clusters are cheapest through
//! a compute software rasterizer, larger clusters prefer mesh shaders, then
//! hardware indirect draws, then a plain mesh fallback on capability-poor
//! backends. The CPU golden
//! [`select_raster_path`](prism_render_architecture::virtual_geometry::select_raster_path)
//! owns that decision; [`GpuRasterClassifier`] is the on-device twin that runs
//! one thread per cluster and returns the same
//! [`GeometryRasterPath`](prism_render_architecture::virtual_geometry::GeometryRasterPath)
//! index the reference does.
//!
//! # Portability
//!
//! The kernel is integer branching plus a single `<=` comparison of the
//! per-cluster `max_triangle_pixels` against the clamped threshold, in the
//! portable core-WGSL subset, so unlike the 64-bit payload twin it needs no
//! optional device feature and runs unmodified on Metal, Vulkan and DX12.
//!
//! # Correctness model
//!
//! The decision performs no floating-point arithmetic: the only float
//! operations are `max(threshold, 0.0)` and a single `<=` compare, evaluated on
//! identical operands on both sides, so every emitted path index is bit-exact
//! against the reference regardless of fused-multiply-add contraction. The
//! parity test asserts index-for-index equality rather than a tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard cluster raster-path selection heuristic and `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::{ClusterRasterStats, RasterCapability};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Capability bit for an available mesh/amplification shader pipeline. Matches
/// `CAP_MESH_SHADER` in `shaders/raster_path_classify.wesl`.
const CAP_MESH_SHADER: u32 = 1;
/// Capability bit for available hardware indirect (multi-)draw. Matches
/// `CAP_HARDWARE_INDIRECT` in `shaders/raster_path_classify.wesl`.
const CAP_HARDWARE_INDIRECT: u32 = 2;

/// Uniform parameters for one classify dispatch. Layout matches `Params` in
/// `shaders/raster_path_classify.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    capability_bits: u32,
    threshold: f32,
    pad0: u32,
}

/// One cluster's raster statistics. `8`-byte stride, matching `ClusterStats` in
/// the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuClusterStats {
    max_triangle_pixels: f32,
    triangle_count: u32,
}

/// A compiled, reusable raster-path classify pipeline.
pub struct GpuRasterClassifier {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuRasterClassifier {
    /// Compiles the classify kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-WGSL subset, so unlike
    /// [`GpuPayloadRaster::new`](crate::GpuPayloadRaster::new) this never
    /// returns [`None`]: it compiles on any backend `ctx` acquired.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuRasterClassifier {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_raster_path_classify"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/raster_path_classify.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_raster_path_classify_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_raster_path_classify_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_raster_path_classify_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("classify"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuRasterClassifier {
            module,
            layout,
            pipeline,
        }
    }

    /// Classifies each cluster in `stats` under `capability` and
    /// `software_pixel_threshold`, returning one path index per cluster in
    /// input order.
    ///
    /// The returned `u32` equals
    /// [`select_raster_path`](prism_render_architecture::virtual_geometry::select_raster_path)`(..) as u32`
    /// for the same inputs: `0` = `MeshShader`, `1` = `ComputeSoftware`,
    /// `2` = `IndirectHardware`, `3` = `FallbackMesh`.
    #[must_use]
    pub fn classify(
        &self,
        ctx: &GpuContext,
        stats: &[ClusterRasterStats],
        capability: RasterCapability,
        software_pixel_threshold: f32,
    ) -> Vec<u32> {
        if stats.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let mut capability_bits = 0u32;
        if capability.mesh_shader {
            capability_bits |= CAP_MESH_SHADER;
        }
        if capability.hardware_indirect {
            capability_bits |= CAP_HARDWARE_INDIRECT;
        }

        let params = Params {
            count: stats.len() as u32,
            capability_bits,
            threshold: software_pixel_threshold,
            pad0: 0,
        };

        let gpu_stats: Vec<GpuClusterStats> = stats
            .iter()
            .map(|s| GpuClusterStats {
                max_triangle_pixels: s.max_triangle_pixels,
                triangle_count: s.triangle_count,
            })
            .collect();

        let out_bytes = (stats.len() as u64) * 4;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_raster_path_classify_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let stats_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_raster_path_classify_stats"),
            contents: bytemuck::cast_slice(&gpu_stats),
            usage: BufferUsages::STORAGE,
        });
        let paths_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_raster_path_classify_paths"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let paths_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_raster_path_classify_paths_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_raster_path_classify_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: stats_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: paths_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_raster_path_classify_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_raster_path_classify_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (stats.len() as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&paths_buf, 0, &paths_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        paths_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = paths_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let paths = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        paths_stage.unmap();
        debug_assert_eq!(paths.len(), stats.len());
        paths
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
