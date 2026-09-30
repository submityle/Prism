//! `wgpu` compute twin of the virtual-geometry world-to-screen vertex
//! projection ([`project_vertex`](prism_render_architecture::virtual_geometry::project_vertex)).
//!
//! The vis-buffer software rasterizer transforms each cluster vertex world ->
//! clip -> ndc -> viewport pixels once, before the screen-space fill and the
//! per-triangle gradient setup consume the resulting [`ScreenVertex`]es. The CPU
//! golden [`project_vertex`](prism_render_architecture::virtual_geometry::project_vertex)
//! owns that transform — the column-major clip product, the perspective divide,
//! the y-flipping `ndc_to_uv` and the viewport scale;
//! [`GpuProjectVertex`] is the on-device twin that runs one thread per vertex
//! and returns the same `pos`/`depth` the reference derives, so the vis-buffer
//! path's per-vertex projection stage joins the already-landed screen-space
//! fill and gradient twins into a full device-side chain.
//!
//! # Culling convention
//!
//! The reference returns [`None`] for a vertex on or behind the camera plane
//! (`clip.w <= 0`), where the perspective divide is undefined. The twin mirrors
//! that: such vertices come back as [`None`] in the per-vertex output vector,
//! exactly the entries upstream near-plane culling is expected to exclude.
//!
//! # Portability
//!
//! The kernel is a fixed column-major matrix-vector product, one reciprocal and
//! a handful of affine multiply-adds in the portable core-`WGSL` subset, so it
//! needs no optional device feature and runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Correctness model
//!
//! For world positions and matrix entries on integer / dyadic coordinates each
//! product and partial sum in the clip transform is exactly representable, so
//! the clip value is identical on `CPU` and `GPU` regardless of
//! fused-multiply-add contraction. When `clip.w` is an exact power of two the
//! reciprocal is exact, the perspective divide is exact, and the affine
//! `ndc_to_uv` plus viewport scale stay exact for power-of-two viewports, so
//! the parity test asserts field-for-field equality rather than a tolerance.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard world -> clip -> ndc -> viewport vertex projection for
//! software rasterization plus `wgpu` compute dispatch; no Unreal Engine source
//! or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::virtual_geometry::ScreenVertex;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Uniform parameters for one projection dispatch. Layout matches `Params` in
/// `shaders/project_vertex.wesl`: the column-major clip matrix, the viewport
/// size, the vertex count then one pad word.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    clip_from_world: [[f32; 4]; 4],
    viewport: [f32; 2],
    vertex_count: u32,
    pad0: u32,
}

/// One world-space input vertex. `12`-byte stride, matching `Vertex` in the
/// shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVertex {
    x: f32,
    y: f32,
    z: f32,
}

/// One projected screen vertex. `16`-byte stride, matching `ScreenVertex` in
/// the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuScreenVertex {
    valid: u32,
    x: f32,
    y: f32,
    depth: f32,
}

/// A compiled, reusable world-to-screen vertex-projection pipeline.
pub struct GpuProjectVertex {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuProjectVertex {
    /// Compiles the projection kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so it needs no
    /// optional device feature.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_project_vertex"),
            source: ShaderSource::Wgsl(include_str!("../shaders/project_vertex.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_project_vertex_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_project_vertex_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_project_vertex_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("project"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuProjectVertex {
            module,
            layout,
            pipeline,
        }
    }

    /// Projects each world-space position in `world_positions` through
    /// `clip_from_world` into a [`ScreenVertex`], returning one entry per
    /// position in input order.
    ///
    /// `clip_from_world` is column-major, matching the CPU reference. The
    /// returned entry for position `p` equals
    /// [`project_vertex`](prism_render_architecture::virtual_geometry::project_vertex)`(clip_from_world, p, viewport)`:
    /// [`Some`] with the same `pos`/`depth` for a vertex in front of the camera,
    /// [`None`] for one on or behind the camera plane. An empty
    /// `world_positions` slice yields an empty result — storage buffers cannot
    /// be zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn project(
        &self,
        ctx: &GpuContext,
        clip_from_world: &[[f32; 4]; 4],
        world_positions: &[[f32; 3]],
        viewport: [f32; 2],
    ) -> Vec<Option<ScreenVertex>> {
        if world_positions.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let params = Params {
            clip_from_world: *clip_from_world,
            viewport,
            vertex_count: u32::try_from(world_positions.len())
                .expect("vertex count must fit in u32 for the GPU dispatch"),
            pad0: 0,
        };

        let gpu_vertices: Vec<GpuVertex> = world_positions
            .iter()
            .map(|&p| GpuVertex {
                x: p[0],
                y: p[1],
                z: p[2],
            })
            .collect();

        let out_bytes = (world_positions.len() as u64) * (size_of::<GpuScreenVertex>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_project_vertex_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });
        let vertices_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_project_vertex_vertices"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: BufferUsages::STORAGE,
        });
        let screen_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_project_vertex_screen"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let screen_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_project_vertex_screen_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_project_vertex_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: vertices_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: screen_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_project_vertex_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_project_vertex_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = params.vertex_count.div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&screen_buf, 0, &screen_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        screen_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = screen_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_screen = bytemuck::cast_slice::<u8, GpuScreenVertex>(&view).to_vec();
        drop(view);
        screen_stage.unmap();
        debug_assert_eq!(gpu_screen.len(), world_positions.len());
        gpu_screen
            .into_iter()
            .map(|s| {
                if s.valid == 0 {
                    None
                } else {
                    Some(ScreenVertex::new([s.x, s.y], s.depth))
                }
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
