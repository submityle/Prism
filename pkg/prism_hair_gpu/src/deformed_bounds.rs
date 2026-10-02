//! `wgpu` compute twin of Prism's hair sim -> render-graph scene-bounds fold
//! ([`deformed_bounds`](prism_render_architecture::hair::gpu_scene_handoff::deformed_bounds)).
//!
//! Once a frame's solve + resolve has produced a fresh set of world-space
//! deformed render points, the groom's instance in the shared
//! [`gpu_scene`](prism_render_architecture::gpu_scene) must be told its
//! world-space bounds moved so the incremental uploader re-publishes the bounds
//! row. This crate folds the deformed point cloud into that one
//! [`SceneBounds`](prism_render_architecture::gpu_scene::SceneBounds): the
//! per-axis `min` / `max` over every finite point, then the `center`,
//! `half_extents` and bounding `radius` (half-diagonal) the row carries.
//!
//! # Reduction shape
//!
//! Like [`GpuHairAnalysisReduce`](crate::analysis_reduce), this is a
//! many-inputs-to-one-output fold rather than a one-thread-per-output map: a
//! single `256`-wide workgroup grid-strides the point cloud into private
//! per-axis `min` / `max` partials (plus a `seen` flag so an invocation that
//! folded no finite point contributes nothing), then a shared-memory tree fold
//! collapses them to lane `0`, which derives and writes the box.
//!
//! # Correctness model
//!
//! A point with any non-finite component is skipped bit-faithfully to the
//! golden, so a stray `NaN` / `+/-inf` from a diverged solve cannot poison the
//! box. The `min` / `max` fold selects existing finite coordinates with no
//! arithmetic, so it is bit-exact regardless of fold order; `center`,
//! `half_extents` and `radius` are the same `(min +/- max) * 0.5` and
//! `sqrt(dot(h, h))` the golden computes (the parity test allows a small
//! tolerance for the `radius` square root). An empty groom or all-degenerate
//! input yields a zero-extent box at the origin, exactly as the reference does.
//!
//! # Portability
//!
//! The kernel uses only comparisons, add / multiply, `sqrt` and workgroup shared
//! memory in the portable core-`WGSL` subset -- no `exp`, `pow`, atomics or
//! optional device feature -- so the twin runs unmodified on Metal, Vulkan and
//! DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard single-workgroup shared-memory tree reduction plus
//! `wgpu` compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::gpu_scene::SceneBounds;
use prism_render_architecture::hair::gpu_scene_handoff::deformed_bounds;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of `f32` the kernel writes: `[center.xyz, radius, half_extents.xyz,
/// pad]`, mirroring the `SceneBounds` field order.
const BOUNDS_FLOATS: usize = 8;

/// Uniform parameters. `16`-byte scalar-packed `repr(C)` matching `Params` in
/// `shaders/deformed_bounds.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    point_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable deformed-bounds reduction pipeline.
pub struct GpuHairDeformedBounds {
    #[expect(
        dead_code,
        reason = "the shader module must outlive the pipeline that borrows it"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuHairDeformedBounds {
    /// Compiles the deformed-bounds reduction kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset plus workgroup
    /// shared memory, so no optional device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuHairDeformedBounds {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_hair_deformed_bounds"),
            source: ShaderSource::Wgsl(include_str!("../shaders/deformed_bounds.wesl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_hair_deformed_bounds_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_hair_deformed_bounds_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_hair_deformed_bounds_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuHairDeformedBounds {
            module,
            layout,
            pipeline,
        }
    }

    /// Folds a groom's deformed render `points` into one world-space
    /// [`SceneBounds`].
    ///
    /// The result equals
    /// [`deformed_bounds`](prism_render_architecture::hair::gpu_scene_handoff::deformed_bounds)
    /// to within the tolerance documented on this module (bit-exact selection
    /// of the `min` / `max` coordinates; a small tolerance only on the `radius`
    /// square root). An empty `points` slice returns the golden's zero-extent
    /// origin box without touching the device (storage buffers cannot be
    /// zero-sized).
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, points: &[[f32; 3]]) -> SceneBounds {
        // Empty groom: nothing to fold; return the golden's empty result
        // without a dispatch.
        if points.is_empty() {
            return deformed_bounds(points);
        }

        let point_count = points.len();
        let mut flat: Vec<f32> = Vec::with_capacity(point_count * 3);
        for p in points {
            flat.push(p[0]);
            flat.push(p[1]);
            flat.push(p[2]);
        }

        let uniform = Params {
            point_count: point_count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let device = ctx.device();
        let out_bytes = (BOUNDS_FLOATS * size_of::<f32>()) as u64;

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deformed_bounds_params"),
            contents: bytemuck::bytes_of(&uniform),
            usage: BufferUsages::UNIFORM,
        });
        let points_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_hair_deformed_bounds_points"),
            contents: bytemuck::cast_slice(&flat),
            usage: BufferUsages::STORAGE,
        });
        let out_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deformed_bounds_out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_hair_deformed_bounds_out_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_hair_deformed_bounds_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: points_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_hair_deformed_bounds_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_hair_deformed_bounds_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup cooperatively reduces the whole point cloud.
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &out_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        out_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();

        read_bounds(&out_stage)
    }
}

/// Runs the golden reduction directly; a thin re-export so the parity test can
/// name one reference path.
#[must_use]
pub fn reference_deformed_bounds(points: &[[f32; 3]]) -> SceneBounds {
    deformed_bounds(points)
}

/// Reads the eight packed `f32` back from a staging buffer and rebuilds the
/// [`SceneBounds`] row, then unmaps.
fn read_bounds(stage: &wgpu::Buffer) -> SceneBounds {
    let view = stage
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range should be available after poll");
    let values = bytemuck::cast_slice::<u8, f32>(&view);
    let bounds = SceneBounds {
        center: [values[0], values[1], values[2]],
        radius: values[3],
        half_extents: [values[4], values[5], values[6]],
        _padding: values[7],
    };
    drop(view);
    stage.unmap();
    bounds
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
