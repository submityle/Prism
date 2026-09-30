//! `wgpu` compute twin of the cloud screen-space motion-vector composition
//! ([`composite_motion_vector`](prism_render_architecture::volumetric::temporal::composite_motion_vector)).
//!
//! The temporal reprojection (design section 10) drives the cloud history
//! resample with a screen-space motion vector that is the pure sum of the
//! wind-driven cloud advection velocity and the camera-induced screen motion:
//!
//! ```text
//! result = cloud_advection + camera_motion
//! ```
//!
//! Keeping it a pure [`Vec2`](prism_render_architecture::volumetric::Vec2) sum
//! makes the reprojection deterministic and trivially testable. The `CPU`
//! golden
//! [`composite_motion_vector`](prism_render_architecture::volumetric::temporal::composite_motion_vector)
//! owns that algebra; [`GpuCompositeMotionVector`] is the on-device twin that
//! runs one thread per query and reproduces the same vector.
//!
//! # Correctness model
//!
//! The kernel evaluates the same closed-form vector add as the `CPU`, so the
//! only slack is a legal multiply-add contraction of a few `ULP`. The parity
//! test asserts a tight tolerance (`abs_diff < 1e-6` or `rel_diff < 1e-5`) on
//! each component, so a degenerate kernel could not pass.
//!
//! # Portability
//!
//! The kernel is a single `vec2` add in the portable core-`WGSL` subset — no
//! `exp`, `pow` or optional device feature — so it runs unmodified on Metal,
//! Vulkan and DX12.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard screen-space motion-vector composition plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.
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

/// One motion-vector query: the cloud advection velocity and the camera-induced
/// screen motion, each a screen-space 2-D vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompositeMotionVectorQuery {
    /// Wind-driven cloud drift projected to screen space, `x` component.
    pub advection_x: f32,
    /// Wind-driven cloud drift projected to screen space, `y` component.
    pub advection_y: f32,
    /// Camera-induced screen motion, `x` component.
    pub camera_x: f32,
    /// Camera-induced screen motion, `y` component.
    pub camera_y: f32,
}

/// One composited screen-space motion vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionVector {
    /// The composited screen-space motion, `x` component.
    pub x: f32,
    /// The composited screen-space motion, `y` component.
    pub y: f32,
}

/// One query as uploaded. `16`-byte `repr(C)` matching `Query` in
/// `shaders/composite_motion_vector.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    advection_x: f32,
    advection_y: f32,
    camera_x: f32,
    camera_y: f32,
}

/// One result as read back. `16`-byte stride matching `results` in the shader
/// (the motion `xy` plus two pad words).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuMotion {
    x: f32,
    y: f32,
    pad0: f32,
    pad1: f32,
}

/// Uniform parameters for one dispatch. Layout matches `Params` in
/// `shaders/composite_motion_vector.wesl`: the query count plus three pad words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// A compiled, reusable motion-vector composition pipeline.
pub struct GpuCompositeMotionVector {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuCompositeMotionVector {
    /// Compiles the motion-vector composition kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuCompositeMotionVector {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_composite_motion_vector"),
            source: ShaderSource::Wgsl(
                include_str!("../shaders/composite_motion_vector.wesl").into(),
            ),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_composite_motion_vector_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_composite_motion_vector_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_composite_motion_vector_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("composite_motion_vector_main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuCompositeMotionVector {
            module,
            layout,
            pipeline,
        }
    }

    /// Evaluates the motion-vector composition for every query in `queries`,
    /// returning one vector per query in input order.
    ///
    /// The returned vector for query `q` equals
    /// [`composite_motion_vector`](prism_render_architecture::volumetric::temporal::composite_motion_vector)`(advection, camera)`
    /// to within the tolerance documented on this module. An empty `queries`
    /// slice yields an empty result — storage buffers cannot be zero-sized, so
    /// it is handled by an early return.
    #[must_use]
    pub fn eval(
        &self,
        ctx: &GpuContext,
        queries: &[CompositeMotionVectorQuery],
    ) -> Vec<MotionVector> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();

        let gpu_queries: Vec<GpuQuery> = queries
            .iter()
            .map(|q| GpuQuery {
                advection_x: q.advection_x,
                advection_y: q.advection_y,
                camera_x: q.camera_x,
                camera_y: q.camera_y,
            })
            .collect();

        let gpu_params = Params {
            count: queries.len() as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (queries.len() as u64) * (size_of::<GpuMotion>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_composite_motion_vector_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let queries_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_composite_motion_vector_queries"),
            contents: bytemuck::cast_slice(&gpu_queries),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_composite_motion_vector_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_composite_motion_vector_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_composite_motion_vector_bind_group"),
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
            label: Some("prism_volumetric_composite_motion_vector_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_composite_motion_vector_pass"),
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
        let gpu_results = bytemuck::cast_slice::<u8, GpuMotion>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), queries.len());
        gpu_results
            .into_iter()
            .map(|m| MotionVector { x: m.x, y: m.y })
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
