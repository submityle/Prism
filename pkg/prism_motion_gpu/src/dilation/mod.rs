//! Host orchestration of the closest-depth motion-vector dilation kernel.
//!
//! [`GpuDilate`] compiles `shaders/dilate_closest_depth.wgsl` once and runs the
//! per-pixel neighborhood gather that mirrors the CPU golden
//! [`dilate_closest_depth`](prism_render_architecture::motion::dilation::dilate_closest_depth).
//! One invocation produces one output pixel by scanning its clamped square
//! neighborhood in the identical row-major order and adopting the velocity of
//! the strictly-nearest neighbor under the active
//! [`DepthOrder`](prism_render_architecture::motion::dilation::DepthOrder).
//!
//! Because the kernel only compares depths and copies whole velocity vectors —
//! it never does arithmetic on a velocity — the device output equals the golden
//! bit-for-bit; the parity tests compare against the golden directly with exact
//! equality rather than a tolerance.
//!
//! Standard `wgpu` compute orchestration. No Unreal Engine source or derived
//! code.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::motion::Vec2;
use prism_render_architecture::motion::dilation::{DepthField, DepthOrder, VelocityField};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::buffer;
use crate::context::GpuContext;

/// Compute workgroup edge length (the kernel is `@workgroup_size(8, 8, 1)`).
pub const TILE: u32 = 8;

/// `Pod` mirror of [`Vec2`] matching the `vec2<f32>` storage layout in the
/// shader (two tightly-packed `f32`s, 8-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVec2 {
    x: f32,
    y: f32,
}

impl GpuVec2 {
    fn from_vec2(v: Vec2) -> GpuVec2 {
        GpuVec2 { x: v.x, y: v.y }
    }

    fn to_vec2(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
}

/// Uniform block shared with `Params` in `dilate_closest_depth.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    radius: u32,
    order_flag: u32,
}

/// Maps a [`DepthOrder`] to the shader's `order_flag` (0 = `SmallerIsCloser`).
fn order_flag(order: DepthOrder) -> u32 {
    match order {
        DepthOrder::SmallerIsCloser => 0,
        DepthOrder::LargerIsCloser => 1,
    }
}

/// Flattens a [`DepthField`] back into its row-major buffer via the public
/// accessor, so the twin never reaches into the golden's private storage.
fn depth_to_row_major(depth: &DepthField) -> Vec<f32> {
    let width = depth.width();
    let height = depth.height();
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            out.push(depth.get(x, y).expect("in-bounds by construction"));
        }
    }
    out
}

/// Compiled dilation pipeline and its bind-group layout.
pub struct GpuDilate {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

impl GpuDilate {
    /// Compiles the closest-depth dilation kernel on `ctx`'s device.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuDilate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_motion_dilate"),
            source: ShaderSource::Wgsl(include_str!("../shaders/dilate_closest_depth.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_motion_dilate_layout"),
            entries: &[
                buffer_layout(0, BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                buffer_layout(1, BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                buffer_layout(2, BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                buffer_layout(3, BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_motion_dilate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_motion_dilate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuDilate { pipeline, layout }
    }

    /// Dilates `velocity` on the device, returning a new field whose every pixel
    /// holds the velocity of its strictly-nearest neighbor (by depth) within
    /// `radius`.
    ///
    /// Returns [`None`] when `velocity` and `depth` disagree on dimensions,
    /// matching the golden contract. A `radius` of `0` copies the input.
    #[must_use]
    pub fn dilate(
        &self,
        ctx: &GpuContext,
        velocity: &VelocityField,
        depth: &DepthField,
        radius: usize,
        order: DepthOrder,
    ) -> Option<VelocityField> {
        if velocity.width() != depth.width() || velocity.height() != depth.height() {
            return None;
        }

        let width = velocity.width();
        let height = velocity.height();
        let count = width * height;
        if count == 0 {
            // Degenerate field: no pixels to dispatch, mirror the golden's
            // empty result exactly.
            return Some(VelocityField::zeroed(width, height));
        }

        let device = ctx.device();

        let vel_in: Vec<GpuVec2> = velocity
            .as_slice()
            .iter()
            .map(|&v| GpuVec2::from_vec2(v))
            .collect();
        let depth_in = depth_to_row_major(depth);

        let params = Params {
            width: u32::try_from(width).expect("field width fits in u32"),
            height: u32::try_from(height).expect("field height fits in u32"),
            radius: u32::try_from(radius).expect("radius fits in u32"),
            order_flag: order_flag(order),
        };

        let out_bytes = (count * size_of::<GpuVec2>()) as u64;
        let params_buf = buffer::uniform(device, "prism_motion_dilate_params", &params);
        let vel_buf = buffer::storage_read(device, "prism_motion_dilate_velocity", &vel_in);
        let depth_buf = buffer::storage_read(device, "prism_motion_dilate_depth", &depth_in);
        let out_buf = buffer::storage_rw_zeroed(device, "prism_motion_dilate_out", out_bytes);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_motion_dilate_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: vel_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: depth_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_motion_dilate_encoder"),
        });
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_motion_dilate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(params.width.div_ceil(TILE), params.height.div_ceil(TILE), 1);
        }

        let stage = buffer::staging(device, "prism_motion_dilate_stage", out_bytes);
        buffer::copy(&mut enc, &out_buf, &stage, out_bytes);
        ctx.queue().submit([enc.finish()]);

        let out_words = buffer::read_back::<GpuVec2>(ctx, &stage);
        let pixels: Vec<Vec2> = out_words
            .iter()
            .take(count)
            .map(|&w| w.to_vec2())
            .collect();
        VelocityField::from_pixels(width, height, pixels)
    }
}

/// One storage/uniform bind-group-layout entry visible to the compute stage.
fn buffer_layout(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
