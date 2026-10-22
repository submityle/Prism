//! Bounded indirect graphics consumer for the planetary placement ABI.
//!
//! This module consumes the producer's 116-byte (29 `u32`) records, compact
//! source-slot indices and four-word indirect arguments. It is intentionally
//! unrelated to the planar 64-byte placement ABI and is not a product scene
//! renderer.
//!
//! Graphics hazards declared by this consumer (the existing planetary graph
//! plan only lowers producer compute hazards): records and compact indices are
//! vertex-stage storage reads; indirect arguments are an indirect-command
//! read; the caller's color target is a single-sample color-attachment write;
//! the frame/view uniform is a vertex-stage uniform read. The caller must order
//! producer compute before this draw in the same queue/graph and provide the
//! required storage-write -> vertex-storage-read and storage-write ->
//! indirect-read barriers when using a graph that does not infer them.

use std::{
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{GpuContext, GpuPlanetPlacementRegistrationSnapshot};

const SHADER: &str = include_str!("shaders/planet_placement_draw.wgsl");
const UNIFORM_BYTES: u64 = 80;

/// A caller-owned single-sample 2D color attachment.
pub struct PlanetPlacementColorTarget<'a> {
    /// Render attachment view. Its underlying texture must be 2D and single-sample.
    pub view: &'a wgpu::TextureView,
    /// Must match the target view's color format.
    pub format: wgpu::TextureFormat,
    /// Target width used to convert the instance size from pixels to clip space.
    pub width: u32,
    /// Target height used to convert the instance size from pixels to clip space.
    pub height: u32,
    /// Must be exactly one; multisampled targets are rejected.
    pub sample_count: u32,
}

/// Caller-selected transform and billboard size for one frame-local placement draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetPlacementDrawParams {
    /// Column-major transform from frame-local meters to clip space.
    pub frame_view: [[f32; 4]; 4],
    /// Square billboard width and height in target pixels; must be finite and positive.
    pub instance_size_pixels: f32,
}

/// Draw validation or bounded submission failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanetPlacementDrawError {
    /// This consumer's bounded queue already has this many unfinished draws.
    InFlightLimit,
    /// A zero-sized target cannot define pixel-to-clip conversion.
    EmptyTarget,
    /// Planet placements require a single-sample target.
    MultisampledTarget,
    /// The target format differs from the format used to create this pipeline.
    TargetFormatMismatch,
    /// Billboard size must be finite and positive.
    InvalidInstanceSize,
    /// Every frame-view matrix element must be finite.
    InvalidTransform,
}

impl std::fmt::Display for PlanetPlacementDrawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InFlightLimit => f.write_str("planet placement draw in-flight limit reached"),
            Self::EmptyTarget => f.write_str("planet placement draw target is empty"),
            Self::MultisampledTarget => {
                f.write_str("planet placement draw requires a single-sample target")
            }
            Self::TargetFormatMismatch => {
                f.write_str("planet placement draw target format does not match its pipeline")
            }
            Self::InvalidInstanceSize => {
                f.write_str("planet placement instance size must be finite and positive")
            }
            Self::InvalidTransform => {
                f.write_str("planet placement frame-view transform must be finite")
            }
        }
    }
}

impl std::error::Error for PlanetPlacementDrawError {}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    frame_view: [[f32; 4]; 4],
    target_size: [f32; 2],
    instance_size_pixels: f32,
    _padding: f32,
}

struct PendingDraw {
    done: Arc<AtomicBool>,
}

/// Completion receipt for one indirect planetary draw.
pub struct PlanetPlacementDrawSubmission {
    done: Arc<AtomicBool>,
}

impl PlanetPlacementDrawSubmission {
    /// True only after the native queue's completion callback has run.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

/// Fixed-pipeline indirect graphics consumer for planetary placement records.
///
/// Each accepted submission owns one bounded uniform allocation and bind
/// group; the in-flight cap bounds those allocations. A completion callback
/// captures the registration snapshot and per-draw bindings so all producer
/// buffers remain alive until the graphics command itself completes.
pub struct PlanetPlacementDraw {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    max_in_flight: NonZeroUsize,
    pending: Vec<PendingDraw>,
}

impl PlanetPlacementDraw {
    /// Creates a renderer for one output format and a fixed maximum number of
    /// outstanding draw submissions.
    #[must_use]
    pub fn new(
        context: &GpuContext,
        format: wgpu::TextureFormat,
        max_in_flight: NonZeroUsize,
    ) -> Self {
        let device = context.device();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("planet placement draw bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(UNIFORM_BYTES),
                    },
                    count: None,
                },
                storage_layout(1),
                storage_layout(2),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("planet placement record indirect draw"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("planet placement draw pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("planet placement billboard pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            device: device.clone(),
            queue: context.queue().clone(),
            format,
            layout,
            pipeline,
            max_in_flight,
            pending: Vec::with_capacity(max_in_flight.get()),
        }
    }

    /// Returns the number of draws whose queue completion has not yet been observed.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Reaps completed submissions without blocking.
    pub fn poll(&mut self) -> usize {
        let before = self.pending.len();
        self.pending
            .retain(|pending| !pending.done.load(Ordering::Acquire));
        before - self.pending.len()
    }

    /// Encodes and submits a real indirect render draw into a caller-owned target.
    ///
    /// The snapshot must belong to the same device as this consumer, and its
    /// producer's compute submission must precede this call on the queue (or be
    /// ordered by the caller's render graph). Signed local millimetre values
    /// remain integer limbs until the vertex shader converts them to meters.
    pub fn submit(
        &mut self,
        snapshot: GpuPlanetPlacementRegistrationSnapshot,
        target: PlanetPlacementColorTarget<'_>,
        params: PlanetPlacementDrawParams,
    ) -> Result<PlanetPlacementDrawSubmission, PlanetPlacementDrawError> {
        self.poll();
        if self.pending.len() == self.max_in_flight.get() {
            return Err(PlanetPlacementDrawError::InFlightLimit);
        }
        if target.width == 0 || target.height == 0 {
            return Err(PlanetPlacementDrawError::EmptyTarget);
        }
        if target.sample_count != 1 {
            return Err(PlanetPlacementDrawError::MultisampledTarget);
        }
        if target.format != self.format {
            return Err(PlanetPlacementDrawError::TargetFormatMismatch);
        }
        if !params.instance_size_pixels.is_finite() || params.instance_size_pixels <= 0.0 {
            return Err(PlanetPlacementDrawError::InvalidInstanceSize);
        }
        if !params
            .frame_view
            .iter()
            .flatten()
            .all(|element| element.is_finite())
        {
            return Err(PlanetPlacementDrawError::InvalidTransform);
        }

        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("planet placement per-draw frame view"),
                contents: bytemuck::bytes_of(&DrawUniform {
                    frame_view: params.frame_view,
                    target_size: [target.width as f32, target.height as f32],
                    instance_size_pixels: params.instance_size_pixels,
                    _padding: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("planet placement draw snapshot bindings"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: snapshot.records.buffer(),
                        offset: 0,
                        size: wgpu::BufferSize::new(snapshot.records.bytes()),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: snapshot.compact.buffer(),
                        offset: 0,
                        size: wgpu::BufferSize::new(snapshot.compact.bytes()),
                    }),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("planet placement indirect draw encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("planet placement indirect billboard pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw_indirect(snapshot.indirect.buffer(), 0);
        }
        self.queue.submit([encoder.finish()]);
        let done = Arc::new(AtomicBool::new(false));
        let callback_done = done.clone();
        // Captures every object referenced by this command, including the
        // producer snapshot; producer-compute completion alone is insufficient.
        self.queue.on_submitted_work_done(move || {
            drop((snapshot, bind_group, uniform));
            callback_done.store(true, Ordering::Release);
        });
        self.pending.push(PendingDraw { done: done.clone() });
        Ok(PlanetPlacementDrawSubmission { done })
    }
}

fn storage_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
