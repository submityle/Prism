//! A format-correct final raster pass; swapchain images need only attachment
//! usage. Sampling the RGBA scene target handles BGRA and sRGB surfaces through
//! the GPU's typed load/store conversion rather than an invalid raw image copy.

use prism_render_driver::*;
use prism_render_graph::{ImportedTexture, PassFlags, RenderGraph, TextureDesc, TextureHandle};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

pub(super) struct PresentPipeline<D: RenderDevice> {
    device: Arc<D>,
    shader: Option<ShaderModuleId>,
    bindings: Option<BindGroupLayoutId>,
    layout: Option<PipelineLayoutId>,
    pipeline: Option<RenderPipelineId>,
    group: Option<BindGroupId>,
    in_flight: Arc<AtomicUsize>,
    destroy_allowed: Arc<AtomicBool>,
}

impl<D: RenderDevice> PresentPipeline<D> {
    pub(super) fn new(
        device: Arc<D>,
        source_view: TextureViewId,
        format: TextureFormat,
        in_flight: Arc<AtomicUsize>,
        destroy_allowed: Arc<AtomicBool>,
    ) -> Self {
        // Own each allocation immediately, including during backend validation
        // panics. An incomplete replacement never damages the live pipeline.
        let mut owned = Self {
            device,
            shader: None,
            bindings: None,
            layout: None,
            pipeline: None,
            group: None,
            in_flight,
            destroy_allowed,
        };
        let device = &*owned.device;
        let shader_source = include_str!("present.wgsl").replace(
            "const TARGET_SRGB: bool = false;",
            if format.is_srgb() {
                "const TARGET_SRGB: bool = true;"
            } else {
                "const TARGET_SRGB: bool = false;"
            },
        );
        let shader = device.create_shader_module(&ShaderModuleDescriptor {
            label: Some("window present conversion".into()),
            source: ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_source)),
        });
        owned.shader = Some(shader);
        let bindings = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("window present texture".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        owned.bindings = Some(bindings);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("window present layout".into()),
            bind_group_layouts: vec![bindings],
            push_constant_ranges: vec![],
        });
        owned.layout = Some(layout);
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("window present conversion".into()),
            layout: Some(layout),
            vertex: VertexState {
                module: shader,
                entry_point: "vs_main".into(),
                buffers: vec![],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: shader,
                entry_point: "fs_main".into(),
                targets: vec![Some(ColorTargetState::opaque(format))],
            }),
        });
        owned.pipeline = Some(pipeline);
        let group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("window present source".into()),
            layout: Some(bindings),
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(source_view),
            }],
        });
        owned.group = Some(group);
        owned
    }

    pub(super) fn append(
        &self,
        graph: &mut RenderGraph,
        source: TextureHandle,
        image: SurfaceImage,
    ) {
        let destination = graph.import_texture(
            "window-swapchain",
            TextureDesc::color(image.format),
            ImportedTexture {
                texture: image.texture,
                default_view: image.view,
                entry_state: TextureState::initial(),
                exit_state: TextureState::present(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let pipeline = self.pipeline.expect("initialized present pipeline");
        let group = self.group.expect("initialized present bindings");
        graph.add_raster_pass("window-present", PassFlags::SIDE_EFFECT, move |builder| {
            builder.sample(source);
            builder.color_attachment(destination);
            move |ctx| {
                ctx.set_render_pipeline(pipeline)
                    .set_render_bind_group(0, group, vec![])
                    .draw(0..3, 0..1);
            }
        });
    }
}

impl<D: RenderDevice> Drop for PresentPipeline<D> {
    fn drop(&mut self) {
        // Unconfirmed submissions keep their ids in the owning backend tables
        // until device teardown. Never explicitly free resources still in use.
        if !self.destroy_allowed.load(Ordering::Acquire)
            || self.in_flight.load(Ordering::Acquire) != 0
        {
            return;
        }
        let device = &*self.device;
        if let Some(id) = self.group {
            device.destroy_bind_group(id);
        }
        if let Some(id) = self.pipeline {
            device.destroy_render_pipeline(id);
        }
        if let Some(id) = self.layout {
            device.destroy_pipeline_layout(id);
        }
        if let Some(id) = self.bindings {
            device.destroy_bind_group_layout(id);
        }
        if let Some(id) = self.shader {
            device.destroy_shader_module(id);
        }
    }
}
