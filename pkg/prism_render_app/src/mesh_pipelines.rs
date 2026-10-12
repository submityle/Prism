//! The mesh pipelines the queue specializes through.
//!
//! Specializing a mesh pipeline needs two things the queue cannot own: the shader
//! and target state (the renderer's, injected through
//! [`MeshPipelineProvider`]) and the device (the module must exist before a
//! descriptor can name it). This module is the render-world resource that holds
//! the provider, plus the spine's default provider.
//!
//! # The spine's default is a stand-in, on purpose
//!
//! [`SpineMeshPipeline`] is a flat-shaded pass-through: it draws geometry where
//! its vertices say, in one colour, with no lighting. It exists so the render
//! spine is runnable and testable on its own — a frame that queues a mesh draws
//! it — and it is **replaced** by installing a real provider
//! ([`MeshPipelines::new`]) once the material system's shading is ported onto
//! this spine. It is not a rendering model, and nothing keys off it.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use prism_ecs::prelude::Resource;
use prism_render_driver::{
    ColorTargetState, DepthStencilState, FragmentState, MultisampleState, PrimitiveState,
    RenderDevice, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderModuleId, TextureFormat,
    VertexState,
};

use crate::mesh_pipeline::{MeshPipelineKey, MeshPipelineProvider};

/// The pipeline provider every queued mesh draw specializes through.
#[derive(Resource)]
pub struct MeshPipelines {
    provider: Box<dyn MeshPipelineProvider + Send + Sync>,
}

impl MeshPipelines {
    /// Wraps `provider` as the render world's mesh pipeline source.
    pub fn new(provider: impl MeshPipelineProvider + Send + Sync + 'static) -> Self {
        Self {
            provider: Box::new(provider),
        }
    }

    /// The spine's default: a flat pass-through pipeline.
    ///
    /// Built against `device` because a pipeline descriptor names a shader
    /// module, and the module has to exist first.
    #[must_use]
    pub fn spine(device: &dyn RenderDevice, color_format: TextureFormat) -> Self {
        Self::new(SpineMeshPipeline::new(device, color_format))
    }

    /// The wrapped provider.
    #[must_use]
    pub fn provider(&self) -> &(dyn MeshPipelineProvider + Send + Sync) {
        &*self.provider
    }
}

impl MeshPipelineProvider for MeshPipelines {
    fn descriptor(&self, variant: &MeshPipelineKey) -> RenderPipelineDescriptor {
        self.provider.descriptor(variant)
    }
}

/// The spine's stand-in pipeline: position in, one colour out.
///
/// See the module docs — this is what makes the spine runnable before the
/// material system's shading is ported, and installing a different
/// [`MeshPipelineProvider`] replaces it.
pub struct SpineMeshPipeline {
    module: ShaderModuleId,
    color_format: TextureFormat,
}

impl SpineMeshPipeline {
    /// The vertex stage: the mesh's first attribute is its clip-space position.
    ///
    /// The fragment stage is one flat colour. Both entry points live in one
    /// module, so a variant costs no extra shader creation.
    const WGSL: &'static str = "\
@vertex fn vs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {\n\
    return vec4<f32>(position, 1.0);\n\
}\n\
@fragment fn fs() -> @location(0) vec4<f32> {\n\
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);\n\
}\n";

    /// Creates the pipeline source against `device`.
    #[must_use]
    pub fn new(device: &dyn RenderDevice, color_format: TextureFormat) -> Self {
        Self {
            module: device.create_shader_module(&ShaderModuleDescriptor::wgsl(Self::WGSL)),
            color_format,
        }
    }
}

impl MeshPipelineProvider for SpineMeshPipeline {
    fn descriptor(&self, variant: &MeshPipelineKey) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("prism-spine-mesh".into()),
            layout: None,
            vertex: VertexState {
                module: self.module,
                entry_point: "vs".into(),
                // Replaced by the specializer with the mesh's own layout, which
                // is the layout the pipeline key hashes.
                buffers: Vec::new(),
            },
            primitive: PrimitiveState::default(),
            // The pass a variant names decides whether there is depth; a variant
            // with no depth format must not declare depth state, or the pipeline
            // would not match its pass.
            depth_stencil: variant
                .depth_format
                .map(DepthStencilState::reverse_z),
            multisample: MultisampleState {
                count: variant.sample_count,
                ..Default::default()
            },
            fragment: Some(FragmentState {
                module: self.module,
                entry_point: "fs".into(),
                targets: vec![Some(ColorTargetState::opaque(self.color_format))],
            }),
        }
    }
}

