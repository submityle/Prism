//! # `prism_render_driver`
//!
//! Prism's render hardware interface (RHI): a backend-agnostic description of
//! GPU resources, pipeline state, and command submission that decouples the
//! renderer from any single graphics API. Concrete backends (wgpu, Vulkan, …)
//! live in separate crates and implement the traits here; this crate is a
//! pure, deterministic, `no_std + alloc` type system with no `unsafe`.
//!
//! ## Layout
//! - Resource descriptors: [`buffer`], [`texture`], [`sampler`], [`shader`].
//! - Pipeline state: [`format`], [`vertex`], [`blend`], [`state`],
//!   [`binding`], [`pipeline`].
//! - Command model: [`command`].
//! - Device contract: [`capabilities`], [`resource`], [`device`].
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

mod binding;
mod blend;
mod buffer;
mod capabilities;
mod color;
mod command;
mod compare;
mod device;
mod flags;
mod format;
mod pipeline;
mod resource;
mod sampler;
mod shader;
mod state;
mod texture;
mod vertex;

pub use binding::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, SamplerBindingType, StorageTextureAccess,
};
pub use blend::{BlendComponent, BlendFactor, BlendOperation, BlendState};
pub use buffer::BufferDescriptor;
pub use capabilities::{Backend, DeviceCapabilities, Features, Limits};
pub use color::Color;
pub use command::{
    ColorAttachment, CommandBuffer, CommandEncoder, ComputeCommand, DepthLoadOp, DepthOperations,
    DepthStencilAttachment, IndexBufferBinding, LoadOp, Pass, RenderCommand, RenderPassDescriptor,
    ScissorRect, StencilLoadOp, StencilOperations, StoreOp, Viewport,
};
pub use compare::CompareFunction;
pub use device::{ImageDataLayout, RenderDevice, RenderQueue, TextureWrite};
pub use flags::{BufferUsages, ColorWrites, ShaderStages, TextureUsages};
pub use format::{FormatAspects, TextureFormat, TextureSampleType, VertexFormat};
pub use pipeline::{
    ColorTargetState, ComputePipelineDescriptor, FragmentState, PipelineLayoutDescriptor,
    PushConstantRange, RenderPipelineDescriptor, VertexState,
};
pub use resource::{
    BindGroupId, BindGroupLayoutId, BufferId, ComputePipelineId, PipelineLayoutId, RenderPipelineId,
    ResourceId, SamplerId, ShaderModuleId, TextureId, TextureViewId,
};
pub use sampler::{AddressMode, FilterMode, SamplerBorderColor, SamplerDescriptor};
pub use shader::{ShaderModuleDescriptor, ShaderSource};
pub use state::{
    DepthBiasState, DepthStencilState, Face, FrontFace, IndexFormat, MultisampleState,
    PolygonMode, PrimitiveState, PrimitiveTopology, StencilFaceState, StencilOperation,
    StencilState,
};
pub use texture::{
    Extent3d, TextureAspect, TextureDescriptor, TextureDimension, TextureViewDescriptor,
    TextureViewDimension,
};
pub use vertex::{VertexAttribute, VertexBufferLayout, VertexStepMode};

/// The common imports for building render pipelines and recording commands.
///
/// `use prism_render_driver::prelude::*;` brings the most frequently used
/// descriptors, state, and ids into scope without pulling in every type.
pub mod prelude {
    pub use crate::{
        AddressMode, BindGroupDescriptor, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
        BindingResource, BlendState, BufferDescriptor, BufferId, BufferUsages, Color,
        ColorTargetState, ColorWrites, CommandBuffer, CommandEncoder, CompareFunction,
        ComputePipelineDescriptor, DepthStencilState, DeviceCapabilities, Extent3d, Features,
        FilterMode, FragmentState, IndexFormat, Limits, LoadOp, MultisampleState,
        PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderCommand, RenderDevice,
        RenderPassDescriptor, RenderPipelineDescriptor, RenderQueue, SamplerDescriptor, ShaderStages,
        StoreOp, TextureDescriptor, TextureFormat, TextureId, TextureUsages, TextureViewDescriptor,
        VertexBufferLayout, VertexFormat, VertexState,
    };
}

#[cfg(test)]
mod tests;
