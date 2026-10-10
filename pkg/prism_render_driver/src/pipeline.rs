//! Pipeline layout and render/compute pipeline descriptors.

use crate::blend::BlendState;
use crate::flags::{ColorWrites, ShaderStages};
use crate::format::TextureFormat;
use crate::resource::{BindGroupLayoutId, PipelineLayoutId, ShaderModuleId};
use crate::state::{DepthStencilState, MultisampleState, PrimitiveState};
use crate::vertex::VertexBufferLayout;
use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

/// A byte range of push constants visible to a set of stages.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PushConstantRange {
    /// The stages that can read this range.
    pub stages: ShaderStages,
    /// The start byte offset.
    pub start: u32,
    /// The end byte offset (exclusive).
    pub end: u32,
}

/// A pipeline layout: the ordered bind group layouts and push constant ranges a
/// pipeline draws its resources from.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct PipelineLayoutDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The bind group layouts, indexed by `@group(n)`.
    pub bind_group_layouts: Vec<BindGroupLayoutId>,
    /// Push constant ranges.
    pub push_constant_ranges: Vec<PushConstantRange>,
}

/// The vertex stage configuration of a render pipeline.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VertexState {
    /// The shader module providing the vertex entry point.
    pub module: ShaderModuleId,
    /// The `@vertex` entry point name.
    pub entry_point: Cow<'static, str>,
    /// The vertex buffer layouts, indexed by vertex buffer slot.
    pub buffers: Vec<VertexBufferLayout>,
}

/// The blend/write configuration of one color render target.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ColorTargetState {
    /// The target texture format.
    pub format: TextureFormat,
    /// The blend state, or `None` to overwrite without blending.
    pub blend: Option<BlendState>,
    /// Which channels are written.
    pub write_mask: ColorWrites,
}

impl ColorTargetState {
    /// An opaque, fully-written target with no blending.
    #[must_use]
    pub const fn opaque(format: TextureFormat) -> Self {
        Self {
            format,
            blend: None,
            write_mask: ColorWrites::all(),
        }
    }

    /// An alpha-blended, fully-written target.
    #[must_use]
    pub const fn alpha_blended(format: TextureFormat) -> Self {
        Self {
            format,
            blend: Some(BlendState::ALPHA_BLENDING),
            write_mask: ColorWrites::all(),
        }
    }
}

/// The fragment stage configuration of a render pipeline.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FragmentState {
    /// The shader module providing the fragment entry point.
    pub module: ShaderModuleId,
    /// The `@fragment` entry point name.
    pub entry_point: Cow<'static, str>,
    /// The color targets, indexed by `@location(n)`; `None` disables a slot.
    pub targets: Vec<Option<ColorTargetState>>,
}

/// A request to create a render pipeline.
#[derive(Clone, PartialEq, Debug)]
pub struct RenderPipelineDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The pipeline layout, or `None` for an auto-derived layout.
    pub layout: Option<PipelineLayoutId>,
    /// The vertex stage.
    pub vertex: VertexState,
    /// Primitive assembly and rasterization state.
    pub primitive: PrimitiveState,
    /// The depth/stencil state, or `None` when there is no such attachment.
    pub depth_stencil: Option<DepthStencilState>,
    /// Multisample state.
    pub multisample: MultisampleState,
    /// The fragment stage, or `None` for a depth-only pipeline.
    pub fragment: Option<FragmentState>,
}

impl RenderPipelineDescriptor {
    /// The number of color targets this pipeline writes (0 when depth-only).
    #[must_use]
    pub fn color_target_count(&self) -> usize {
        self.fragment.as_ref().map_or(0, |frag| {
            frag.targets.iter().filter(|t| t.is_some()).count()
        })
    }

    /// Whether the pipeline's declared state is internally consistent: the
    /// primitive state must be valid and the multisample sample count non-zero.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.primitive.is_valid() && self.multisample.count >= 1
    }
}

/// A request to create a compute pipeline.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ComputePipelineDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The pipeline layout, or `None` for an auto-derived layout.
    pub layout: Option<PipelineLayoutId>,
    /// The shader module providing the compute entry point.
    pub module: ShaderModuleId,
    /// The `@compute` entry point name.
    pub entry_point: Cow<'static, str>,
}
