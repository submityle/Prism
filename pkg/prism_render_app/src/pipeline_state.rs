//! Descriptor-derived fixed-function identity, with a versioned stable encoding.
//!
//! These named values, not their hash, are the semantic source of truth. Enum
//! tags below are explicit wire tags independent of RHI discriminant order.
//! Unknown future RHI formats fail loudly until this schema is extended.

use prism_render_architecture::shader_package::PipelineStateHash;
use prism_render_driver as rhi;

macro_rules! state_enum {
    ($name:ident, $source:ident, {$($variant:ident = $tag:literal),+ $(,)?}) => {
        #[doc = concat!("Stable identity tags for RHI `", stringify!($source), "`.")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(u64)]
        pub enum $name { $($variant = $tag),+ }

        impl From<rhi::$source> for $name {
            fn from(value: rhi::$source) -> Self {
                // The wildcard is required only for non_exhaustive RHI enums.
                #[allow(unreachable_patterns)]
                match value {
                    $(rhi::$source::$variant => Self::$variant,)+
                    _ => panic!("unsupported pipeline-state {}: {value:?}", stringify!($source)),
                }
            }
        }
    };
}

state_enum!(AttachmentFormat, TextureFormat, {
    R8Unorm = 0, R8Uint = 1, R16Float = 2, R32Float = 3, R32Uint = 4,
    Rg8Unorm = 5, Rg16Float = 6, Rg32Float = 7, Rgba8Unorm = 8,
    Rgba8UnormSrgb = 9, Bgra8Unorm = 10, Bgra8UnormSrgb = 11,
    Rgb10a2Unorm = 12, Rg11b10Float = 13, Rgba16Float = 14,
    Rgba32Float = 15, Depth32Float = 16, Depth24PlusStencil8 = 17,
});
state_enum!(AttributeFormat, VertexFormat, {
    Float32 = 0, Float32x2 = 1, Float32x3 = 2, Float32x4 = 3,
    Float16x2 = 4, Float16x4 = 5, Uint32 = 6, Uint32x2 = 7,
    Uint32x4 = 8, Sint32 = 9, Unorm8x4 = 10, Snorm8x4 = 11,
});
state_enum!(Topology, PrimitiveTopology, {
    PointList = 0, LineList = 1, LineStrip = 2, TriangleList = 3, TriangleStrip = 4,
});
state_enum!(StripIndexFormat, IndexFormat, { Uint16 = 0, Uint32 = 1 });
state_enum!(Winding, FrontFace, { Ccw = 0, Cw = 1 });
state_enum!(CullFace, Face, { Front = 0, Back = 1 });
state_enum!(FillMode, PolygonMode, { Fill = 0, Line = 1, Point = 2 });
state_enum!(Comparison, CompareFunction, {
    Never = 0, Less = 1, Equal = 2, LessEqual = 3, Greater = 4,
    NotEqual = 5, GreaterEqual = 6, Always = 7,
});
state_enum!(StencilOp, StencilOperation, {
    Keep = 0, Zero = 1, Replace = 2, Invert = 3, IncrementClamp = 4,
    DecrementClamp = 5, IncrementWrap = 6, DecrementWrap = 7,
});
state_enum!(Factor, BlendFactor, {
    Zero = 0, One = 1, Src = 2, OneMinusSrc = 3, SrcAlpha = 4,
    OneMinusSrcAlpha = 5, Dst = 6, OneMinusDst = 7, DstAlpha = 8,
    OneMinusDstAlpha = 9, Constant = 10, OneMinusConstant = 11,
});
state_enum!(BlendOp, BlendOperation, {
    Add = 0, Subtract = 1, ReverseSubtract = 2, Min = 3, Max = 4,
});
state_enum!(StepMode, VertexStepMode, { Vertex = 0, Instance = 1 });

/// Exact IEEE-754 bits, giving depth bias reflexive equality and total ordering.
/// Signed zero and different NaN payloads remain distinct; no descriptor bits
/// are silently normalized or lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FloatBits(pub u32);

impl From<f32> for FloatBits {
    fn from(value: f32) -> Self {
        Self(value.to_bits())
    }
}

/// All fixed-function fields exposed by the RHI render pipeline descriptor.
/// Full ordered layouts/targets (including disabled slots) participate in key
/// equality. Labels, shader handles/entry points and pipeline-layout handles
/// are excluded: the separate program identity owns those contracts.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineState {
    pub vertex_buffers: Vec<VertexBufferState>,
    pub rasterizer: RasterizerState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    /// `None` means no fragment stage; `Some([])` is a fragment stage without
    /// color outputs and is a different pipeline.
    pub color_targets: Option<Vec<Option<ColorTargetState>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexBufferState {
    pub array_stride: u64,
    pub step_mode: StepMode,
    pub attributes: Vec<VertexAttributeState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexAttributeState {
    pub format: AttributeFormat,
    pub offset: u64,
    pub shader_location: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RasterizerState {
    pub topology: Topology,
    pub strip_index_format: Option<StripIndexFormat>,
    pub front_face: Winding,
    pub cull_mode: Option<CullFace>,
    pub polygon_mode: FillMode,
    pub unclipped_depth: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DepthStencilState {
    pub format: AttachmentFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: Comparison,
    pub stencil: StencilState,
    pub bias: DepthBiasState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StencilState {
    pub front: StencilFaceState,
    pub back: StencilFaceState,
    pub read_mask: u32,
    pub write_mask: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StencilFaceState {
    pub compare: Comparison,
    pub fail_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub pass_op: StencilOp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DepthBiasState {
    pub constant: i32,
    pub slope_scale: FloatBits,
    pub clamp: FloatBits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColorTargetState {
    pub format: AttachmentFormat,
    pub blend: Option<BlendState>,
    pub write_mask: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlendState {
    pub color: BlendComponent,
    pub alpha: BlendComponent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlendComponent {
    pub src_factor: Factor,
    pub dst_factor: Factor,
    pub operation: BlendOp,
}

impl PipelineState {
    /// Exhaustive destructuring deliberately makes a newly added descriptor or
    /// nested RHI field a compilation error until its identity is accounted for.
    #[must_use]
    pub fn from_descriptor(descriptor: &rhi::RenderPipelineDescriptor) -> Self {
        let rhi::RenderPipelineDescriptor {
            label: _,
            layout: _,
            vertex,
            primitive,
            depth_stencil,
            multisample,
            fragment,
        } = descriptor;
        let rhi::VertexState {
            module: _,
            entry_point: _,
            buffers,
        } = vertex;
        let rhi::PrimitiveState {
            topology,
            strip_index_format,
            front_face,
            cull_mode,
            polygon_mode,
            unclipped_depth,
        } = *primitive;
        let rhi::MultisampleState {
            count,
            mask,
            alpha_to_coverage_enabled,
        } = *multisample;
        Self {
            vertex_buffers: buffers.iter().map(VertexBufferState::from).collect(),
            rasterizer: RasterizerState {
                topology: topology.into(),
                strip_index_format: strip_index_format.map(Into::into),
                front_face: front_face.into(),
                cull_mode: cull_mode.map(Into::into),
                polygon_mode: polygon_mode.into(),
                unclipped_depth,
            },
            depth_stencil: depth_stencil.as_ref().map(DepthStencilState::from),
            multisample: MultisampleState {
                count,
                mask,
                alpha_to_coverage_enabled,
            },
            color_targets: fragment.as_ref().map(|fragment| {
                let rhi::FragmentState {
                    module: _,
                    entry_point: _,
                    targets,
                } = fragment;
                targets
                    .iter()
                    .map(|target| target.as_ref().map(ColorTargetState::from))
                    .collect()
            }),
        }
    }

    /// Stable FNV-1a encoding for DriverLocal storage, never semantic identity.
    /// Schema v1: `b"PSTATE"`, then every scalar/tag/length as LE u64; booleans
    /// and option tags are 0/1, i32 is sign-extended to u64, float bits are u32
    /// zero-extended to u64. Sequence order and disabled target slots are kept.
    /// Changing the encoding requires bumping the version and golden fixture.
    #[must_use]
    pub fn to_hash(&self) -> PipelineStateHash {
        let mut out = StateEncoder::new(false);
        self.encode(&mut out);
        PipelineStateHash(out.hash)
    }

    /// Canonical bytes for verifying a persisted PSO against its complete
    /// semantic state. Persistence must compare these bytes before importing a
    /// driver blob: a compact hash alone is not collision-safe identity.
    /// Uses exactly the same versioned encoding as [`Self::to_hash`].
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = StateEncoder::new(true);
        self.encode(&mut out);
        out.canonical.expect("canonical encoding enabled")
    }

    fn encode(&self, out: &mut StateEncoder) {
        let Self {
            vertex_buffers,
            rasterizer,
            depth_stencil,
            multisample,
            color_targets,
        } = self;
        out.bytes(b"PSTATE");
        out.word(1); // schema version
        out.word(vertex_buffers.len() as u64);
        for buffer in vertex_buffers {
            buffer.encode(out);
        }
        rasterizer.encode(out);
        out.option(depth_stencil, |out, state| state.encode(out));
        multisample.encode(out);
        out.option(color_targets, |out, targets| {
            out.word(targets.len() as u64);
            for target in targets {
                out.option(target, |out, state| state.encode(out));
            }
        });
    }
}

impl From<&rhi::VertexBufferLayout> for VertexBufferState {
    fn from(value: &rhi::VertexBufferLayout) -> Self {
        let rhi::VertexBufferLayout {
            array_stride,
            step_mode,
            attributes,
        } = value;
        Self {
            array_stride: *array_stride,
            step_mode: (*step_mode).into(),
            attributes: attributes
                .iter()
                .map(|attribute| {
                    let rhi::VertexAttribute {
                        format,
                        offset,
                        shader_location,
                    } = *attribute;
                    VertexAttributeState {
                        format: format.into(),
                        offset,
                        shader_location,
                    }
                })
                .collect(),
        }
    }
}

impl From<&rhi::DepthStencilState> for DepthStencilState {
    fn from(value: &rhi::DepthStencilState) -> Self {
        let rhi::DepthStencilState {
            format,
            depth_write_enabled,
            depth_compare,
            stencil,
            bias,
        } = *value;
        let rhi::StencilState {
            front,
            back,
            read_mask,
            write_mask,
        } = stencil;
        let rhi::DepthBiasState {
            constant,
            slope_scale,
            clamp,
        } = bias;
        Self {
            format: format.into(),
            depth_write_enabled,
            depth_compare: depth_compare.into(),
            stencil: StencilState {
                front: front.into(),
                back: back.into(),
                read_mask,
                write_mask,
            },
            bias: DepthBiasState {
                constant,
                slope_scale: slope_scale.into(),
                clamp: clamp.into(),
            },
        }
    }
}

impl From<rhi::StencilFaceState> for StencilFaceState {
    fn from(value: rhi::StencilFaceState) -> Self {
        let rhi::StencilFaceState {
            compare,
            fail_op,
            depth_fail_op,
            pass_op,
        } = value;
        Self {
            compare: compare.into(),
            fail_op: fail_op.into(),
            depth_fail_op: depth_fail_op.into(),
            pass_op: pass_op.into(),
        }
    }
}

impl From<&rhi::ColorTargetState> for ColorTargetState {
    fn from(value: &rhi::ColorTargetState) -> Self {
        let rhi::ColorTargetState {
            format,
            blend,
            write_mask,
        } = *value;
        Self {
            format: format.into(),
            write_mask: write_mask.bits(),
            blend: blend.map(|blend| {
                let rhi::BlendState { color, alpha } = blend;
                BlendState {
                    color: color.into(),
                    alpha: alpha.into(),
                }
            }),
        }
    }
}

impl From<rhi::BlendComponent> for BlendComponent {
    fn from(value: rhi::BlendComponent) -> Self {
        let rhi::BlendComponent {
            src_factor,
            dst_factor,
            operation,
        } = value;
        Self {
            src_factor: src_factor.into(),
            dst_factor: dst_factor.into(),
            operation: operation.into(),
        }
    }
}

struct StateEncoder {
    hash: u64,
    canonical: Option<Vec<u8>>,
}
impl StateEncoder {
    fn new(collect: bool) -> Self {
        Self {
            hash: 0xcbf2_9ce4_8422_2325,
            canonical: collect.then(Vec::new),
        }
    }
    fn bytes(&mut self, bytes: &[u8]) {
        if let Some(canonical) = &mut self.canonical {
            canonical.extend_from_slice(bytes);
        }
        for &byte in bytes {
            self.hash ^= u64::from(byte);
            self.hash = self.hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn word(&mut self, word: u64) {
        self.bytes(&word.to_le_bytes());
    }
    fn option<T>(&mut self, value: &Option<T>, encode: impl FnOnce(&mut Self, &T)) {
        self.word(u64::from(value.is_some()));
        if let Some(value) = value {
            encode(self, value);
        }
    }
}

// Each encoder destructures without `..`: adding a named state field without
// encoding it fails compilation, in addition to the field mutation test gate.
impl VertexBufferState {
    fn encode(&self, out: &mut StateEncoder) {
        let Self {
            array_stride,
            step_mode,
            attributes,
        } = self;
        out.word(*array_stride);
        out.word(*step_mode as u64);
        out.word(attributes.len() as u64);
        for attribute in attributes {
            let VertexAttributeState {
                format,
                offset,
                shader_location,
            } = attribute;
            out.word(*format as u64);
            out.word(*offset);
            out.word(u64::from(*shader_location));
        }
    }
}
impl RasterizerState {
    fn encode(&self, out: &mut StateEncoder) {
        let Self {
            topology,
            strip_index_format,
            front_face,
            cull_mode,
            polygon_mode,
            unclipped_depth,
        } = self;
        out.word(*topology as u64);
        out.option(strip_index_format, |out, value| out.word(*value as u64));
        out.word(*front_face as u64);
        out.option(cull_mode, |out, value| out.word(*value as u64));
        out.word(*polygon_mode as u64);
        out.word(u64::from(*unclipped_depth));
    }
}
impl DepthStencilState {
    fn encode(&self, out: &mut StateEncoder) {
        let Self {
            format,
            depth_write_enabled,
            depth_compare,
            stencil,
            bias,
        } = self;
        let StencilState {
            front,
            back,
            read_mask,
            write_mask,
        } = stencil;
        let DepthBiasState {
            constant,
            slope_scale,
            clamp,
        } = bias;
        out.word(*format as u64);
        out.word(u64::from(*depth_write_enabled));
        out.word(*depth_compare as u64);
        front.encode(out);
        back.encode(out);
        out.word(u64::from(*read_mask));
        out.word(u64::from(*write_mask));
        out.word(i64::from(*constant) as u64);
        out.word(u64::from(slope_scale.0));
        out.word(u64::from(clamp.0));
    }
}
impl StencilFaceState {
    fn encode(&self, out: &mut StateEncoder) {
        let Self {
            compare,
            fail_op,
            depth_fail_op,
            pass_op,
        } = self;
        out.word(*compare as u64);
        out.word(*fail_op as u64);
        out.word(*depth_fail_op as u64);
        out.word(*pass_op as u64);
    }
}
impl MultisampleState {
    fn encode(&self, out: &mut StateEncoder) {
        let Self {
            count,
            mask,
            alpha_to_coverage_enabled,
        } = self;
        out.word(u64::from(*count));
        out.word(*mask);
        out.word(u64::from(*alpha_to_coverage_enabled));
    }
}
impl ColorTargetState {
    fn encode(&self, out: &mut StateEncoder) {
        let Self {
            format,
            blend,
            write_mask,
        } = self;
        out.word(*format as u64);
        out.option(blend, |out, blend| {
            let BlendState { color, alpha } = blend;
            color.encode(out);
            alpha.encode(out);
        });
        out.word(u64::from(*write_mask));
    }
}
impl BlendComponent {
    fn encode(&self, out: &mut StateEncoder) {
        let Self {
            src_factor,
            dst_factor,
            operation,
        } = self;
        out.word(*src_factor as u64);
        out.word(*dst_factor as u64);
        out.word(*operation as u64);
    }
}

#[cfg(test)]
#[path = "pipeline_state_tests.rs"]
mod tests;
