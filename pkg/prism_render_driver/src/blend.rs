//! Color/alpha blending state for render targets.

/// A per-channel multiplier applied to a blend operand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BlendFactor {
    /// `0`.
    Zero,
    /// `1`.
    One,
    /// The source color.
    Src,
    /// `1 - source color`.
    OneMinusSrc,
    /// The source alpha.
    SrcAlpha,
    /// `1 - source alpha`.
    OneMinusSrcAlpha,
    /// The destination color.
    Dst,
    /// `1 - destination color`.
    OneMinusDst,
    /// The destination alpha.
    DstAlpha,
    /// `1 - destination alpha`.
    OneMinusDstAlpha,
    /// The configured blend constant.
    Constant,
    /// `1 - blend constant`.
    OneMinusConstant,
}

impl BlendFactor {
    /// Whether this factor references the pipeline blend constant, requiring
    /// the encoder to have one set.
    #[must_use]
    pub const fn uses_constant(self) -> bool {
        matches!(self, Self::Constant | Self::OneMinusConstant)
    }
}

/// How weighted source and destination operands are combined.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum BlendOperation {
    /// `src * src_factor + dst * dst_factor`.
    #[default]
    Add,
    /// `src * src_factor - dst * dst_factor`.
    Subtract,
    /// `dst * dst_factor - src * src_factor`.
    ReverseSubtract,
    /// `min(src, dst)` (factors ignored).
    Min,
    /// `max(src, dst)` (factors ignored).
    Max,
}

/// A blend equation for one channel group (color or alpha).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlendComponent {
    /// The factor applied to the source operand.
    pub src_factor: BlendFactor,
    /// The factor applied to the destination operand.
    pub dst_factor: BlendFactor,
    /// How the weighted operands combine.
    pub operation: BlendOperation,
}

impl BlendComponent {
    /// Replace: `src` fully overwrites `dst`.
    pub const REPLACE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    };

    /// Standard non-premultiplied alpha blending.
    pub const ALPHA: Self = Self {
        src_factor: BlendFactor::SrcAlpha,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    };

    /// Additive blending (`src + dst`).
    pub const ADDITIVE: Self = Self {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    };
}

/// The full blend state for a color target: a color equation plus an alpha
/// equation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlendState {
    /// The RGB blend equation.
    pub color: BlendComponent,
    /// The alpha blend equation.
    pub alpha: BlendComponent,
}

impl BlendState {
    /// Opaque: both channels replace the destination.
    pub const REPLACE: Self = Self {
        color: BlendComponent::REPLACE,
        alpha: BlendComponent::REPLACE,
    };

    /// Standard non-premultiplied alpha over.
    pub const ALPHA_BLENDING: Self = Self {
        color: BlendComponent::ALPHA,
        alpha: BlendComponent::REPLACE,
    };

    /// Premultiplied alpha over.
    pub const PREMULTIPLIED_ALPHA: Self = Self {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::REPLACE,
    };

    /// Additive on both channels.
    pub const ADDITIVE: Self = Self {
        color: BlendComponent::ADDITIVE,
        alpha: BlendComponent::ADDITIVE,
    };

    /// Whether any component references the pipeline blend constant.
    #[must_use]
    pub const fn uses_constant(self) -> bool {
        self.color.src_factor.uses_constant()
            || self.color.dst_factor.uses_constant()
            || self.alpha.src_factor.uses_constant()
            || self.alpha.dst_factor.uses_constant()
    }
}
