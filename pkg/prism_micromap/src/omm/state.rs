//! Per-micro-triangle opacity states and the two `DXR` micromap formats.

/// Opacity classification of a single micro-triangle.
///
/// The discriminants match the `DXR` / `VK_EXT_opacity_micromap` special-value
/// encoding so a packed `4-state` micromap is byte-compatible with hardware:
/// `0` transparent, `1` opaque, `2` unknown-transparent, `3` unknown-opaque.
/// In a `2-state` micromap only [`OpacityState::Transparent`] (`0`) and
/// [`OpacityState::Opaque`] (`1`) are representable; the two unknown states are
/// resolved conservatively by [`OpacityState::to_2state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OpacityState {
    /// The micro-triangle is fully transparent; a ray hit can be skipped
    /// without entering the any-hit shader.
    Transparent = 0,
    /// The micro-triangle is fully opaque; a ray hit is accepted without
    /// entering the any-hit shader.
    Opaque = 1,
    /// Coverage is mixed; traversal must fall back to the any-hit shader. The
    /// `2-state` resolution of this value is [`OpacityState::Transparent`].
    UnknownTransparent = 2,
    /// Coverage is mixed; traversal must fall back to the any-hit shader. The
    /// `2-state` resolution of this value is [`OpacityState::Opaque`].
    UnknownOpaque = 3,
}

impl OpacityState {
    /// Returns the raw `2-bit` `DXR` discriminant for this state.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns `true` when the state is one of the two mixed-coverage
    /// (`Unknown`) classifications.
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::UnknownTransparent | Self::UnknownOpaque)
    }

    /// Collapses the state to its conservative `2-state` value.
    ///
    /// Mixed-coverage micro-triangles become [`OpacityState::Opaque`] so the
    /// hardware still invokes the any-hit shader (never silently dropping a
    /// potentially visible fragment), except for
    /// [`OpacityState::UnknownTransparent`], which the baker only emits when
    /// the dominant coverage is transparent.
    #[must_use]
    pub const fn to_2state(self) -> Self {
        match self {
            Self::Transparent | Self::UnknownTransparent => Self::Transparent,
            Self::Opaque | Self::UnknownOpaque => Self::Opaque,
        }
    }

    /// Reconstructs a state from its raw `2-bit` discriminant.
    ///
    /// Returns [`None`] when `bits` is outside the `0..=3` range.
    #[must_use]
    pub const fn from_u8(bits: u8) -> Option<Self> {
        match bits {
            0 => Some(Self::Transparent),
            1 => Some(Self::Opaque),
            2 => Some(Self::UnknownTransparent),
            3 => Some(Self::UnknownOpaque),
            _ => None,
        }
    }
}

/// Encoding width of a baked micromap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OmmFormat {
    /// `1-bit` per micro-triangle: only `Transparent` / `Opaque`. Mixed
    /// coverage is resolved via [`OpacityState::to_2state`]. This is the
    /// `DXR` `OC1_2_STATE` format.
    TwoState,
    /// `2-bit` per micro-triangle: the full four-state encoding including the
    /// two `Unknown` fallbacks. This is the `DXR` `OC1_4_STATE` format.
    FourState,
}

impl OmmFormat {
    /// Number of bits each micro-triangle occupies in this format.
    #[must_use]
    pub const fn bits_per_micro_triangle(self) -> u32 {
        match self {
            Self::TwoState => 1,
            Self::FourState => 2,
        }
    }

    /// Normalises `state` to the representable set for this format.
    #[must_use]
    pub const fn normalize(self, state: OpacityState) -> OpacityState {
        match self {
            Self::TwoState => state.to_2state(),
            Self::FourState => state,
        }
    }
}
