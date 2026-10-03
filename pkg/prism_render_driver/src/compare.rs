//! Comparison functions shared by depth tests, stencil tests, and comparison
//! samplers.

/// The predicate applied when comparing an incoming value against a stored
/// value (for example a depth fragment against the depth buffer).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum CompareFunction {
    /// The test never passes.
    Never,
    /// Passes when `new < stored`.
    Less,
    /// Passes when `new == stored`.
    Equal,
    /// Passes when `new <= stored`.
    LessEqual,
    /// Passes when `new > stored`.
    Greater,
    /// Passes when `new != stored`.
    NotEqual,
    /// Passes when `new >= stored`.
    GreaterEqual,
    /// The test always passes.
    #[default]
    Always,
}

impl CompareFunction {
    /// Whether this function depends on the stored value at all. `Never` and
    /// `Always` are constant and let backends skip loading the comparand.
    #[must_use]
    pub const fn needs_reference(self) -> bool {
        !matches!(self, Self::Never | Self::Always)
    }
}
