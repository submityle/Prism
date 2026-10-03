//! Shader module description.

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

/// The source form of a shader module.
///
/// Backends consume whichever representation they support; the RHI stays
/// language-agnostic by carrying the source opaquely. SPIR-V words are stored
/// owned so the descriptor is `'static` and cheap to move between threads.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ShaderSource {
    /// WGSL source text.
    Wgsl(Cow<'static, str>),
    /// Prism's WESL source text (WGSL superset used by the renderer).
    Wesl(Cow<'static, str>),
    /// Pre-compiled SPIR-V words.
    SpirV(Vec<u32>),
}

impl ShaderSource {
    /// Borrows the text source, or `None` for binary SPIR-V.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Wgsl(src) | Self::Wesl(src) => Some(src),
            Self::SpirV(_) => None,
        }
    }
}

/// A request to create a shader module.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ShaderModuleDescriptor {
    /// A debug label surfaced in GPU tooling.
    pub label: Option<String>,
    /// The shader source.
    pub source: ShaderSource,
}

impl ShaderModuleDescriptor {
    /// Builds a descriptor from static WGSL text.
    #[must_use]
    pub fn wgsl(source: &'static str) -> Self {
        Self {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(source)),
        }
    }

    /// Builds a descriptor from static WESL text.
    #[must_use]
    pub fn wesl(source: &'static str) -> Self {
        Self {
            label: None,
            source: ShaderSource::Wesl(Cow::Borrowed(source)),
        }
    }
}
