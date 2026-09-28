//! Compilation targets and shader stages understood by the toolchain.

use core::fmt;

/// A backend code-generation target for `slangc`.
///
/// The variants map directly onto `slangc -target <name>`. WGSL is Prism's
/// primary product because the runtime goes through `wgpu`; the remaining
/// targets exist so a single Slang source can also feed native backends and
/// the CPU golden-reference path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    /// WGSL for `wgpu` (primary runtime product).
    Wgsl,
    /// SPIR-V for Vulkan / ray tracing (optional high-tier native path).
    Spirv,
    /// Metal Shading Language (Apple native).
    Metal,
    /// C++ host source — the CPU golden-reference target.
    ///
    /// Emitting the same closure as CPU code eliminates the hand-aligned
    /// second implementation that reflection alone cannot guarantee.
    CppHost,
}

impl Target {
    /// The `-target` token passed to `slangc`.
    pub fn slangc_name(self) -> &'static str {
        match self {
            Target::Wgsl => "wgsl",
            Target::Spirv => "spirv",
            Target::Metal => "metal",
            Target::CppHost => "cpp",
        }
    }

    /// The conventional file extension for this target's artifact.
    pub fn extension(self) -> &'static str {
        match self {
            Target::Wgsl => "wgsl",
            Target::Spirv => "spv",
            Target::Metal => "metal",
            Target::CppHost => "cpp",
        }
    }

    /// Whether the artifact is textual (vs. a binary blob).
    pub fn is_text(self) -> bool {
        !matches!(self, Target::Spirv)
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slangc_name())
    }
}

/// A shader stage / entry-point kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    /// Vertex stage.
    Vertex,
    /// Fragment / pixel stage.
    Fragment,
    /// Compute stage.
    Compute,
    /// Ray generation stage.
    RayGeneration,
    /// Ray closest-hit stage.
    ClosestHit,
    /// Ray any-hit stage.
    AnyHit,
    /// Ray miss stage.
    Miss,
}

impl ShaderStage {
    /// The `-stage` token passed to `slangc`.
    pub fn slangc_name(self) -> &'static str {
        match self {
            ShaderStage::Vertex => "vertex",
            ShaderStage::Fragment => "fragment",
            ShaderStage::Compute => "compute",
            ShaderStage::RayGeneration => "raygeneration",
            ShaderStage::ClosestHit => "closesthit",
            ShaderStage::AnyHit => "anyhit",
            ShaderStage::Miss => "miss",
        }
    }
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slangc_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_names_are_slangc_tokens() {
        assert_eq!(Target::Wgsl.slangc_name(), "wgsl");
        assert_eq!(Target::Spirv.slangc_name(), "spirv");
        assert_eq!(Target::Metal.slangc_name(), "metal");
        assert_eq!(Target::CppHost.slangc_name(), "cpp");
    }

    #[test]
    fn only_spirv_is_binary() {
        assert!(Target::Wgsl.is_text());
        assert!(Target::Metal.is_text());
        assert!(Target::CppHost.is_text());
        assert!(!Target::Spirv.is_text());
    }

    #[test]
    fn stage_names_are_slangc_tokens() {
        assert_eq!(ShaderStage::Compute.slangc_name(), "compute");
        assert_eq!(ShaderStage::ClosestHit.slangc_name(), "closesthit");
        assert_eq!(ShaderStage::RayGeneration.slangc_name(), "raygeneration");
    }
}
