//! Single-owner GPU backend contracts.

/// Selects one independently owned backend. Resources never cross modes
/// without an explicit interop implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendMode {
    VulkanFirst,
    WgpuCompatibility,
}

/// Vulkan capability tiers consumed by higher-level render features.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum VulkanTier {
    #[default]
    Core13,
    MeshShader,
    RayQuery,
    Full,
}

/// The sole owner of resource creation, command recording, and submission.
pub trait RenderBackend {
    fn mode(&self) -> BackendMode;
    fn tier(&self) -> VulkanTier;
    fn wait_idle_for_shutdown(&mut self);
}
