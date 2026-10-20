//! Multi-viewport render-target ownership and lifetime contracts.

use alloc::collections::BTreeMap;
use core::fmt;

use crate::{
    GpuPickingBackend, GpuPickingConfig, GpuPickingError, GpuPickingFrame, SceneIdPass, ViewportId,
};

/// Stable identity of a viewport render-target allocation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct RenderTargetHandle(u64);

impl RenderTargetHandle {
    /// Creates a handle from persisted or externally allocated data.
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric handle value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Immutable lifecycle state shared with renderer scheduling and readback code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderTargetState {
    /// Viewport owning this target.
    pub viewport: ViewportId,
    /// Stable target allocation identity.
    pub handle: RenderTargetHandle,
    /// Viewport/camera generation used when the target was last configured.
    pub viewport_generation: u64,
    /// Target allocation generation, incremented after each resize.
    pub target_generation: u64,
    /// Current ID-pass extent.
    pub config: GpuPickingConfig,
}

/// A target allocation and its real wgpu ID-buffer backend.
pub struct ViewportRenderTarget {
    state: RenderTargetState,
    backend: GpuPickingBackend,
}

impl fmt::Debug for ViewportRenderTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ViewportRenderTarget")
            .field("state", &self.state)
            .field("backend", &self.backend)
            .finish()
    }
}

impl ViewportRenderTarget {
    fn new(state: RenderTargetState, backend: GpuPickingBackend) -> Self {
        Self { state, backend }
    }

    /// Returns lifecycle metadata for this target.
    #[must_use]
    pub const fn state(&self) -> RenderTargetState {
        self.state
    }

    /// Returns the real ID-buffer backend.
    #[must_use]
    pub const fn backend(&self) -> &GpuPickingBackend {
        &self.backend
    }

    /// Starts an ID-pass frame bound to this target's current lifecycle state.
    ///
    /// Request validation therefore uses the exact viewport, generation, target
    /// generation, and allocation handle owned by this registry entry.
    #[must_use]
    pub fn begin_frame(&self, scene: &SceneIdPass) -> GpuPickingFrame<'_> {
        self.backend.begin_frame(self.state, scene.generation())
    }

    /// Resizes the target and advances its allocation generation.
    pub fn resize(
        &mut self,
        viewport_generation: u64,
        config: GpuPickingConfig,
    ) -> Result<bool, RenderTargetError> {
        if viewport_generation != self.state.viewport_generation {
            return Err(RenderTargetError::ViewportGenerationMismatch {
                expected: self.state.viewport_generation,
                actual: viewport_generation,
            });
        }
        if config == self.state.config {
            return Ok(false);
        }
        let generation = self
            .state
            .target_generation
            .checked_add(1)
            .ok_or(RenderTargetError::GenerationExhausted)?;
        self.backend
            .resize(config)
            .map_err(RenderTargetError::Gpu)?;
        self.state.config = config;
        self.state.target_generation = generation;
        Ok(true)
    }

    /// Advances viewport/camera generation; backward transitions are rejected.
    pub fn set_viewport_generation(&mut self, generation: u64) -> Result<(), RenderTargetError> {
        if generation < self.state.viewport_generation {
            return Err(RenderTargetError::ViewportGenerationMismatch {
                expected: self.state.viewport_generation,
                actual: generation,
            });
        }
        self.state.viewport_generation = generation;
        Ok(())
    }
}

/// Failure from target allocation or lifecycle transitions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RenderTargetError {
    /// The viewport already owns a target.
    AlreadyOpen(ViewportId),
    /// The viewport has no target.
    Unknown(ViewportId),
    /// The supplied viewport generation does not match the target's state.
    ViewportGenerationMismatch {
        /// Generation recorded by the target.
        expected: u64,
        /// Generation supplied by the caller.
        actual: u64,
    },
    /// The target generation cannot advance further.
    GenerationExhausted,
    /// The target backend rejected a resource operation.
    Gpu(GpuPickingError),
    /// The target handle space is exhausted.
    HandleExhausted,
}

impl fmt::Display for RenderTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyOpen(id) => write!(
                formatter,
                "render target already open for viewport {}",
                id.raw()
            ),
            Self::Unknown(id) => {
                write!(formatter, "unknown render target for viewport {}", id.raw())
            }
            Self::ViewportGenerationMismatch { expected, actual } => write!(
                formatter,
                "render target viewport generation mismatch: expected {expected}, got {actual}"
            ),
            Self::GenerationExhausted => formatter.write_str("render target generation exhausted"),
            Self::Gpu(error) => write!(formatter, "render target GPU operation failed: {error}"),
            Self::HandleExhausted => formatter.write_str("render target handle space exhausted"),
        }
    }
}

impl core::error::Error for RenderTargetError {}

/// Deterministic owner of all open viewport render-target allocations.
pub struct RenderTargetRegistry {
    targets: BTreeMap<ViewportId, ViewportRenderTarget>,
    next_handle: u64,
}

impl Default for RenderTargetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for RenderTargetRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RenderTargetRegistry")
            .field("targets", &self.targets.keys())
            .field("next_handle", &self.next_handle)
            .finish()
    }
}

impl RenderTargetRegistry {
    /// Creates an empty render-target registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            targets: BTreeMap::new(),
            next_handle: 1,
        }
    }

    /// Allocates a target for one viewport using the existing GPU picking backend.
    pub fn open(
        &mut self,
        viewport: ViewportId,
        viewport_generation: u64,
        backend: GpuPickingBackend,
    ) -> Result<RenderTargetState, RenderTargetError> {
        if self.targets.contains_key(&viewport) {
            return Err(RenderTargetError::AlreadyOpen(viewport));
        }
        let handle = RenderTargetHandle(self.next_handle);
        self.next_handle = self
            .next_handle
            .checked_add(1)
            .ok_or(RenderTargetError::HandleExhausted)?;
        let state = RenderTargetState {
            viewport,
            handle,
            viewport_generation,
            target_generation: 0,
            config: backend.config(),
        };
        self.targets
            .insert(viewport, ViewportRenderTarget::new(state, backend));
        Ok(state)
    }

    /// Closes a target and returns its final lifecycle state.
    pub fn close(&mut self, viewport: ViewportId) -> Result<RenderTargetState, RenderTargetError> {
        self.targets
            .remove(&viewport)
            .map(|target| target.state())
            .ok_or(RenderTargetError::Unknown(viewport))
    }

    /// Resizes a target while preserving its stable handle.
    pub fn resize(
        &mut self,
        viewport: ViewportId,
        viewport_generation: u64,
        config: GpuPickingConfig,
    ) -> Result<RenderTargetState, RenderTargetError> {
        let target = self
            .targets
            .get_mut(&viewport)
            .ok_or(RenderTargetError::Unknown(viewport))?;
        target.resize(viewport_generation, config)?;
        Ok(target.state())
    }

    /// Updates the associated viewport generation for a camera or settings change.
    pub fn set_viewport_generation(
        &mut self,
        viewport: ViewportId,
        generation: u64,
    ) -> Result<RenderTargetState, RenderTargetError> {
        let target = self
            .targets
            .get_mut(&viewport)
            .ok_or(RenderTargetError::Unknown(viewport))?;
        target.set_viewport_generation(generation)?;
        Ok(target.state())
    }

    /// Returns one open target.
    #[must_use]
    pub fn get(&self, viewport: ViewportId) -> Option<&ViewportRenderTarget> {
        self.targets.get(&viewport)
    }

    /// Returns current lifecycle metadata for one open target.
    #[must_use]
    pub fn state(&self, viewport: ViewportId) -> Option<RenderTargetState> {
        self.targets.get(&viewport).map(ViewportRenderTarget::state)
    }

    /// Returns all open targets in stable viewport order.
    pub fn iter(&self) -> impl Iterator<Item = &ViewportRenderTarget> {
        self.targets.values()
    }

    /// Returns the number of open targets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Returns whether no target is open.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use super::*;

    #[test]
    fn open_resize_close_preserves_handle_and_invalidates_target_generation() {
        let Some((adapter, device, queue)) = crate::test_support::device() else {
            return;
        };
        let backend =
            GpuPickingBackend::new(&device, &adapter, &queue, GpuPickingConfig::new(4, 4))
                .expect("Rgba32Uint render target");
        let viewport = ViewportId::from_raw(7);
        let mut registry = RenderTargetRegistry::new();
        let opened = registry.open(viewport, 3, backend).expect("open");
        assert_eq!(opened.handle.raw(), 1);
        assert_eq!(opened.target_generation, 0);
        let other = ViewportId::from_raw(8);
        let backend =
            GpuPickingBackend::new(&device, &adapter, &queue, GpuPickingConfig::new(2, 2))
                .expect("second target");
        let other_state = registry.open(other, 0, backend).expect("second viewport");
        assert_ne!(opened.handle, other_state.handle);
        assert_eq!(registry.len(), 2);
        assert_eq!(
            registry.resize(viewport, 3, GpuPickingConfig::new(0, 4)),
            Err(RenderTargetError::Gpu(GpuPickingError::InvalidExtent {
                width: 0,
                height: 4
            }))
        );
        assert_eq!(registry.state(viewport), Some(opened));
        let resized = registry
            .resize(viewport, 3, GpuPickingConfig::new(8, 4))
            .expect("resize");
        assert_eq!(resized.handle, opened.handle);
        assert_eq!(resized.target_generation, 1);
        assert_eq!(
            registry
                .resize(viewport, 3, resized.config)
                .expect("no-op resize"),
            resized
        );
        assert_eq!(registry.state(other), Some(other_state));
        assert_eq!(
            registry
                .set_viewport_generation(viewport, 4)
                .expect("generation")
                .viewport_generation,
            4
        );
        assert_eq!(
            registry.close(viewport).expect("close").viewport_generation,
            4
        );
        assert_eq!(registry.state(viewport), None);
        assert_eq!(registry.state(other), Some(other_state));
        let backend = GpuPickingBackend::new(&device, &adapter, &queue, opened.config)
            .expect("replacement target");
        let reopened = registry.open(viewport, 3, backend).expect("reopen target");
        assert_ne!(reopened.handle, opened.handle);
        assert_eq!(reopened.target_generation, opened.target_generation);
        assert!(matches!(
            registry.set_viewport_generation(viewport, 2),
            Err(RenderTargetError::ViewportGenerationMismatch { .. })
        ));
        registry.close(viewport).expect("close replacement");
        registry.close(other).expect("close other");
        assert!(registry.is_empty());
        crate::test_support::diagnostic(format_args!(
            "EXERCISED viewport GPU multi-target resize/close/reopen lifecycle"
        ));
    }
}
