//! Bidirectional mapping between the OS-assigned `winit::window::WindowId` and
//! the kernel's stable [`prism_window::WindowId`].
//!
//! The kernel never sees platform ids: it allocates its own monotonically
//! increasing ids through the registry, and this map is the only place the two
//! namespaces meet. Keeping both directions in one owner avoids the classic
//! "two maps drift out of sync" bug.

use std::collections::HashMap;

use prism_window::WindowId;

/// A two-way `winit <-> kernel` window-id translation table.
///
/// Both directions are updated together through [`Self::insert`] and
/// [`Self::remove_by_kernel`]/[`Self::remove_by_winit`], so they can never
/// disagree.
#[derive(Debug, Default)]
pub struct WindowIdMap {
    winit_to_kernel: HashMap<winit::window::WindowId, WindowId>,
    kernel_to_winit: HashMap<WindowId, winit::window::WindowId>,
}

impl WindowIdMap {
    /// An empty map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of live associations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.winit_to_kernel.len()
    }

    /// Whether the map holds no associations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.winit_to_kernel.is_empty()
    }

    /// Associates a platform id with a kernel id. If either id was already
    /// mapped, the stale association on both sides is dropped first so the two
    /// directions stay consistent. Returns the kernel id previously bound to
    /// `platform`, if any.
    pub fn insert(
        &mut self,
        platform: winit::window::WindowId,
        kernel: WindowId,
    ) -> Option<WindowId> {
        // Drop any prior binding of this kernel id to a different platform id.
        if let Some(old_platform) = self.kernel_to_winit.insert(kernel, platform)
            && old_platform != platform
        {
            self.winit_to_kernel.remove(&old_platform);
        }
        let previous = self.winit_to_kernel.insert(platform, kernel);
        if let Some(old_kernel) = previous
            && old_kernel != kernel
        {
            self.kernel_to_winit.remove(&old_kernel);
        }
        previous
    }

    /// Resolves a platform id to its kernel id.
    #[must_use]
    pub fn kernel(&self, platform: winit::window::WindowId) -> Option<WindowId> {
        self.winit_to_kernel.get(&platform).copied()
    }

    /// Resolves a kernel id to its platform id.
    #[must_use]
    pub fn winit(&self, kernel: WindowId) -> Option<winit::window::WindowId> {
        self.kernel_to_winit.get(&kernel).copied()
    }

    /// Removes the association for a kernel id (window destroyed). Returns the
    /// platform id that was unbound.
    pub fn remove_by_kernel(&mut self, kernel: WindowId) -> Option<winit::window::WindowId> {
        let platform = self.kernel_to_winit.remove(&kernel)?;
        self.winit_to_kernel.remove(&platform);
        Some(platform)
    }

    /// Removes the association for a platform id. Returns the kernel id that was
    /// unbound.
    pub fn remove_by_winit(&mut self, platform: winit::window::WindowId) -> Option<WindowId> {
        let kernel = self.winit_to_kernel.remove(&platform)?;
        self.kernel_to_winit.remove(&kernel);
        Some(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wid(n: u64) -> winit::window::WindowId {
        winit::window::WindowId::from(n)
    }

    #[test]
    fn insert_and_resolve_both_directions() {
        let mut map = WindowIdMap::new();
        assert!(map.is_empty());
        assert_eq!(map.insert(wid(10), WindowId(1)), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.kernel(wid(10)), Some(WindowId(1)));
        assert_eq!(map.winit(WindowId(1)), Some(wid(10)));
        assert_eq!(map.kernel(wid(99)), None);
    }

    #[test]
    fn reinsert_same_platform_rebinds_without_leaking() {
        let mut map = WindowIdMap::new();
        map.insert(wid(10), WindowId(1));
        // Same platform id now points at a new kernel id (e.g. recreated window).
        assert_eq!(map.insert(wid(10), WindowId(2)), Some(WindowId(1)));
        assert_eq!(map.len(), 1);
        assert_eq!(map.kernel(wid(10)), Some(WindowId(2)));
        // The stale kernel->platform entry must be gone, not dangling.
        assert_eq!(map.winit(WindowId(1)), None);
        assert_eq!(map.winit(WindowId(2)), Some(wid(10)));
    }

    #[test]
    fn remove_by_kernel_clears_both_sides() {
        let mut map = WindowIdMap::new();
        map.insert(wid(10), WindowId(1));
        assert_eq!(map.remove_by_kernel(WindowId(1)), Some(wid(10)));
        assert!(map.is_empty());
        assert_eq!(map.kernel(wid(10)), None);
        assert_eq!(map.winit(WindowId(1)), None);
    }

    #[test]
    fn remove_by_winit_clears_both_sides() {
        let mut map = WindowIdMap::new();
        map.insert(wid(10), WindowId(1));
        assert_eq!(map.remove_by_winit(wid(10)), Some(WindowId(1)));
        assert!(map.is_empty());
    }
}
