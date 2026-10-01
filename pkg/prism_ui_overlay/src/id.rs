//! Overlay identity.

use core::fmt;

/// An opaque, cheap, `Copy` handle to an overlay held by an
/// [`OverlayManager`](crate::OverlayManager).
///
/// Ids are allocated monotonically and are never reused within a single
/// manager, so a stale id simply fails to match after its overlay is
/// dismissed rather than aliasing a different overlay.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct OverlayId(pub(crate) u64);

impl OverlayId {
    /// Returns the raw monotonic value backing this id.
    ///
    /// Useful for logging or building stable external keys; it carries no
    /// meaning beyond uniqueness and allocation order.
    #[must_use]
    pub fn to_raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for OverlayId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
