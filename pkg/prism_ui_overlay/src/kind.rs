//! Overlay kinds and their z-priority ordering.

use core::cmp::Ordering;

/// The semantic kind of an overlay.
///
/// Each kind carries a fixed [z-priority](OverlayKind::z_priority) that
/// determines how overlays stack: within a single
/// [`OverlayManager`](crate::OverlayManager) all overlays are drawn in
/// ascending priority order, and ties are broken by insertion order.
///
/// The ordering is, from bottom to top: [`Modal`](OverlayKind::Modal),
/// [`Popover`](OverlayKind::Popover), [`Tooltip`](OverlayKind::Tooltip),
/// [`Toast`](OverlayKind::Toast). Modals sit at the base of the overlay stack
/// (they capture interaction behind a backdrop), transient hints and
/// notifications float above so they are never obscured.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OverlayKind {
    /// A transient hint anchored to a widget, e.g. a hover tooltip.
    Tooltip,
    /// A contextual surface anchored to a trigger, e.g. a menu or combo box.
    Popover,
    /// A blocking surface that captures interaction behind a backdrop.
    Modal,
    /// A transient notification that floats above all other overlays.
    Toast,
}

impl OverlayKind {
    /// The stacking priority of this kind.
    ///
    /// Higher values are drawn later and therefore appear on top. The concrete
    /// numbers are spaced so callers may reason about relative order without
    /// depending on exact values.
    #[must_use]
    pub fn z_priority(self) -> u16 {
        match self {
            OverlayKind::Modal => 100,
            OverlayKind::Popover => 200,
            OverlayKind::Tooltip => 300,
            OverlayKind::Toast => 400,
        }
    }

    /// Whether this kind contributes a scrim/backdrop element when rendered.
    ///
    /// Only [`Modal`](OverlayKind::Modal) overlays draw a backdrop.
    #[must_use]
    pub fn has_backdrop(self) -> bool {
        matches!(self, OverlayKind::Modal)
    }

    /// Whether this kind is a modal.
    #[must_use]
    pub fn is_modal(self) -> bool {
        matches!(self, OverlayKind::Modal)
    }
}

impl PartialOrd for OverlayKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OverlayKind {
    fn cmp(&self, other: &Self) -> Ordering {
        self.z_priority().cmp(&other.z_priority())
    }
}

#[cfg(test)]
mod tests {
    use super::OverlayKind;

    #[test]
    fn priority_orders_bottom_to_top() {
        assert!(OverlayKind::Modal < OverlayKind::Popover);
        assert!(OverlayKind::Popover < OverlayKind::Tooltip);
        assert!(OverlayKind::Tooltip < OverlayKind::Toast);
    }

    #[test]
    fn only_modal_has_backdrop() {
        assert!(OverlayKind::Modal.has_backdrop());
        assert!(OverlayKind::Modal.is_modal());
        for k in [
            OverlayKind::Tooltip,
            OverlayKind::Popover,
            OverlayKind::Toast,
        ] {
            assert!(!k.has_backdrop());
            assert!(!k.is_modal());
        }
    }
}
