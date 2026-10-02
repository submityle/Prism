//! Two-level invalidation state for incremental layout.
//!
//! Each box carries two independent dirty bits that mirror Flutter's
//! `markNeedsLayout` / `markNeedsPaint` split:
//!
//! * [`DirtyFlags::needs_layout`] — the box's geometry must be recomputed.
//! * [`DirtyFlags::needs_paint`] — only the box's visual appearance changed,
//!   so it must be repainted but its geometry is still valid.
//!
//! Separating the two prevents a pure appearance change (for example a color
//! swap) from forcing an expensive re-layout. See the crate design notes
//! (§9.3) for the full cost model.

use crate::geometry::Dimension;
use crate::style::LayoutStyle;

/// The invalidation state of a single box.
///
/// A freshly created box starts fully dirty (`needs_layout == true`,
/// `needs_paint == true`) because it has never been laid out or painted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirtyFlags {
    needs_layout: bool,
    needs_paint: bool,
}

impl DirtyFlags {
    /// Flags for a box that has never been laid out: both bits set.
    pub const DIRTY: Self = Self {
        needs_layout: true,
        needs_paint: true,
    };

    /// Flags for a box that is fully up to date: both bits clear.
    pub const CLEAN: Self = Self {
        needs_layout: false,
        needs_paint: false,
    };

    /// Returns `true` when the box's geometry must be recomputed.
    pub fn needs_layout(self) -> bool {
        self.needs_layout
    }

    /// Returns `true` when the box must be repainted.
    pub fn needs_paint(self) -> bool {
        self.needs_paint
    }

    /// Marks the geometry dirty.
    ///
    /// A geometry change always implies a repaint, so this also sets the
    /// paint bit.
    pub fn mark_needs_layout(&mut self) {
        self.needs_layout = true;
        self.needs_paint = true;
    }

    /// Marks only the appearance dirty, leaving the geometry bit untouched.
    pub fn mark_needs_paint(&mut self) {
        self.needs_paint = true;
    }

    /// Clears the geometry bit after a successful layout pass.
    ///
    /// The paint bit is deliberately preserved: recomputing geometry does not
    /// discharge a pending repaint request on its own.
    pub fn clear_needs_layout(&mut self) {
        self.needs_layout = false;
    }

    /// Clears the paint bit after a successful paint pass.
    pub fn clear_needs_paint(&mut self) {
        self.needs_paint = false;
    }
}

impl Default for DirtyFlags {
    fn default() -> Self {
        Self::DIRTY
    }
}

/// Returns `true` when `style` describes a *relayout boundary*.
///
/// A relayout boundary is a box whose border-box size is uniquely determined
/// by its own constraints and therefore cannot be affected by changes inside
/// its subtree. A box with both a definite `width` and a definite `height`
/// ([`Dimension::Points`]) qualifies: no matter how its children re-measure,
/// its own size stays fixed, so a child's `needs_layout` never needs to
/// propagate past it.
///
/// Dirty propagation stops *at* such a boundary (the boundary itself is still
/// marked, because its subtree must be re-solved), but never continues to the
/// boundary's parent.
pub fn is_relayout_boundary(style: &LayoutStyle) -> bool {
    matches!(style.size.width, Dimension::Points(_))
        && matches!(style.size.height, Dimension::Points(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Size;

    #[test]
    fn default_is_fully_dirty() {
        let flags = DirtyFlags::default();
        assert!(flags.needs_layout());
        assert!(flags.needs_paint());
    }

    #[test]
    fn mark_needs_layout_also_marks_paint() {
        let mut flags = DirtyFlags::CLEAN;
        flags.mark_needs_layout();
        assert!(flags.needs_layout());
        assert!(flags.needs_paint());
    }

    #[test]
    fn mark_needs_paint_leaves_layout_clean() {
        let mut flags = DirtyFlags::CLEAN;
        flags.mark_needs_paint();
        assert!(!flags.needs_layout());
        assert!(flags.needs_paint());
    }

    #[test]
    fn clear_bits_are_independent() {
        let mut flags = DirtyFlags::DIRTY;
        flags.clear_needs_layout();
        assert!(!flags.needs_layout());
        assert!(flags.needs_paint());
        flags.clear_needs_paint();
        assert!(!flags.needs_paint());
    }

    #[test]
    fn fixed_box_is_a_boundary() {
        let style = LayoutStyle {
            size: Size::new(Dimension::Points(100.0), Dimension::Points(50.0)),
            ..LayoutStyle::default()
        };
        assert!(is_relayout_boundary(&style));
    }

    #[test]
    fn auto_box_is_not_a_boundary() {
        let style = LayoutStyle {
            size: Size::new(Dimension::Points(100.0), Dimension::Auto),
            ..LayoutStyle::default()
        };
        assert!(!is_relayout_boundary(&style));
        assert!(!is_relayout_boundary(&LayoutStyle::default()));
    }
}
