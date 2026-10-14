//! Deterministic tab-header windowing for large tab stacks.

use core::ops::Range;

/// Computes the tab indexes that should be materialized for a horizontal tab
/// strip. The active tab is retained even when it lies outside the viewport.
pub fn visible_tab_range(
    total: usize,
    scroll_px: u32,
    viewport_px: u32,
    tab_width_px: u32,
    overscan: usize,
    active: usize,
) -> Range<usize> {
    if total == 0 || tab_width_px == 0 || viewport_px == 0 {
        return 0..0;
    }
    let first = (scroll_px / tab_width_px) as usize;
    let visible = ((viewport_px.saturating_add(tab_width_px - 1)) / tab_width_px) as usize;
    let mut start = first.saturating_sub(overscan).min(total);
    let mut end = first
        .saturating_add(visible)
        .saturating_add(overscan)
        .min(total);
    if active < start {
        start = active;
    }
    if active >= end {
        end = active.saturating_add(1).min(total);
    }
    start..end
}

#[cfg(test)]
mod tests {
    use super::visible_tab_range;

    #[test]
    fn active_tab_is_retained_outside_the_viewport() {
        assert_eq!(visible_tab_range(100, 0, 200, 100, 1, 50), 0..51);
    }

    #[test]
    fn overscan_and_bounds_are_stable() {
        assert_eq!(visible_tab_range(10, 500, 200, 100, 2, 7), 3..9);
        assert_eq!(visible_tab_range(0, 0, 100, 20, 2, 0), 0..0);
    }
}
