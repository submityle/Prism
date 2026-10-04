//! Collapsing a class's responsive overrides into distinct viewport segments.
//!
//! [`MediaResolver`] answers "what is live at *this* width?". This module
//! answers the complementary planning question: "across *all* widths, how many
//! visually distinct states does this class actually have, and where do they
//! begin?".
//!
//! A class only changes its resolved property map at the `min-width` thresholds
//! of the breakpoints it defines, so its appearance is a step function of
//! viewport width. [`responsive_segments`] walks the breakpoint thresholds in
//! ascending order and emits one [`ResponsiveSegment`] per *distinct* resolved
//! [`PropMap`], coalescing adjacent breakpoints whose overrides leave the
//! merged map unchanged. The result lets a layout cache the handful of states a
//! class can take and re-resolve only when a resize crosses a real boundary,
//! instead of recomputing on every pixel of a drag.
//!
//! Resolution here mirrors [`MediaResolver::active_props`]: base plus matching
//! breakpoints, *without* interaction states (which are not width-driven).
//!
//! # Example
//!
//! ```
//! use prism_ui_scoped::responsive_segments;
//! use prism_ui_style::{Breakpoint, Class, StyleProp, StyleValue};
//!
//! // 14px by default, 24px from `md` (768px) up — two distinct states.
//! let class = Class::new("title")
//!     .with(StyleProp::FontSize, StyleValue::px(14.0))
//!     .with_breakpoint(Breakpoint::Md, StyleProp::FontSize, StyleValue::px(24.0));
//!
//! let segments = responsive_segments(&class);
//! assert_eq!(segments.len(), 2);
//! assert_eq!(segments[0].min_width, 0.0);
//! assert_eq!(segments[1].min_width, 768.0);
//! ```

use alloc::vec::Vec;

use prism_ui_style::{Breakpoint, Class, PropMap};

use crate::media::MediaResolver;

/// One contiguous viewport-width segment over which a class resolves to a
/// single, constant property map.
///
/// The segment is live for widths in `[min_width, next.min_width)` (or
/// `[min_width, ∞)` for the last segment). The first segment of any class
/// always begins at `0.0`.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponsiveSegment {
    /// The inclusive minimum viewport width, in logical pixels, at which this
    /// segment becomes live.
    pub min_width: f32,
    /// The resolved property map in effect throughout the segment.
    pub props: PropMap,
}

/// Collapses `class` into its distinct responsive segments, ascending by width.
///
/// Adjacent breakpoints that resolve to an identical [`PropMap`] are merged, so
/// the returned list has one entry per *visually distinct* state. The first
/// entry always starts at `0.0`; every `min_width` is one of the breakpoint
/// thresholds in [`Breakpoint::ALL`] and the list is strictly ascending by
/// width. Interaction states are not considered — only base and breakpoint
/// overrides, matching [`MediaResolver::active_props`].
#[must_use]
pub fn responsive_segments(class: &Class) -> Vec<ResponsiveSegment> {
    let mut segments: Vec<ResponsiveSegment> = Vec::new();
    for bp in Breakpoint::ALL {
        let min_width = bp.min_width();
        let props = MediaResolver::new(min_width).active_props(class);
        let changed = segments.last().is_none_or(|last| last.props != props);
        if changed {
            segments.push(ResponsiveSegment { min_width, props });
        }
    }
    segments
}

/// Returns the segment live at `width`, i.e. the last segment whose `min_width`
/// does not exceed `width`.
///
/// `segments` is assumed to be the ascending output of [`responsive_segments`].
/// Returns `None` only when `width` is below the first segment's `min_width`
/// (which never happens for the `0.0`-anchored output of
/// [`responsive_segments`] at any non-negative width).
#[must_use]
pub fn segment_at(segments: &[ResponsiveSegment], width: f32) -> Option<&ResponsiveSegment> {
    segments.iter().rev().find(|segment| segment.min_width <= width)
}

#[cfg(test)]
mod tests {
    use super::{responsive_segments, segment_at};
    use alloc::vec::Vec;
    use prism_ui_style::{Breakpoint, Class, PropMap, StyleProp, StyleValue};

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    const PROPS: [StyleProp; 3] = [StyleProp::FontSize, StyleProp::Width, StyleProp::Height];

    fn random_class(rng: &mut SplitMix64) -> Class {
        let mut class = Class::new("r");
        // Zero or more base props.
        for &prop in &PROPS {
            if rng.below(2) == 0 {
                class = class.with(prop, StyleValue::px(rng.below(4) as f32));
            }
        }
        // Random overrides at random breakpoints.
        for bp in Breakpoint::ALL {
            for &prop in &PROPS {
                if rng.below(3) == 0 {
                    class = class.with_breakpoint(bp, prop, StyleValue::px(rng.below(4) as f32));
                }
            }
        }
        class
    }

    #[test]
    fn example_class_has_two_segments() {
        let class = Class::new("title")
            .with(StyleProp::FontSize, StyleValue::px(14.0))
            .with_breakpoint(Breakpoint::Md, StyleProp::FontSize, StyleValue::px(24.0));
        let segments = responsive_segments(&class);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].min_width, 0.0);
        assert_eq!(segments[1].min_width, 768.0);
    }

    #[test]
    fn empty_class_has_one_empty_segment() {
        let segments = responsive_segments(&Class::new("empty"));
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].min_width, 0.0);
        assert!(segments[0].props.is_empty());
    }

    #[test]
    fn redundant_breakpoint_override_is_coalesced() {
        // The `md` override re-sets the same value, so it adds no new segment.
        let class = Class::new("c")
            .with(StyleProp::Width, StyleValue::px(10.0))
            .with_breakpoint(Breakpoint::Md, StyleProp::Width, StyleValue::px(10.0));
        let segments = responsive_segments(&class);
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn segment_at_matches_resolver_over_random_inputs() {
        use crate::media::MediaResolver;

        let mut rng = SplitMix64(0x5EC0_D1AB_0000_0001);
        for _ in 0..2_000 {
            let class = random_class(&mut rng);
            let segments = responsive_segments(&class);

            // Invariants: anchored at 0.0, strictly ascending, every boundary a
            // real breakpoint threshold.
            assert_eq!(segments[0].min_width, 0.0);
            for pair in segments.windows(2) {
                assert!(pair[0].min_width < pair[1].min_width);
            }
            for segment in &segments {
                assert!(Breakpoint::ALL.iter().any(|bp| bp.min_width() == segment.min_width));
            }

            // Independent oracle: dense-grid sample of the step function.
            // Walking width by width is a different method than the
            // implementation's threshold enumeration, and must agree.
            let mut expected: Vec<(f32, PropMap)> = Vec::new();
            for w_int in 0..=1_400u32 {
                let width = w_int as f32;
                let props = MediaResolver::new(width).active_props(&class);
                let changed = expected.last().is_none_or(|(_, p)| *p != props);
                if changed {
                    expected.push((width, props));
                }
            }
            assert_eq!(expected.len(), segments.len());
            for (seg, (w, props)) in segments.iter().zip(&expected) {
                assert_eq!(seg.min_width, *w);
                assert_eq!(&seg.props, props);
            }

            // The lookup must equal a fresh resolve at many probe widths.
            for _ in 0..8 {
                let width = rng.below(2_000) as f32;
                let looked_up = segment_at(&segments, width).map(|s| &s.props);
                let resolved = MediaResolver::new(width).active_props(&class);
                assert_eq!(looked_up, Some(&resolved));
            }
        }
    }

    #[test]
    fn segment_at_is_none_below_first_width() {
        let class = Class::new("x").with(StyleProp::Width, StyleValue::px(1.0));
        let segments = responsive_segments(&class);
        assert!(segment_at(&segments, -1.0).is_none());
        assert!(segment_at(&segments, 0.0).is_some());
    }
}
