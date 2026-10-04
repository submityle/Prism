//! CSS Grid track sizing.
//!
//! This module implements the one-dimensional track sizing algorithm from
//! [CSS Grid Layout Level 1 §11](https://www.w3.org/TR/css-grid-1/#algo-track-sizing),
//! resolving a list of track definitions (`grid-template-columns` /
//! `grid-template-rows`) into concrete pixel sizes.
//!
//! Unlike [`crate::grid::GridProtocol`] — which auto-sizes a *fixed column
//! count* to the widest/tallest child — this solver understands the full set
//! of CSS track sizing functions:
//!
//! * fixed lengths (`100px`),
//! * percentages (`25%`) resolved against a definite container length,
//! * `auto`, `min-content`, and `max-content`,
//! * flexible `fr` tracks (`1fr`, `2fr`, `0.5fr`),
//! * `minmax(min, max)`,
//! * `fit-content(limit)`,
//! * and integer [`repeat()`](repeat) expansion.
//!
//! # Content contributions
//!
//! Intrinsic sizing functions (`auto` / `min-content` / `max-content` /
//! `fit-content`) need to know how large the items in a track want to be. The
//! caller supplies this per track as a [`TrackContent`] pair of
//! `min_content` / `max_content` extents (computed by measuring the track's
//! items). Tracks whose sizing functions are purely extrinsic (fixed length,
//! percentage, or `fr`) ignore their [`TrackContent`].
//!
//! # Scope
//!
//! This is the *non-spanning* track sizing model: each item is assumed to
//! occupy a single track, so its contribution lands wholly in that track.
//! Distributing a spanning item's contribution across the tracks it covers
//! ([CSS Grid §11.5.1](https://www.w3.org/TR/css-grid-1/#algo-spanning-items))
//! is a separate concern handled by grid item placement and is intentionally
//! out of scope here. Likewise `repeat(auto-fill, …)` / `repeat(auto-fit, …)`
//! require item placement and are not provided; use the integer [`repeat()`]
//! helper for a known repetition count.
//!
//! Finally, *content distribution* ([CSS Grid §11.8 "stretch auto
//! tracks"](https://www.w3.org/TR/css-grid-1/#algo-stretch)) is also out of
//! scope: when the resolved tracks under-fill a definite container (for
//! example all-fixed tracks, or capped `auto` tracks with no `fr`), the
//! leftover space is the container's free space. Whether that space stretches
//! `auto` tracks or is consumed by alignment is governed by
//! `justify-content` / `align-content`, which are not modelled here; the
//! caller distributes any leftover during grid alignment.
//!
//! ```
//! use prism_ui_layout::geometry::AvailableSpace;
//! use prism_ui_layout::track_sizing::{resolve_track_sizes, TrackContent, TrackSizingFunction};
//!
//! // grid-template-columns: 100px 1fr 2fr; width: 500px;
//! let tracks = [
//!     TrackSizingFunction::points(100.0),
//!     TrackSizingFunction::fr(1.0),
//!     TrackSizingFunction::fr(2.0),
//! ];
//! let content = [TrackContent::ZERO; 3];
//! let sizes = resolve_track_sizes(
//!     &tracks,
//!     &content,
//!     AvailableSpace::Definite(500.0),
//!     0.0,
//! );
//! // 100px fixed, then 400px of free space split 1:2.
//! assert_eq!(sizes, vec![100.0, 400.0 / 3.0, 800.0 / 3.0]);
//! ```

use alloc::vec::Vec;

use crate::geometry::AvailableSpace;

/// Numerical tolerance used for float comparisons and distribution cut-offs.
const EPS: f32 = 1.0e-4;

/// A single `<track-breadth>`: one component of a track sizing function.
///
/// Used both standalone (as a `<track-size>`) and as the `min` / `max`
/// argument of [`TrackSizingFunction::MinMax`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackBreadth {
    /// `auto`: the track's min-content size as a minimum, max-content as a
    /// maximum.
    Auto,
    /// `min-content`: the largest minimum content contribution in the track.
    MinContent,
    /// `max-content`: the largest maximum content contribution in the track.
    MaxContent,
    /// An absolute length in layout units. Negative values clamp to zero.
    Points(f32),
    /// A fraction of the definite container length, where `1.0` means 100%.
    /// When the container length is indefinite this behaves as `auto`.
    Percent(f32),
    /// A flexible `<flex>` factor (`fr`). Only meaningful as a maximum; when
    /// used as a minimum it is treated as `auto`.
    Fr(f32),
}

impl TrackBreadth {
    fn is_flex(self) -> bool {
        matches!(self, TrackBreadth::Fr(_))
    }
}

/// A CSS Grid track sizing function (one entry of a track template).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackSizingFunction {
    /// A single `<track-breadth>` (e.g. `100px`, `1fr`, `auto`,
    /// `min-content`).
    Breadth(TrackBreadth),
    /// `minmax(min, max)`. Per spec a flexible `min` is invalid and is
    /// treated as `auto`.
    MinMax {
        /// Lower bound breadth.
        min: TrackBreadth,
        /// Upper bound breadth.
        max: TrackBreadth,
    },
    /// `fit-content(limit)`: equivalent to
    /// `minmax(auto, min(max-content, max(auto, limit)))`. `limit` is an
    /// absolute length in layout units.
    FitContent(f32),
}

impl TrackSizingFunction {
    /// `100px` — a fixed length track.
    pub fn points(value: f32) -> Self {
        TrackSizingFunction::Breadth(TrackBreadth::Points(value))
    }

    /// `25%` — a percentage of the definite container length (`1.0` = 100%).
    pub fn percent(fraction: f32) -> Self {
        TrackSizingFunction::Breadth(TrackBreadth::Percent(fraction))
    }

    /// `auto` — min-content floor, max-content ceiling.
    pub fn auto() -> Self {
        TrackSizingFunction::Breadth(TrackBreadth::Auto)
    }

    /// `min-content`.
    pub fn min_content() -> Self {
        TrackSizingFunction::Breadth(TrackBreadth::MinContent)
    }

    /// `max-content`.
    pub fn max_content() -> Self {
        TrackSizingFunction::Breadth(TrackBreadth::MaxContent)
    }

    /// `Nfr` — a flexible track with the given flex factor.
    pub fn fr(factor: f32) -> Self {
        TrackSizingFunction::Breadth(TrackBreadth::Fr(factor))
    }

    /// `minmax(min, max)`.
    pub fn minmax(min: TrackBreadth, max: TrackBreadth) -> Self {
        TrackSizingFunction::MinMax { min, max }
    }

    /// `fit-content(limit)`.
    pub fn fit_content(limit: f32) -> Self {
        TrackSizingFunction::FitContent(limit)
    }

    /// Decomposes the function into its effective `(min, max)` breadths plus
    /// an optional `fit-content` clamp limit.
    fn decompose(self) -> (TrackBreadth, TrackBreadth, Option<f32>) {
        match self {
            TrackSizingFunction::Breadth(TrackBreadth::Fr(f)) => {
                // A standalone flex is equivalent to `minmax(auto, Nfr)`.
                (TrackBreadth::Auto, TrackBreadth::Fr(f), None)
            }
            TrackSizingFunction::Breadth(b) => (b, b, None),
            TrackSizingFunction::MinMax { min, max } => {
                let min = if min.is_flex() { TrackBreadth::Auto } else { min };
                (min, max, None)
            }
            TrackSizingFunction::FitContent(limit) => {
                (TrackBreadth::Auto, TrackBreadth::MaxContent, Some(limit.max(0.0)))
            }
        }
    }
}

/// The intrinsic content contributions of the items assigned to one track.
///
/// `min_content` is the largest minimum size an item in the track needs (the
/// point below which content would overflow); `max_content` is the largest
/// preferred size. Both are clamped to be non-negative and `max_content` is
/// never treated as smaller than `min_content`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackContent {
    /// Largest min-content contribution of the track's items.
    pub min_content: f32,
    /// Largest max-content contribution of the track's items.
    pub max_content: f32,
}

impl TrackContent {
    /// A track with no content, contributing zero to intrinsic sizing.
    pub const ZERO: Self = Self {
        min_content: 0.0,
        max_content: 0.0,
    };

    /// Creates a content pair from a min-content and max-content extent.
    pub fn new(min_content: f32, max_content: f32) -> Self {
        Self {
            min_content,
            max_content,
        }
    }

    /// Creates a content pair whose min- and max-content extents are equal
    /// (an item that cannot stretch or shrink).
    pub fn fixed(extent: f32) -> Self {
        Self {
            min_content: extent,
            max_content: extent,
        }
    }

    /// Returns a copy with non-negative extents and `max_content >=
    /// min_content`.
    fn normalized(self) -> Self {
        let min = self.min_content.max(0.0);
        let max = self.max_content.max(0.0).max(min);
        Self {
            min_content: min,
            max_content: max,
        }
    }
}

/// Expands an integer `repeat(count, template)` into a flat track list.
///
/// This is the deterministic, placement-independent form of CSS
/// `repeat()`. `repeat(auto-fill, …)` and `repeat(auto-fit, …)` depend on the
/// container size and item placement and are not provided here.
pub fn repeat(count: usize, template: &[TrackSizingFunction]) -> Vec<TrackSizingFunction> {
    let mut out = Vec::with_capacity(count.saturating_mul(template.len()));
    for _ in 0..count {
        out.extend_from_slice(template);
    }
    out
}

/// A track's resolved sizing state during the algorithm.
#[derive(Clone, Copy, Debug)]
struct Track {
    /// Current base size (grows during the algorithm, never shrinks).
    base: f32,
    /// Growth limit; may be [`f32::INFINITY`] for flexible / unbounded maxima.
    limit: f32,
    /// Flex factor when the track has a flexible maximum, else `None`.
    flex: Option<f32>,
}

fn resolve_min_base(min: TrackBreadth, content: TrackContent, avail: Option<f32>) -> f32 {
    match min {
        TrackBreadth::Points(p) => p.max(0.0),
        TrackBreadth::Percent(f) => match avail {
            Some(a) => (a * f).max(0.0),
            None => content.min_content,
        },
        TrackBreadth::MaxContent => content.max_content,
        // `Fr` is an invalid minimum and already mapped to `auto`, so it
        // resolves identically to `auto`/`min-content`.
        TrackBreadth::Auto | TrackBreadth::MinContent | TrackBreadth::Fr(_) => {
            content.min_content
        }
    }
}

fn resolve_max_limit(max: TrackBreadth, content: TrackContent, avail: Option<f32>) -> f32 {
    match max {
        TrackBreadth::Points(p) => p.max(0.0),
        TrackBreadth::Percent(f) => match avail {
            Some(a) => (a * f).max(0.0),
            // An indefinite percentage maximum behaves as `auto`.
            None => content.max_content,
        },
        TrackBreadth::Auto | TrackBreadth::MaxContent => content.max_content,
        TrackBreadth::MinContent => content.min_content,
        TrackBreadth::Fr(_) => f32::INFINITY,
    }
}

fn build_track(func: TrackSizingFunction, content: TrackContent, avail: Option<f32>) -> Track {
    let (min_b, max_b, fit_cap) = func.decompose();
    let base = resolve_min_base(min_b, content, avail);
    let mut limit = resolve_max_limit(max_b, content, avail).max(base);
    if let Some(cap) = fit_cap {
        // fit-content clamps the growth limit to `max(base, limit)` while
        // never exceeding the max-content size already stored in `limit`.
        limit = limit.min(cap.max(base)).max(base);
    }
    let flex = match max_b {
        TrackBreadth::Fr(f) => Some(f.max(0.0)),
        _ => None,
    };
    Track { base, limit, flex }
}

/// Grows the base sizes of non-flexible, finite-limit tracks by distributing
/// positive free space equally, freezing each track as it reaches its growth
/// limit (CSS Grid §11.6 "Maximize Tracks").
///
/// Flexible tracks are frozen at their base here; their final size is decided
/// by [`find_fr_size`] afterwards.
fn maximize(space_to_fill: f32, tracks: &mut [Track]) {
    let growable = |t: &Track| t.flex.is_none() && t.limit.is_finite() && t.limit - t.base > EPS;

    let base_sum: f32 = tracks.iter().map(|t| t.base).sum();
    let mut free = space_to_fill - base_sum;
    if free <= EPS {
        return;
    }

    // Each iteration either fully distributes the free space (no track clamps)
    // or freezes at least one track, so the loop runs at most `len + 1` times.
    for _ in 0..=tracks.len() {
        let count = tracks.iter().filter(|t| growable(t)).count();
        if count == 0 || free <= EPS {
            break;
        }
        let share = free / count as f32;
        if share <= EPS {
            break;
        }
        let mut distributed = 0.0;
        for t in tracks.iter_mut().filter(|t| growable(t)) {
            let add = share.min(t.limit - t.base);
            t.base += add;
            distributed += add;
        }
        if distributed <= EPS {
            break;
        }
        free -= distributed;
    }
}

/// Computes the used flex fraction (the size of a single `fr`) for the
/// flexible tracks, following CSS Grid §11.7 "Find the size of an fr".
///
/// `space_to_fill` is the available length minus the sum of fixed gaps. The
/// returned value `f` is used to size each flexible track as
/// `max(base, f * flex_factor)`.
fn find_fr_size(space_to_fill: f32, tracks: &[Track]) -> f32 {
    // A track is "inflexible" either because it is non-flex, or because a
    // previous iteration demoted it (its base exceeds its hypothetical size).
    let mut inflexible: Vec<bool> = tracks.iter().map(|t| t.flex.is_none()).collect();

    // Each demotion freezes at least one flexible track, bounding iterations.
    for _ in 0..=tracks.len() {
        let mut leftover = space_to_fill;
        let mut factor_sum = 0.0_f32;
        let mut any_flex = false;
        for (t, &inf) in tracks.iter().zip(inflexible.iter()) {
            if inf {
                leftover -= t.base;
            } else if let Some(f) = t.flex {
                factor_sum += f;
                any_flex = true;
            }
        }
        if !any_flex || leftover <= EPS {
            return 0.0;
        }
        // A flex factor sum below 1 is clamped to 1 so sub-`1fr` tracks do not
        // over-fill the container.
        let hypothetical = leftover / factor_sum.max(1.0);

        let mut changed = false;
        for (t, inf) in tracks.iter().zip(inflexible.iter_mut()) {
            if !*inf
                && let Some(f) = t.flex
                && hypothetical * f < t.base - EPS
            {
                *inf = true;
                changed = true;
            }
        }
        if !changed {
            return hypothetical;
        }
    }
    0.0
}

fn gaps_count(track_count: usize) -> f32 {
    if track_count <= 1 {
        0.0
    } else {
        (track_count - 1) as f32
    }
}

/// Resolves a track template into concrete per-track pixel sizes.
///
/// `tracks` is the track template (`grid-template-columns` /
/// `grid-template-rows`). `content[i]` supplies the intrinsic content
/// contributions of track `i`; a shorter slice is treated as [`TrackContent::ZERO`]
/// for the missing trailing tracks. `available` is the inner length of the
/// container on this axis, and `gap` is the fixed gap between adjacent tracks
/// (negative gaps clamp to zero).
///
/// The returned vector has the same length as `tracks`. Sizes always satisfy
/// each track's resolved minimum and may collectively exceed `available` when
/// the minimums do not fit (the normal CSS overflow case).
pub fn resolve_track_sizes(
    tracks: &[TrackSizingFunction],
    content: &[TrackContent],
    available: AvailableSpace,
    gap: f32,
) -> Vec<f32> {
    if tracks.is_empty() {
        return Vec::new();
    }

    let avail = available.into_option();
    let mut states: Vec<Track> = tracks
        .iter()
        .enumerate()
        .map(|(i, &func)| {
            let c = content.get(i).copied().unwrap_or(TrackContent::ZERO).normalized();
            build_track(func, c, avail)
        })
        .collect();

    match available {
        // Under a min-content constraint every track collapses to its base.
        AvailableSpace::MinContent => states.iter().map(|t| t.base).collect(),
        // Under a max-content constraint tracks grow to their growth limit;
        // flexible (unbounded) tracks stay at their base.
        AvailableSpace::MaxContent => states
            .iter()
            .map(|t| if t.limit.is_finite() { t.limit } else { t.base })
            .collect(),
        AvailableSpace::Definite(a) => {
            let gap = gap.max(0.0);
            let space = (a - gap * gaps_count(states.len())).max(0.0);
            let has_flex = states.iter().any(|t| t.flex.is_some());

            // Grow intrinsic / minmax tracks toward their growth limits first.
            maximize(space, &mut states);

            if has_flex {
                let fr = find_fr_size(space, &states);
                states
                    .iter()
                    .map(|t| match t.flex {
                        Some(f) => (fr * f).max(t.base),
                        None => t.base,
                    })
                    .collect()
            } else {
                states.iter().map(|t| t.base).collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() <= 1.0e-2, "expected {b}, got {a}");
    }

    fn approx_slice(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "length mismatch: {got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want.iter()) {
            approx(*g, *w);
        }
    }

    #[test]
    fn fixed_tracks_pass_through() {
        let tracks = [
            TrackSizingFunction::points(100.0),
            TrackSizingFunction::points(50.0),
        ];
        let sizes = resolve_track_sizes(
            &tracks,
            &[TrackContent::ZERO; 2],
            AvailableSpace::Definite(1000.0),
            0.0,
        );
        approx_slice(&sizes, &[100.0, 50.0]);
    }

    #[test]
    fn fixed_plus_single_fr_fills_remainder() {
        let tracks = [TrackSizingFunction::points(100.0), TrackSizingFunction::fr(1.0)];
        let sizes = resolve_track_sizes(
            &tracks,
            &[TrackContent::ZERO; 2],
            AvailableSpace::Definite(500.0),
            0.0,
        );
        approx_slice(&sizes, &[100.0, 400.0]);
    }

    #[test]
    fn fr_distributes_proportionally() {
        let tracks = [
            TrackSizingFunction::fr(1.0),
            TrackSizingFunction::fr(1.0),
            TrackSizingFunction::fr(2.0),
        ];
        let sizes = resolve_track_sizes(
            &tracks,
            &[TrackContent::ZERO; 3],
            AvailableSpace::Definite(400.0),
            0.0,
        );
        approx_slice(&sizes, &[100.0, 100.0, 200.0]);
    }

    #[test]
    fn sub_unit_fr_sum_is_clamped_to_one() {
        // A lone 0.5fr track does NOT fill the container: the flex factor
        // sum (0.5) is clamped up to 1, so one `fr` is 100px and the track
        // takes 0.5 * 100 = 50px, leaving 50px of free (content-distribution)
        // space. This clamp is exactly what prevents sub-1fr over-filling.
        let sizes = resolve_track_sizes(
            &[TrackSizingFunction::fr(0.5)],
            &[TrackContent::ZERO],
            AvailableSpace::Definite(100.0),
            0.0,
        );
        approx_slice(&sizes, &[50.0]);

        // Two 0.5fr tracks sum to 1.0 and split evenly.
        let sizes = resolve_track_sizes(
            &[TrackSizingFunction::fr(0.5), TrackSizingFunction::fr(0.5)],
            &[TrackContent::ZERO; 2],
            AvailableSpace::Definite(100.0),
            0.0,
        );
        approx_slice(&sizes, &[50.0, 50.0]);
    }

    #[test]
    fn minmax_maximizes_to_growth_limit_then_fr_fills_remainder() {
        // Per CSS Grid §11.6, non-flex tracks are grown to their growth
        // limit (here 100px) before §11.7 expands flexible tracks, so the
        // `minmax(50px, 100px)` track resolves to 100px and the 1fr track
        // takes the remaining 900px of a 1000px container.
        let tracks = [
            TrackSizingFunction::minmax(TrackBreadth::Points(50.0), TrackBreadth::Points(100.0)),
            TrackSizingFunction::fr(1.0),
        ];
        let sizes = resolve_track_sizes(
            &tracks,
            &[TrackContent::ZERO; 2],
            AvailableSpace::Definite(1000.0),
            0.0,
        );
        approx_slice(&sizes, &[100.0, 900.0]);
    }

    #[test]
    fn auto_track_grows_to_max_content_then_fr_fills() {
        // 100px auto 1fr, auto content min=50 max=200, width 1000.
        let tracks = [
            TrackSizingFunction::points(100.0),
            TrackSizingFunction::auto(),
            TrackSizingFunction::fr(1.0),
        ];
        let content = [
            TrackContent::ZERO,
            TrackContent::new(50.0, 200.0),
            TrackContent::ZERO,
        ];
        let sizes = resolve_track_sizes(&tracks, &content, AvailableSpace::Definite(1000.0), 0.0);
        approx_slice(&sizes, &[100.0, 200.0, 700.0]);
    }

    #[test]
    fn no_flex_maximizes_up_to_growth_limits() {
        // minmax(0,100) minmax(0,100) in 1000px: no fr, so both grow to limit.
        let tracks = [
            TrackSizingFunction::minmax(TrackBreadth::Points(0.0), TrackBreadth::Points(100.0)),
            TrackSizingFunction::minmax(TrackBreadth::Points(0.0), TrackBreadth::Points(100.0)),
        ];
        let sizes = resolve_track_sizes(
            &tracks,
            &[TrackContent::ZERO; 2],
            AvailableSpace::Definite(1000.0),
            0.0,
        );
        approx_slice(&sizes, &[100.0, 100.0]);
    }

    #[test]
    fn fit_content_clamps_between_limit_and_max_content() {
        // fit-content(150) with content max 300 -> 150.
        let sizes = resolve_track_sizes(
            &[TrackSizingFunction::fit_content(150.0)],
            &[TrackContent::new(0.0, 300.0)],
            AvailableSpace::Definite(1000.0),
            0.0,
        );
        approx_slice(&sizes, &[150.0]);

        // fit-content(150) with content max 100 -> 100 (never exceeds content).
        let sizes = resolve_track_sizes(
            &[TrackSizingFunction::fit_content(150.0)],
            &[TrackContent::new(0.0, 100.0)],
            AvailableSpace::Definite(1000.0),
            0.0,
        );
        approx_slice(&sizes, &[100.0]);
    }

    #[test]
    fn percentages_resolve_against_definite_length() {
        let tracks = [
            TrackSizingFunction::percent(0.5),
            TrackSizingFunction::percent(0.5),
        ];
        let sizes = resolve_track_sizes(
            &tracks,
            &[TrackContent::ZERO; 2],
            AvailableSpace::Definite(400.0),
            0.0,
        );
        approx_slice(&sizes, &[200.0, 200.0]);
    }

    #[test]
    fn gaps_reduce_the_space_available_to_tracks() {
        let tracks = [TrackSizingFunction::fr(1.0), TrackSizingFunction::fr(1.0)];
        let sizes = resolve_track_sizes(
            &tracks,
            &[TrackContent::ZERO; 2],
            AvailableSpace::Definite(100.0),
            20.0,
        );
        // space_to_fill = 100 - 20 = 80, split evenly.
        approx_slice(&sizes, &[40.0, 40.0]);
        approx(sizes.iter().sum::<f32>() + 20.0, 100.0);
    }

    #[test]
    fn repeat_expands_the_template() {
        let expanded = repeat(3, &[TrackSizingFunction::fr(1.0)]);
        assert_eq!(expanded.len(), 3);
        let sizes = resolve_track_sizes(
            &expanded,
            &[TrackContent::ZERO; 3],
            AvailableSpace::Definite(300.0),
            0.0,
        );
        approx_slice(&sizes, &[100.0, 100.0, 100.0]);
    }

    #[test]
    fn min_and_max_content_available_space() {
        let tracks = [
            TrackSizingFunction::auto(),
            TrackSizingFunction::fr(1.0),
        ];
        let content = [TrackContent::new(30.0, 120.0), TrackContent::ZERO];

        let min = resolve_track_sizes(&tracks, &content, AvailableSpace::MinContent, 0.0);
        approx_slice(&min, &[30.0, 0.0]);

        let max = resolve_track_sizes(&tracks, &content, AvailableSpace::MaxContent, 0.0);
        // auto -> max-content 120; fr has no definite basis -> base 0.
        approx_slice(&max, &[120.0, 0.0]);
    }

    #[test]
    fn minimums_overflow_when_they_do_not_fit() {
        let tracks = [
            TrackSizingFunction::points(200.0),
            TrackSizingFunction::points(200.0),
        ];
        let sizes = resolve_track_sizes(
            &tracks,
            &[TrackContent::ZERO; 2],
            AvailableSpace::Definite(100.0),
            0.0,
        );
        approx_slice(&sizes, &[200.0, 200.0]);
    }

    #[test]
    fn fr_track_respects_its_base_minimum() {
        // minmax(300px, 1fr) alongside 1fr in 1000px: the first track cannot
        // shrink below 300 even though an even split would give 500 each...
        // here 300 >= its even share only when squeezed; use a tight box.
        let tracks = [
            TrackSizingFunction::minmax(TrackBreadth::Points(300.0), TrackBreadth::Fr(1.0)),
            TrackSizingFunction::fr(1.0),
        ];
        // In 400px: first track floored at 300, second gets the remaining 100.
        let sizes = resolve_track_sizes(
            &tracks,
            &[TrackContent::ZERO; 2],
            AvailableSpace::Definite(400.0),
            0.0,
        );
        approx_slice(&sizes, &[300.0, 100.0]);
    }

    #[test]
    fn empty_template_resolves_to_empty() {
        let sizes = resolve_track_sizes(&[], &[], AvailableSpace::Definite(500.0), 10.0);
        assert!(sizes.is_empty());
    }

    #[test]
    fn resolution_is_deterministic() {
        let tracks = [
            TrackSizingFunction::points(40.0),
            TrackSizingFunction::fr(1.0),
            TrackSizingFunction::minmax(TrackBreadth::Points(20.0), TrackBreadth::Fr(2.0)),
            TrackSizingFunction::auto(),
        ];
        let content = [
            TrackContent::ZERO,
            TrackContent::ZERO,
            TrackContent::ZERO,
            TrackContent::new(10.0, 60.0),
        ];
        let a = resolve_track_sizes(&tracks, &content, AvailableSpace::Definite(777.0), 7.0);
        let b = resolve_track_sizes(&tracks, &content, AvailableSpace::Definite(777.0), 7.0);
        assert_eq!(a, b);
    }

    // --- Independent oracle for the fr distribution fixpoint ---------------

    /// Brute-force reference for [`find_fr_size`]: searches every subset of
    /// the flexible tracks for the unique self-consistent "flexed" set and
    /// returns the implied fr size. Independent of the iterative demotion
    /// order used by the production code.
    fn oracle_fr(space: f32, tracks: &[Track]) -> f32 {
        let flex_idx: Vec<usize> = (0..tracks.len()).filter(|&i| tracks[i].flex.is_some()).collect();
        let m = flex_idx.len();
        if m == 0 {
            return 0.0;
        }
        let nonflex_base: f32 = tracks
            .iter()
            .filter(|t| t.flex.is_none())
            .map(|t| t.base)
            .sum();

        // Several flexed subsets can each be internally self-consistent
        // (over-demoting also balances). The spec iteration demotes only
        // when forced, yielding the *largest* flexed set, which is the one
        // with the maximum fr size. So search every subset and keep the
        // largest valid fr.
        let mut best = 0.0_f32;
        for mask in 0u32..(1u32 << m) {
            let mut leftover = space - nonflex_base;
            let mut factor_sum = 0.0_f32;
            for (bit, &ti) in flex_idx.iter().enumerate() {
                let flexed = (mask >> bit) & 1 == 1;
                if flexed {
                    factor_sum += tracks[ti].flex.unwrap();
                } else {
                    leftover -= tracks[ti].base;
                }
            }
            // The empty flexed set is the "all tracks demoted" terminal; it
            // implies an fr size of 0 but is only the answer when no
            // non-empty self-consistent set exists, so handle it by falling
            // through to the trailing `0.0` rather than returning early.
            if mask == 0 {
                continue;
            }
            if leftover <= EPS {
                continue;
            }
            let f = leftover / factor_sum.max(1.0);
            let mut valid = true;
            for &ti in &flex_idx {
                let factor = tracks[ti].flex.unwrap();
                let flexed = {
                    let bit = flex_idx.iter().position(|&x| x == ti).unwrap();
                    (mask >> bit) & 1 == 1
                };
                if flexed {
                    if f * factor < tracks[ti].base - EPS {
                        valid = false;
                        break;
                    }
                } else if f * factor >= tracks[ti].base - EPS {
                    valid = false;
                    break;
                }
            }
            if valid && f > best {
                best = f;
            }
        }
        best
    }

    /// Tiny deterministic LCG so the property test needs no external deps.
    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            let t = (self.next_u32() as f32) / (u32::MAX as f32);
            lo + t * (hi - lo)
        }
    }

    #[test]
    fn fr_size_matches_brute_force_oracle() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        for _ in 0..600 {
            let n = 2 + (rng.next_u32() as usize % 5); // 2..=6 tracks
            let mut tracks = Vec::with_capacity(n);
            let mut has_flex = false;
            for _ in 0..n {
                let roll = rng.next_u32() % 3;
                if roll == 0 {
                    // Fixed track.
                    let base = rng.range(0.0, 120.0);
                    tracks.push(Track {
                        base,
                        limit: base,
                        flex: None,
                    });
                } else {
                    // Flexible track with a random base floor and factor.
                    has_flex = true;
                    let base = rng.range(0.0, 80.0);
                    let factor = rng.range(0.2, 3.0);
                    tracks.push(Track {
                        base,
                        limit: f32::INFINITY,
                        flex: Some(factor),
                    });
                }
            }
            if !has_flex {
                continue;
            }
            // Choose a space that is comfortably larger than all bases so the
            // interesting (leftover > 0) branch is exercised.
            let base_sum: f32 = tracks.iter().map(|t| t.base).sum();
            let space = base_sum + rng.range(50.0, 500.0);

            let got = find_fr_size(space, &tracks);
            let want = oracle_fr(space, &tracks);
            approx(got, want);

            // Structural invariant: for the production fr size, flexed tracks
            // meet their base and the fill never exceeds the space.
            let total: f32 = tracks
                .iter()
                .map(|t| match t.flex {
                    Some(f) => (got * f).max(t.base),
                    None => t.base,
                })
                .sum();
            assert!(total <= space + 1.0e-1, "overfilled: {total} > {space}");
        }
    }
}
