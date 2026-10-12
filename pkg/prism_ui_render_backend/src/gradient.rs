//! Linear gradients: project a point onto an axis and blend the color stops.
//!
//! A modern UI surface almost never paints a flat fill — buttons, cards,
//! progress bars and backdrops use a gradient. [`LinearGradient`] is the core
//! primitive: an axis (`start` → `end`) plus ordered [`Stop`]s. Sampling a point
//! projects it onto the axis to a parameter `t` in `0..=1` (clamped) and blends
//! the surrounding stops with [`Color::lerp`](prism_ui_style::Color::lerp), so
//! the shading reuses the same linear-space color math as everything else.
//!
//! Stops are normalised once at construction — offsets clamped into `0..=1` and
//! sorted — so sampling never has to re-order them. All arithmetic is `+ - * /`
//! and comparisons on `f32`; there are no transcendentals, so the module is
//! `no_std`-clean and bit-stable.

use alloc::vec::Vec;

use prism_ui_style::Color;

/// A single color stop at `offset` (`0..=1`) along a gradient axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stop {
    /// Position along the axis, clamped into `0..=1` on construction.
    pub offset: f32,
    /// The color at that position.
    pub color: Color,
}

impl Stop {
    /// A stop at `offset` with `color`.
    #[must_use]
    pub const fn new(offset: f32, color: Color) -> Self {
        Self { offset, color }
    }
}

/// A linear gradient from `start` to `end`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearGradient {
    start: (f32, f32),
    end: (f32, f32),
    stops: Vec<Stop>,
}

impl LinearGradient {
    /// Build a gradient, clamping every stop offset into `0..=1` (a `NaN`
    /// offset becomes `0`) and sorting the stops by offset with a **stable**
    /// sort, so equal offsets keep the order they were given (which lets a
    /// two-stop pair at the same offset act as a hard color break).
    #[must_use]
    pub fn new(start: (f32, f32), end: (f32, f32), mut stops: Vec<Stop>) -> Self {
        for stop in &mut stops {
            stop.offset = if stop.offset.is_nan() {
                0.0
            } else {
                stop.offset.clamp(0.0, 1.0)
            };
        }
        stops.sort_by(|a, b| a.offset.total_cmp(&b.offset));
        Self { start, end, stops }
    }

    /// The axis start point.
    #[must_use]
    pub fn start(&self) -> (f32, f32) {
        self.start
    }

    /// The axis end point.
    #[must_use]
    pub fn end(&self) -> (f32, f32) {
        self.end
    }

    /// The normalised stops, sorted by offset.
    #[must_use]
    pub fn stops(&self) -> &[Stop] {
        &self.stops
    }

    /// The axis parameter `t` of point `p`, clamped into `0..=1`.
    ///
    /// `t` is the scalar projection of `p - start` onto the axis, normalised by
    /// the axis length. A degenerate axis (`start == end`) maps every point to
    /// `0`.
    #[must_use]
    pub fn project(&self, p: (f32, f32)) -> f32 {
        let (dx, dy) = (self.end.0 - self.start.0, self.end.1 - self.start.1);
        let len2 = dx * dx + dy * dy;
        if len2 <= 0.0 {
            return 0.0;
        }
        let t = ((p.0 - self.start.0) * dx + (p.1 - self.start.1) * dy) / len2;
        t.clamp(0.0, 1.0)
    }

    /// Sample the gradient color at point `p` (equivalent to
    /// `sample_t(project(p))`).
    #[must_use]
    pub fn sample(&self, p: (f32, f32)) -> Color {
        self.sample_t(self.project(p))
    }

    /// Sample the gradient color at axis parameter `t` (clamped into `0..=1`; a
    /// `NaN` becomes `0`).
    ///
    /// Before the first stop the first color is returned, after the last stop
    /// the last color, and between two stops the colors are blended by
    /// [`Color::lerp`](prism_ui_style::Color::lerp). Two stops sharing an offset
    /// produce a hard break: up to and including the offset the earlier
    /// interval's end color is used, and strictly after it the later stop's
    /// color begins. No stops yields [`Color::TRANSPARENT`]. The scan is linear
    /// in the stop count, which is a handful in practice.
    #[must_use]
    pub fn sample_t(&self, t: f32) -> Color {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
        let Some((first, rest)) = self.stops.split_first() else {
            return Color::TRANSPARENT;
        };
        if t <= first.offset {
            return first.color;
        }
        let mut prev = *first;
        for stop in rest {
            if t <= stop.offset {
                let span = stop.offset - prev.offset;
                if span <= 0.0 {
                    return stop.color;
                }
                return prev.color.lerp(stop.color, (t - prev.offset) / span);
            }
            prev = *stop;
        }
        prev.color
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::std_instead_of_alloc,
        reason = "tests run under std and use std helpers only"
    )]
    use super::*;
    use alloc::vec;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn colors_close(a: Color, b: Color) -> bool {
        close(a.r, b.r) && close(a.g, b.g) && close(a.b, b.b) && close(a.a, b.a)
    }

    fn bw(start: (f32, f32), end: (f32, f32)) -> LinearGradient {
        LinearGradient::new(
            start,
            end,
            vec![Stop::new(0.0, Color::BLACK), Stop::new(1.0, Color::WHITE)],
        )
    }

    #[test]
    fn endpoints_and_midpoint_blend() {
        let g = bw((0.0, 0.0), (10.0, 0.0));
        assert!(colors_close(g.sample((0.0, 0.0)), Color::BLACK));
        assert!(colors_close(g.sample((10.0, 0.0)), Color::WHITE));
        // Midpoint is the channel-wise lerp (reuses Color::lerp).
        assert!(colors_close(
            g.sample((5.0, 0.0)),
            Color::BLACK.lerp(Color::WHITE, 0.5)
        ));
        assert!(close(g.sample((2.5, 0.0)).r, 0.25));
    }

    #[test]
    fn perpendicular_points_share_a_color() {
        let g = bw((0.0, 0.0), (10.0, 0.0));
        // Only the projection on the axis matters.
        assert!(colors_close(g.sample((3.0, -7.0)), g.sample((3.0, 42.0))));
    }

    #[test]
    fn sampling_clamps_beyond_the_axis() {
        let g = bw((0.0, 0.0), (10.0, 0.0));
        assert!(colors_close(g.sample((-100.0, 0.0)), Color::BLACK));
        assert!(colors_close(g.sample((100.0, 0.0)), Color::WHITE));
    }

    #[test]
    fn equal_offsets_form_a_hard_break() {
        let g = LinearGradient::new(
            (0.0, 0.0),
            (1.0, 0.0),
            vec![
                Stop::new(0.0, Color::BLACK),
                Stop::new(0.5, Color::WHITE),
                Stop::new(0.5, Color::BLACK),
                Stop::new(1.0, Color::WHITE),
            ],
        );
        // At the shared offset the first ramp reaches WHITE; the second ramp then
        // starts from BLACK (a hard break, no blend across it).
        assert!(colors_close(g.sample_t(0.5), Color::WHITE));
        // 0.6 is 20% of the way along the second (BLACK -> WHITE) ramp.
        assert!(close(g.sample_t(0.6).r, 0.2));
    }

    #[test]
    fn stops_are_clamped_and_sorted_on_construction() {
        let g = LinearGradient::new(
            (0.0, 0.0),
            (1.0, 0.0),
            vec![
                Stop::new(2.0, Color::WHITE), // clamped to 1.0
                Stop::new(-1.0, Color::BLACK), // clamped to 0.0
            ],
        );
        assert_eq!(g.stops()[0].offset, 0.0);
        assert_eq!(g.stops()[1].offset, 1.0);
        assert!(colors_close(g.sample_t(0.0), Color::BLACK));
        assert!(colors_close(g.sample_t(1.0), Color::WHITE));
    }

    #[test]
    fn single_stop_is_constant_and_empty_is_transparent() {
        let one = LinearGradient::new((0.0, 0.0), (1.0, 0.0), vec![Stop::new(0.3, Color::WHITE)]);
        assert!(colors_close(one.sample_t(0.0), Color::WHITE));
        assert!(colors_close(one.sample_t(0.9), Color::WHITE));
        let empty = LinearGradient::new((0.0, 0.0), (1.0, 0.0), vec![]);
        assert!(colors_close(empty.sample_t(0.5), Color::TRANSPARENT));
    }

    #[test]
    fn degenerate_axis_maps_to_the_first_stop() {
        let g = bw((3.0, 3.0), (3.0, 3.0));
        assert_eq!(g.project((9.0, -2.0)), 0.0);
        assert!(colors_close(g.sample((9.0, -2.0)), Color::BLACK));
    }
}
