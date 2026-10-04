//! Swipe / fling direction classification from a release velocity.
//!
//! A [`VelocityTracker`](crate::velocity::VelocityTracker) turns a drag
//! *release* into a velocity vector, but momentum scrolling, page navigation,
//! and dismiss gestures usually act on a single *discrete* direction rather
//! than the raw vector. This module performs that final step: it collapses a
//! velocity into one of four cardinal [`SwipeDirection`]s when the motion is
//! fast enough and clearly dominated by one axis, mirroring the `onFling`
//! callback found in Android's `GestureDetector` and the directional swipe
//! recognizers of Flutter and iOS.
//!
//! # Dominant-axis model
//!
//! A swipe is reported only when the faster axis both clears a minimum speed
//! and sufficiently outpaces the slower axis, which rejects ambiguous diagonal
//! motion. The reported [`Swipe::speed`] is the dominant axis' absolute
//! component speed, **not** the vector magnitude: computing the magnitude would
//! need a square root, and this crate keeps its gesture math to deterministic
//! `+ - * /` and comparisons so results are bit-reproducible across targets.
//!
//! # Axis convention
//!
//! Velocities use the same screen-space axes as [`Point`](crate::geometry::Point):
//! `x` grows rightward and `y` grows downward. A positive `x` velocity is
//! therefore a [`SwipeDirection::Right`] swipe and a positive `y` velocity is a
//! [`SwipeDirection::Down`] swipe.
//!
//! ```
//! use prism_ui_input::geometry::Point;
//! use prism_ui_input::{classify_swipe, SwipeConfig, SwipeDirection};
//!
//! let config = SwipeConfig::new(50.0);
//! // Fast, clearly horizontal release.
//! let swipe = classify_swipe(Point::new(-800.0, 60.0), &config).expect("swipe");
//! assert_eq!(swipe.direction, SwipeDirection::Left);
//! assert!((swipe.speed - 800.0).abs() < f32::EPSILON);
//!
//! // Too slow to count as a fling.
//! assert!(classify_swipe(Point::new(10.0, 5.0), &config).is_none());
//! ```

use crate::geometry::Point;

/// One of the four cardinal swipe directions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SwipeDirection {
    /// Motion toward decreasing `y` (upward on screen).
    Up,
    /// Motion toward increasing `y` (downward on screen).
    Down,
    /// Motion toward decreasing `x` (leftward on screen).
    Left,
    /// Motion toward increasing `x` (rightward on screen).
    Right,
}

/// Thresholds that decide whether a velocity counts as a swipe.
///
/// `min_speed` is the smallest dominant-axis speed (in logical pixels per
/// second) that is accepted as an intentional fling; slower releases return
/// `None`. `min_dominance` is how many times faster the dominant axis must be
/// than the other axis, which rejects diagonal motion: a value of `1.0`
/// accepts any axis that is at least tied for fastest, while larger values
/// (`1.5`-`2.0` are typical) demand a clearly single-axis gesture.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SwipeConfig {
    /// Minimum dominant-axis speed, in logical pixels per second.
    pub min_speed: f32,
    /// Required ratio of the dominant axis speed to the other axis speed.
    pub min_dominance: f32,
}

impl SwipeConfig {
    /// Default dominance ratio: the dominant axis need only match or exceed the
    /// other axis, leaving the axis tie-break rule in [`classify_swipe`] to
    /// resolve a perfectly diagonal release.
    pub const DEFAULT_MIN_DOMINANCE: f32 = 1.0;

    /// Creates a config with the given minimum speed and the default dominance
    /// ratio of [`SwipeConfig::DEFAULT_MIN_DOMINANCE`].
    #[must_use]
    pub const fn new(min_speed: f32) -> Self {
        Self {
            min_speed,
            min_dominance: Self::DEFAULT_MIN_DOMINANCE,
        }
    }
}

impl Default for SwipeConfig {
    /// A `50` px/s floor with the default dominance ratio, a reasonable
    /// starting point for touch-sized fling detection.
    fn default() -> Self {
        Self::new(50.0)
    }
}

/// A recognized swipe: a discrete [`SwipeDirection`] and the dominant-axis
/// speed that produced it, in logical pixels per second.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Swipe {
    /// The cardinal direction of the swipe.
    pub direction: SwipeDirection,
    /// The absolute speed of the dominant axis, in logical pixels per second.
    pub speed: f32,
}

/// Classifies a release `velocity` into a discrete [`Swipe`], or returns `None`
/// when the motion is too slow or too diagonal to be an intentional fling.
///
/// The faster axis is chosen as the dominant one; on an exact tie the
/// horizontal axis wins, making the result deterministic. The dominant axis
/// must reach `config.min_speed` and be at least `config.min_dominance` times
/// the other axis' speed. The returned [`Swipe::speed`] is the dominant axis'
/// absolute component.
#[must_use]
pub fn classify_swipe(velocity: Point<f32>, config: &SwipeConfig) -> Option<Swipe> {
    let ax = abs(velocity.x);
    let ay = abs(velocity.y);

    // Horizontal is dominant on a tie so the classification is deterministic.
    let horizontal = ax >= ay;
    let dominant = if horizontal { ax } else { ay };
    let other = if horizontal { ay } else { ax };

    if dominant < config.min_speed {
        return None;
    }
    // `dominant >= other * min_dominance`, written as a single comparison so a
    // non-finite product cannot slip past the threshold.
    if dominant < other * config.min_dominance {
        return None;
    }

    let direction = if horizontal {
        if velocity.x > 0.0 {
            SwipeDirection::Right
        } else {
            SwipeDirection::Left
        }
    } else if velocity.y > 0.0 {
        SwipeDirection::Down
    } else {
        SwipeDirection::Up
    };

    Some(Swipe {
        direction,
        speed: dominant,
    })
}

/// Absolute value without `f32::abs`, keeping the gesture math obviously pure.
fn abs(value: f32) -> f32 {
    if value < 0.0 { -value } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            let t = (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32);
            lo + t * (hi - lo)
        }
    }

    /// Independent reference: recompute the predicate from scratch.
    fn oracle(vx: f32, vy: f32, config: &SwipeConfig) -> Option<Swipe> {
        let ax = if vx < 0.0 { -vx } else { vx };
        let ay = if vy < 0.0 { -vy } else { vy };
        let horizontal = ax >= ay;
        let (dominant, other) = if horizontal { (ax, ay) } else { (ay, ax) };
        if dominant < config.min_speed {
            return None;
        }
        if dominant < other * config.min_dominance {
            return None;
        }
        let direction = if horizontal {
            if vx > 0.0 {
                SwipeDirection::Right
            } else {
                SwipeDirection::Left
            }
        } else if vy > 0.0 {
            SwipeDirection::Down
        } else {
            SwipeDirection::Up
        };
        Some(Swipe {
            direction,
            speed: dominant,
        })
    }

    #[test]
    fn matches_independent_oracle_over_random_inputs() {
        let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
        for _ in 0..20_000 {
            let vx = rng.range(-2000.0, 2000.0);
            let vy = rng.range(-2000.0, 2000.0);
            let min_speed = rng.range(0.0, 1500.0);
            let min_dominance = rng.range(1.0, 3.0);
            let config = SwipeConfig {
                min_speed,
                min_dominance,
            };

            let got = classify_swipe(Point::new(vx, vy), &config);
            let expected = oracle(vx, vy, &config);
            assert_eq!(got, expected, "vx={vx} vy={vy} config={config:?}");
        }
    }

    #[test]
    fn fixed_cardinal_directions() {
        let config = SwipeConfig::new(100.0);
        assert_eq!(
            classify_swipe(Point::new(500.0, 0.0), &config).unwrap().direction,
            SwipeDirection::Right
        );
        assert_eq!(
            classify_swipe(Point::new(-500.0, 0.0), &config).unwrap().direction,
            SwipeDirection::Left
        );
        assert_eq!(
            classify_swipe(Point::new(0.0, 500.0), &config).unwrap().direction,
            SwipeDirection::Down
        );
        assert_eq!(
            classify_swipe(Point::new(0.0, -500.0), &config).unwrap().direction,
            SwipeDirection::Up
        );
    }

    #[test]
    fn speed_is_dominant_axis_component() {
        let config = SwipeConfig::new(100.0);
        let swipe = classify_swipe(Point::new(-900.0, 120.0), &config).unwrap();
        assert_eq!(swipe.direction, SwipeDirection::Left);
        assert!((swipe.speed - 900.0).abs() < f32::EPSILON);
    }

    #[test]
    fn below_min_speed_is_rejected() {
        let config = SwipeConfig::new(300.0);
        assert!(classify_swipe(Point::new(100.0, 20.0), &config).is_none());
    }

    #[test]
    fn diagonal_rejected_by_dominance() {
        // Near-equal axes: dominant barely exceeds other, fails a 1.5 ratio.
        let config = SwipeConfig {
            min_speed: 100.0,
            min_dominance: 1.5,
        };
        assert!(classify_swipe(Point::new(400.0, 390.0), &config).is_none());
        // Make x clearly dominant and it passes.
        assert_eq!(
            classify_swipe(Point::new(800.0, 390.0), &config).unwrap().direction,
            SwipeDirection::Right
        );
    }

    #[test]
    fn axis_tie_prefers_horizontal() {
        // |vx| == |vy|: horizontal wins the tie-break deterministically.
        let config = SwipeConfig::new(100.0);
        let swipe = classify_swipe(Point::new(-300.0, 300.0), &config).unwrap();
        assert_eq!(swipe.direction, SwipeDirection::Left);
        assert!((swipe.speed - 300.0).abs() < f32::EPSILON);
    }

    #[test]
    fn zero_velocity_is_none() {
        let config = SwipeConfig::default();
        assert!(classify_swipe(Point::new(0.0, 0.0), &config).is_none());
    }

    #[test]
    fn default_config_values() {
        let config = SwipeConfig::default();
        assert!((config.min_speed - 50.0).abs() < f32::EPSILON);
        assert!((config.min_dominance - SwipeConfig::DEFAULT_MIN_DOMINANCE).abs() < f32::EPSILON);
    }
}
