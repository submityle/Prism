//! **M5 — per-world / local time domains.** Layered time-scale scopes.
//!
//! Beyond the global [`Time<Virtual>`](crate::Time) scale, AAA engines need
//! *local* time scaling: a background world paused while another runs, or a
//! localized slow-motion field around a boss ability. A [`TimeScaleDomain`] is
//! one such scope. Domains compose multiplicatively, so a child domain's rate
//! layers on top of its parent (and ultimately the global clock), per the
//! design doc's "域缩放叠加在全局 scale 之上".
//!
//! A domain can hold an eased [`ScaleTransition`](crate::ScaleTransition) so a
//! scope can smoothly ramp its rate (bullet-time in/out) rather than snapping.
//! All math is deterministic `f64`/[`Duration`] arithmetic, so composed scales
//! are bit-identical across runs — the per-world determinism contract.

use crate::{Duration, Easing, ScaleTransition};

/// A local time-scale scope layered over its parent (world/global) time.
///
/// The *effective* scale is the requested scale (or the in-flight eased ramp's
/// current value) multiplied by `0.0` when paused. [`scale_delta`] applies the
/// effective scale to an incoming parent delta; compose scopes by feeding one
/// domain's output into the next.
///
/// [`scale_delta`]: TimeScaleDomain::scale_delta
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeScaleDomain {
    /// Smooth ramp carrying the current (possibly transitioning) scale. When
    /// settled, `current()` is the steady requested scale.
    transition: ScaleTransition,
    /// Whether this scope is frozen (effective scale `0.0`).
    paused: bool,
}

impl TimeScaleDomain {
    /// A pass-through domain: scale `1.0`, not paused.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Self::new(1.0)
    }

    /// A domain at a fixed `scale` (clamped to `>= 0`, non-finite ignored to
    /// `1.0`), not paused.
    #[inline]
    #[must_use]
    pub fn new(scale: f64) -> Self {
        Self {
            transition: ScaleTransition::settled(sanitize_scale(scale, 1.0)),
            paused: false,
        }
    }

    /// A domain created already paused at `scale`.
    #[inline]
    #[must_use]
    pub fn paused_at(scale: f64) -> Self {
        let mut d = Self::new(scale);
        d.paused = true;
        d
    }

    /// The requested (steady or in-flight) scale, ignoring pause.
    #[inline]
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.transition.current()
    }

    /// The effective scale actually applied: the current scale, or `0.0` while
    /// paused.
    #[inline]
    #[must_use]
    pub fn effective_scale(&self) -> f64 {
        if self.paused {
            0.0
        } else {
            self.transition.current().max(0.0)
        }
    }

    /// Set the scale immediately (no ramp). Clamped to `>= 0`; non-finite
    /// values are ignored.
    #[inline]
    pub fn set_scale(&mut self, scale: f64) {
        if scale.is_finite() {
            self.transition = ScaleTransition::settled(scale.max(0.0));
        }
    }

    /// Smoothly ramp the scale toward `target` over `duration` using `easing`,
    /// starting from the current value (continuous, no snap). Clamped to
    /// `>= 0`; non-finite targets are ignored.
    #[inline]
    pub fn ease_to(&mut self, target: f64, duration: Duration, easing: Easing) {
        if target.is_finite() {
            self.transition
                .retarget_with(target.max(0.0), duration, easing);
        }
    }

    /// Progress any in-flight eased transition by `delta` (typically the
    /// parent/real delta). Call once per update before reading the scale.
    #[inline]
    pub fn advance(&mut self, delta: Duration) {
        self.transition.advance(delta);
    }

    /// Whether an eased transition is still in progress.
    #[inline]
    #[must_use]
    pub fn is_transitioning(&self) -> bool {
        !self.transition.is_complete()
    }

    /// Pause the scope (effective scale becomes `0.0`; the requested scale is
    /// preserved for unpause).
    #[inline]
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Unpause the scope, restoring the requested scale.
    #[inline]
    pub fn unpause(&mut self) {
        self.paused = false;
    }

    /// Whether the scope is paused.
    #[inline]
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Apply this scope's effective scale to a parent delta.
    ///
    /// Compose scopes by chaining: `child.scale_delta(parent.scale_delta(dt))`,
    /// or use [`compose`](Self::compose) to fold a parent scale factor in.
    #[inline]
    #[must_use]
    pub fn scale_delta(&self, parent_delta: Duration) -> Duration {
        let s = self.effective_scale();
        if s == 1.0 {
            parent_delta
        } else if s == 0.0 {
            Duration::ZERO
        } else {
            parent_delta.mul_f64(s)
        }
    }

    /// The composite scale `parent_scale * self.effective_scale()`, for layering
    /// this scope on top of a parent (world or global) scale factor.
    #[inline]
    #[must_use]
    pub fn compose(&self, parent_scale: f64) -> f64 {
        parent_scale * self.effective_scale()
    }
}

impl Default for TimeScaleDomain {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

/// Fold a chain of domains (outermost first) into a single composite effective
/// scale. Equivalent to multiplying every domain's `effective_scale`.
///
/// ```
/// use prism_time::{TimeScaleDomain, composite_scale};
///
/// let world = TimeScaleDomain::new(0.5); // background world at half speed
/// let local = TimeScaleDomain::new(0.5); // slow-mo field within it
/// let s = composite_scale([&world, &local]);
/// assert!((s - 0.25).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn composite_scale<'a, I>(domains: I) -> f64
where
    I: IntoIterator<Item = &'a TimeScaleDomain>,
{
    domains
        .into_iter()
        .fold(1.0, |acc, d| acc * d.effective_scale())
}

/// Sanitize a scale: non-finite falls back to `fallback`, negatives clamp to 0.
#[inline]
fn sanitize_scale(scale: f64, fallback: f64) -> f64 {
    if scale.is_finite() {
        scale.max(0.0)
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::{composite_scale, TimeScaleDomain};
    use crate::{Duration, Easing};

    #[test]
    fn identity_passes_delta_through() {
        let d = TimeScaleDomain::identity();
        assert_eq!(d.effective_scale(), 1.0);
        assert_eq!(
            d.scale_delta(Duration::from_millis(16)),
            Duration::from_millis(16)
        );
    }

    #[test]
    fn half_speed_halves_delta() {
        let d = TimeScaleDomain::new(0.5);
        assert_eq!(
            d.scale_delta(Duration::from_millis(20)),
            Duration::from_millis(10)
        );
    }

    #[test]
    fn pause_freezes_scope() {
        let mut d = TimeScaleDomain::new(2.0);
        d.pause();
        assert_eq!(d.effective_scale(), 0.0);
        assert_eq!(d.scale_delta(Duration::from_secs(1)), Duration::ZERO);
        d.unpause();
        assert_eq!(d.effective_scale(), 2.0);
    }

    #[test]
    fn negative_and_nonfinite_scales_are_sanitized() {
        let d = TimeScaleDomain::new(-3.0);
        assert_eq!(d.effective_scale(), 0.0);
        let mut d2 = TimeScaleDomain::new(1.0);
        d2.set_scale(f64::NAN);
        assert_eq!(d2.effective_scale(), 1.0); // ignored
        d2.set_scale(-1.0);
        assert_eq!(d2.effective_scale(), 0.0);
    }

    #[test]
    fn nested_domains_compose_multiplicatively() {
        let world = TimeScaleDomain::new(0.5);
        let local = TimeScaleDomain::new(0.25);
        // Via explicit chaining of deltas.
        let dt = Duration::from_millis(80);
        let chained = local.scale_delta(world.scale_delta(dt));
        assert_eq!(chained, Duration::from_millis(10)); // 80 * 0.5 * 0.25
                                                        // Via composite_scale helper.
        let s = composite_scale([&world, &local]);
        assert!((s - 0.125).abs() < 1e-12);
        // Via compose fold.
        assert!((local.compose(world.effective_scale()) - 0.125).abs() < 1e-12);
    }

    #[test]
    fn paused_parent_zeroes_whole_chain() {
        let mut world = TimeScaleDomain::new(1.0);
        world.pause();
        let local = TimeScaleDomain::new(2.0);
        assert_eq!(composite_scale([&world, &local]), 0.0);
    }

    #[test]
    fn eased_transition_is_smooth_and_settles() {
        let mut d = TimeScaleDomain::new(1.0);
        d.ease_to(0.2, Duration::from_secs(1), Easing::SmoothStep);
        assert!(d.is_transitioning());
        assert!((d.scale() - 1.0).abs() < 1e-12);
        d.advance(Duration::from_millis(500));
        // smoothstep(0.5) = 0.5 -> 1.0 + (0.2-1.0)*0.5 = 0.6
        assert!((d.scale() - 0.6).abs() < 1e-9, "{}", d.scale());
        d.advance(Duration::from_millis(500));
        assert!(!d.is_transitioning());
        assert!((d.scale() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn default_is_identity() {
        assert_eq!(TimeScaleDomain::default(), TimeScaleDomain::identity());
    }
}
