//! Frame-time and memory-budget adaptive quality control.
//!
//! A shipping AAA renderer cannot render every knob at maximum on every
//! machine and hold frame rate, so it *adapts*: when frames run long it trades
//! image quality for speed, and when there is headroom it spends it back. This
//! module owns the `CPU`-verifiable contract for that feedback loop.
//!
//! The contract has three parts:
//!
//! 1. **Target** ([`QualityTarget`]) — the frame-time and memory budget the
//!    controller aims for, plus whether adaptation is enabled at all.
//! 2. **Decision** ([`QualityDecision`]) — the per-frame settlement of the
//!    tunable knobs: render scale, geometry error tolerance, shadow page
//!    budget, and global-illumination ray density.
//! 3. **Controller** ([`QualityController`]) — the trait a control policy
//!    implements. The reference policy is
//!    [`AdaptiveQualityController`](controller::AdaptiveQualityController), a
//!    deterministic proportional-integral (`PID`-family) loop; see [`controller`].
//!
//! All arithmetic is basic (`+ - * /` and `round`/`clamp`-style branches) with
//! no transcendental functions, so a controller's output is a deterministic
//! function of its input sequence. The `GPU`-side application of a decision is
//! out of scope and pending the `GPU` backend.

pub mod controller;

pub use controller::{AdaptiveQualityController, QualityBounds, SHADOW_PAGE_BYTES};

/// The budget an adaptive controller steers toward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityTarget {
    /// Target frame time in milliseconds (e.g. `16.67` for 60 Hz).
    pub frame_time_ms: f32,
    /// Resident memory budget in bytes; caps memory-bound knobs such as the
    /// shadow page budget.
    pub memory_bytes: u64,
    /// Whether adaptation is enabled. When `false`, the controller runs at full
    /// quality and ignores the measured frame time.
    pub adaptive: bool,
}

/// The per-frame settlement of the adaptively controlled quality knobs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityDecision {
    /// Render scale in `(0, 1]` for the temporal upsampler.
    pub render_scale: f32,
    /// Tolerated screen-space geometry error in pixels (drives `LOD` / virtual
    /// geometry selection). Smaller is higher quality.
    pub geometry_error_pixels: f32,
    /// Number of shadow pages the virtual-shadow allocator may keep resident.
    pub shadow_page_budget: u32,
    /// Global-illumination ray density scale in `(0, 1]`.
    pub gi_ray_scale: f32,
}

/// A policy that maps a target and a measured frame time to a quality decision.
///
/// Implementors hold whatever state their control law needs (a feedback
/// integrator, a history window, ...) and are updated once per frame.
pub trait QualityController {
    /// Advances the controller by one frame and returns the decision to apply.
    fn update(&mut self, target: QualityTarget, measured_frame_ms: f32) -> QualityDecision;
}

/// Clamps `x` to `[0, 1]`, resolving `NaN` to `0` deterministically.
///
/// The `NaN` check leads so a `NaN` input lands on `0.0` instead of
/// propagating, and so the branch form does not trip the `manual_clamp` lint.
#[must_use]
pub(crate) fn clamp_unit(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// Linear interpolation `a + (b - a) * t`, with `t` expected in `[0, 1]`.
#[must_use]
pub(crate) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Linear interpolation between two unsigned budgets, rounded to the nearest
/// integer and floored at zero.
///
/// Works whether `a <= b` or `a >= b`, so callers may pass endpoints in either
/// order; `round` gives the symmetric nearest-integer result.
#[must_use]
pub(crate) fn lerp_u32(a: u32, b: u32, t: f32) -> u32 {
    let value = lerp(a as f32, b as f32, t).round();
    if value.is_nan() || value < 0.0 {
        0
    } else {
        value as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    #[test]
    fn clamp_unit_resolves_nan_and_range() {
        assert!(approx(clamp_unit(f32::NAN), 0.0));
        assert!(approx(clamp_unit(-2.0), 0.0));
        assert!(approx(clamp_unit(0.4), 0.4));
        assert!(approx(clamp_unit(3.0), 1.0));
    }

    #[test]
    fn lerp_hits_endpoints_and_midpoint() {
        assert!(approx(lerp(2.0, 6.0, 0.0), 2.0));
        assert!(approx(lerp(2.0, 6.0, 1.0), 6.0));
        assert!(approx(lerp(2.0, 6.0, 0.5), 4.0));
    }

    #[test]
    fn lerp_u32_rounds_and_handles_both_orders() {
        assert_eq!(lerp_u32(0, 100, 0.5), 50);
        assert_eq!(lerp_u32(100, 0, 0.5), 50);
        assert_eq!(lerp_u32(0, 10, 0.25), 3); // 2.5 rounds to 3
        assert_eq!(lerp_u32(1024, 8192, 1.0), 8192);
    }
}
