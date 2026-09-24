//! Configuration for continuous collision detection (CCD).
//!
//! Discrete collision detection tests each body only at its predicted
//! end-of-sub-step pose. A body that moves further than its own size in one
//! sub-step can therefore pass completely through thin geometry (a bullet
//! through a wall) without ever generating a contact. CCD guards against this
//! by sweeping the body's core sphere along its sub-step motion and clamping
//! the body to the first surface it would cross.
//!
//! CCD is opt-in per body (see [`crate::state::body::BodyDesc::with_ccd`])
//! because the extra swept query is unnecessary for the slow-moving majority of
//! bodies. This configuration gates the whole feature and tunes when the sweep
//! actually runs.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! speculative "sweep the core sphere and clamp to the time of impact" policy
//! is a standard, publicly documented continuous-collision technique.

/// Global thresholds that govern continuous collision detection.
///
/// A CCD-flagged dynamic body is swept only when it is a genuine fast mover:
/// its sub-step displacement must exceed [`min_motion_ratio`](CcdConfig::min_motion_ratio)
/// times its own core radius. Slow motion falls through to ordinary discrete
/// detection with no added cost.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CcdConfig {
    /// Whether continuous collision detection runs at all. When `false` the
    /// sweep phase is skipped entirely and every body relies on discrete
    /// detection.
    pub enabled: bool,
    /// Minimum ratio of sub-step displacement to core radius before a body is
    /// swept. A value of `0.5` sweeps any body that moves more than half its
    /// own core radius in a single sub-step.
    pub min_motion_ratio: f32,
    /// Safety margin (metres) subtracted from the time of impact so a clamped
    /// body stops just short of the surface rather than exactly touching it.
    pub skin: f32,
}

impl CcdConfig {
    /// Default minimum motion ratio: sweep bodies that move more than half a
    /// core radius per sub-step.
    pub const DEFAULT_MIN_MOTION_RATIO: f32 = 0.5;
    /// Default skin margin in metres.
    pub const DEFAULT_SKIN: f32 = 0.0;
}

impl Default for CcdConfig {
    fn default() -> Self {
        CcdConfig {
            enabled: true,
            min_motion_ratio: Self::DEFAULT_MIN_MOTION_RATIO,
            skin: Self::DEFAULT_SKIN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_spec() {
        let c = CcdConfig::default();
        assert!(c.enabled);
        assert_eq!(c.min_motion_ratio, 0.5);
        assert_eq!(c.skin, 0.0);
    }

    #[test]
    fn config_is_copy_and_eq() {
        let a = CcdConfig::default();
        let b = a;
        assert_eq!(a, b);
    }
}
