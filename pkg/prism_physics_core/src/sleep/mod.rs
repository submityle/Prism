//! Rigid-body sleeping (deactivation) state and policy.
//!
//! This module owns the [`SleepConfig`] tuning parameters and the pure policy
//! functions the solver uses to advance per-body sleep timers and decide when a
//! body may sleep. The per-body sleep flags and timers themselves live as
//! Structure-of-Arrays columns on
//! [`BodyStorage`](crate::state::storage::BodyStorage); this module only holds
//! the configuration and the branch-free timing arithmetic so it can be unit
//! tested in isolation.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

pub mod config;

pub use config::SleepConfig;

/// Advances an idle timer by `dt` seconds for a single step.
///
/// When `below_thresholds` is `true` the body has stayed idle this step, so the
/// accumulated idle time grows by `dt`. Otherwise the body moved above a
/// threshold and the timer resets to zero. `dt` is expected to be
/// non-negative; a negative `dt` is clamped to zero so the timer never runs
/// backwards.
#[must_use]
pub fn advance_sleep_timer(timer: f32, dt: f32, below_thresholds: bool) -> f32 {
    if below_thresholds {
        timer + dt.max(0.0)
    } else {
        0.0
    }
}

/// Returns `true` when an idle timer has reached the configured dwell time and
/// sleeping is enabled.
///
/// This is a pure predicate; it does not mutate any state. Callers combine it
/// per island so an island sleeps only when *every* dynamic member is ready.
#[must_use]
pub fn ready_to_sleep(timer: f32, config: &SleepConfig) -> bool {
    config.enabled && timer >= config.time_to_sleep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timer_accumulates_while_idle() {
        let t = advance_sleep_timer(0.0, 0.1, true);
        assert!((t - 0.1).abs() < 1e-6);
        let t = advance_sleep_timer(t, 0.1, true);
        assert!((t - 0.2).abs() < 1e-6);
    }

    #[test]
    fn timer_resets_when_active() {
        assert_eq!(advance_sleep_timer(0.4, 0.1, false), 0.0);
    }

    #[test]
    fn negative_dt_does_not_rewind() {
        assert_eq!(advance_sleep_timer(0.3, -1.0, true), 0.3);
    }

    #[test]
    fn ready_after_dwell() {
        let c = SleepConfig::default();
        assert!(!ready_to_sleep(0.4, &c));
        assert!(ready_to_sleep(0.5, &c));
        assert!(ready_to_sleep(0.6, &c));
    }

    #[test]
    fn disabled_never_ready() {
        let c = SleepConfig {
            enabled: false,
            ..SleepConfig::default()
        };
        assert!(!ready_to_sleep(100.0, &c));
    }
}
