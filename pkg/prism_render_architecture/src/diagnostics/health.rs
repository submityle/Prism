//! Validated feature-health lifecycle state machine.
//!
//! Every optional rendering feature moves through a small, well-defined
//! lifecycle. Enforcing the legal transitions in one place keeps the rest of
//! the renderer honest: a subsystem cannot claim to be `Active` without having
//! warmed, and a tripped subsystem cannot quietly resume without going back
//! through initialization.
//!
//! # Legal transitions
//!
//! The happy path is a straight climb:
//!
//! ```text
//! Unavailable -> Initializing -> Warming -> Active
//! ```
//!
//! Once live, a feature may oscillate between full and reduced capability:
//!
//! ```text
//! Active  -> Degraded   (self-reported reduced capability)
//! Degraded -> Active     (recovered)
//! ```
//!
//! Any state can be tripped offline by a fault:
//!
//! ```text
//! *any non-Quarantined state* -> Quarantined
//! ```
//!
//! and a quarantined feature only comes back through a controlled restart:
//!
//! ```text
//! Quarantined -> Initializing
//! ```
//!
//! Every other transition (including no-op self transitions and skipping
//! lifecycle stages) is rejected with a [`TransitionError`]. All checks are
//! pure integer/`enum` matches, so the machine is fully deterministic and
//! `CPU`-verifiable.

use super::FeatureHealth;

/// Error describing a rejected health transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransitionError {
    /// State the machine was in.
    pub from: FeatureHealth,
    /// State that was requested and rejected.
    pub to: FeatureHealth,
}

impl core::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "illegal feature-health transition {:?} -> {:?}",
            self.from, self.to
        )
    }
}

/// Returns `true` when `from -> to` is a legal lifecycle transition.
///
/// Quarantine is reachable from any *other* state (a fault can trip a feature
/// at any point), but re-quarantining an already `Quarantined` feature is not a
/// transition and is rejected. No-op self transitions are likewise rejected so
/// callers cannot mask a missing state change.
#[must_use]
pub const fn is_valid_transition(from: FeatureHealth, to: FeatureHealth) -> bool {
    use FeatureHealth::{Active, Degraded, Initializing, Quarantined, Unavailable, Warming};

    // Fault trip: any live/spin-up state can be quarantined, but not itself.
    if matches!(to, Quarantined) {
        return !matches!(from, Quarantined);
    }

    matches!(
        (from, to),
        (Unavailable, Initializing)
            | (Initializing, Warming)
            | (Warming, Active)
            | (Active, Degraded)
            | (Degraded, Active)
            | (Quarantined, Initializing)
    )
}

/// A feature-health lifecycle tracker that only permits legal transitions.
///
/// Construct with [`HealthMachine::new`] (starts [`FeatureHealth::Unavailable`])
/// and drive it with [`HealthMachine::try_transition`] or the named convenience
/// methods. The machine records how many transitions and quarantine trips it
/// has applied, which callers can surface for observability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HealthMachine {
    health: FeatureHealth,
    transitions: u32,
    quarantines: u32,
}

impl HealthMachine {
    /// Creates a machine in the initial [`FeatureHealth::Unavailable`] state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            health: FeatureHealth::Unavailable,
            transitions: 0,
            quarantines: 0,
        }
    }

    /// Creates a machine already in `health` (for restoring persisted state).
    ///
    /// The transition and quarantine counters start at zero; only transitions
    /// applied through this machine are counted.
    #[must_use]
    pub const fn with_health(health: FeatureHealth) -> Self {
        Self {
            health,
            transitions: 0,
            quarantines: 0,
        }
    }

    /// Current health state.
    #[must_use]
    pub const fn health(self) -> FeatureHealth {
        self.health
    }

    /// Number of successful transitions applied so far.
    #[must_use]
    pub const fn transition_count(self) -> u32 {
        self.transitions
    }

    /// Number of times the feature has been quarantined.
    #[must_use]
    pub const fn quarantine_count(self) -> u32 {
        self.quarantines
    }

    /// Returns `true` when the feature is fully operational.
    #[must_use]
    pub const fn is_operational(self) -> bool {
        self.health.is_operational()
    }

    /// Returns `true` when the feature is currently quarantined.
    #[must_use]
    pub const fn is_quarantined(self) -> bool {
        matches!(self.health, FeatureHealth::Quarantined)
    }

    /// Attempts to transition to `to`.
    ///
    /// On success the state is updated, the transition counter is incremented
    /// (and the quarantine counter too when `to` is
    /// [`FeatureHealth::Quarantined`]), and the new state is returned. On an
    /// illegal transition the state is left untouched and a [`TransitionError`]
    /// is returned.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] when `from -> to` is not permitted by
    /// [`is_valid_transition`].
    pub fn try_transition(&mut self, to: FeatureHealth) -> Result<FeatureHealth, TransitionError> {
        if is_valid_transition(self.health, to) {
            self.health = to;
            self.transitions = self.transitions.saturating_add(1);
            if matches!(to, FeatureHealth::Quarantined) {
                self.quarantines = self.quarantines.saturating_add(1);
            }
            Ok(self.health)
        } else {
            Err(TransitionError {
                from: self.health,
                to,
            })
        }
    }

    /// `Unavailable -> Initializing`: begin spinning the feature up.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] unless the current state is
    /// [`FeatureHealth::Unavailable`].
    pub fn initialize(&mut self) -> Result<FeatureHealth, TransitionError> {
        self.try_transition(FeatureHealth::Initializing)
    }

    /// `Initializing -> Warming`: the feature is live but still warming.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] unless the current state is
    /// [`FeatureHealth::Initializing`].
    pub fn warm(&mut self) -> Result<FeatureHealth, TransitionError> {
        self.try_transition(FeatureHealth::Warming)
    }

    /// `Warming -> Active`: the feature is fully operational.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] unless the current state is
    /// [`FeatureHealth::Warming`].
    pub fn activate(&mut self) -> Result<FeatureHealth, TransitionError> {
        self.try_transition(FeatureHealth::Active)
    }

    /// `Active -> Degraded`: report reduced capability.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] unless the current state is
    /// [`FeatureHealth::Active`].
    pub fn degrade(&mut self) -> Result<FeatureHealth, TransitionError> {
        self.try_transition(FeatureHealth::Degraded)
    }

    /// `Degraded -> Active`: recover from reduced capability.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] unless the current state is
    /// [`FeatureHealth::Degraded`].
    pub fn restore(&mut self) -> Result<FeatureHealth, TransitionError> {
        self.try_transition(FeatureHealth::Active)
    }

    /// `*any* -> Quarantined`: trip the feature offline after a fault.
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] when the feature is already quarantined.
    pub fn quarantine(&mut self) -> Result<FeatureHealth, TransitionError> {
        self.try_transition(FeatureHealth::Quarantined)
    }

    /// `Quarantined -> Initializing`: begin controlled recovery.
    ///
    /// A quarantined feature must re-initialize (and warm) before it can serve
    /// again; there is no direct path back to [`FeatureHealth::Active`].
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] unless the current state is
    /// [`FeatureHealth::Quarantined`].
    pub fn recover(&mut self) -> Result<FeatureHealth, TransitionError> {
        self.try_transition(FeatureHealth::Initializing)
    }
}

impl Default for HealthMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::FeatureHealth;

    const ALL: [FeatureHealth; 6] = [
        FeatureHealth::Unavailable,
        FeatureHealth::Initializing,
        FeatureHealth::Warming,
        FeatureHealth::Active,
        FeatureHealth::Degraded,
        FeatureHealth::Quarantined,
    ];

    #[test]
    fn happy_path_climbs_to_active() {
        let mut m = HealthMachine::new();
        assert_eq!(m.health(), FeatureHealth::Unavailable);
        assert_eq!(m.initialize(), Ok(FeatureHealth::Initializing));
        assert_eq!(m.warm(), Ok(FeatureHealth::Warming));
        assert_eq!(m.activate(), Ok(FeatureHealth::Active));
        assert!(m.is_operational());
        assert_eq!(m.transition_count(), 3);
        assert_eq!(m.quarantine_count(), 0);
    }

    #[test]
    fn active_degraded_oscillation() {
        let mut m = HealthMachine::with_health(FeatureHealth::Active);
        assert_eq!(m.degrade(), Ok(FeatureHealth::Degraded));
        assert_eq!(m.restore(), Ok(FeatureHealth::Active));
        assert_eq!(m.degrade(), Ok(FeatureHealth::Degraded));
        assert_eq!(m.transition_count(), 3);
    }

    #[test]
    fn quarantine_from_any_state_then_recover() {
        for &start in &ALL {
            let mut m = HealthMachine::with_health(start);
            if matches!(start, FeatureHealth::Quarantined) {
                // Already quarantined: re-quarantine is rejected.
                assert_eq!(
                    m.quarantine(),
                    Err(TransitionError {
                        from: FeatureHealth::Quarantined,
                        to: FeatureHealth::Quarantined,
                    })
                );
                continue;
            }
            assert_eq!(m.quarantine(), Ok(FeatureHealth::Quarantined));
            assert_eq!(m.quarantine_count(), 1);
            // Recovery must go through Initializing, not straight to Active.
            assert!(m.activate().is_err());
            assert_eq!(m.recover(), Ok(FeatureHealth::Initializing));
        }
    }

    #[test]
    fn recovery_requires_full_reinit_before_active() {
        let mut m = HealthMachine::with_health(FeatureHealth::Quarantined);
        assert_eq!(m.recover(), Ok(FeatureHealth::Initializing));
        assert!(m.activate().is_err()); // cannot skip Warming
        assert_eq!(m.warm(), Ok(FeatureHealth::Warming));
        assert_eq!(m.activate(), Ok(FeatureHealth::Active));
    }

    #[test]
    fn illegal_transitions_are_rejected_and_state_preserved() {
        let mut m = HealthMachine::new();
        // Unavailable cannot jump to Active.
        assert_eq!(
            m.try_transition(FeatureHealth::Active),
            Err(TransitionError {
                from: FeatureHealth::Unavailable,
                to: FeatureHealth::Active,
            })
        );
        // State untouched, no transition counted.
        assert_eq!(m.health(), FeatureHealth::Unavailable);
        assert_eq!(m.transition_count(), 0);
    }

    #[test]
    fn self_transitions_are_rejected() {
        for &s in &ALL {
            let mut m = HealthMachine::with_health(s);
            assert!(m.try_transition(s).is_err());
        }
    }

    #[test]
    fn validity_table_matches_rules() {
        use FeatureHealth::{Active, Degraded, Initializing, Quarantined, Unavailable, Warming};
        // Exhaustive expectation over all 36 ordered pairs.
        let legal = |from: FeatureHealth, to: FeatureHealth| {
            matches!(
                (from, to),
                (Unavailable, Initializing)
                    | (Initializing, Warming)
                    | (Warming, Active)
                    | (Active, Degraded)
                    | (Degraded, Active)
                    | (Quarantined, Initializing)
                    | (Unavailable, Quarantined)
                    | (Initializing, Quarantined)
                    | (Warming, Quarantined)
                    | (Active, Quarantined)
                    | (Degraded, Quarantined)
            )
        };
        for &from in &ALL {
            for &to in &ALL {
                assert_eq!(
                    is_valid_transition(from, to),
                    legal(from, to),
                    "mismatch for {from:?} -> {to:?}"
                );
            }
        }
    }

    #[test]
    fn display_names_states() {
        let e = TransitionError {
            from: FeatureHealth::Unavailable,
            to: FeatureHealth::Active,
        };
        let s = alloc::format!("{e}");
        assert!(s.contains("Unavailable"));
        assert!(s.contains("Active"));
    }

    #[test]
    fn default_is_new() {
        assert_eq!(HealthMachine::default(), HealthMachine::new());
    }
}
