//! By-name feature-health registry with validated updates and aggregation.
//!
//! The registry is the front door to the health subsystem: each optional
//! rendering feature registers a [`FeatureStatus`] under its stable name, and
//! callers query and drive its health through the same validated lifecycle
//! rules enforced by [`is_valid_transition`]. On top of the per-feature view it
//! answers the system-level question every frame scheduler cares about: is the
//! renderer as a whole healthy, or has something degraded?
//!
//! Features are keyed by their `&'static str` name in a [`BTreeMap`], so
//! iteration order is deterministic (lexicographic by name) regardless of
//! registration order.

use alloc::collections::BTreeMap;
use alloc::string::String;

use super::health::{is_valid_transition, TransitionError};
use super::{FeatureHealth, FeatureStatus};

/// Error returned by a registry mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistryError {
    /// No feature is registered under the given name.
    Unknown(&'static str),
    /// The requested health change is not a legal lifecycle transition.
    Transition(TransitionError),
}

impl core::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RegistryError::Unknown(name) => write!(f, "unknown feature {name:?}"),
            RegistryError::Transition(err) => write!(f, "{err}"),
        }
    }
}

impl From<TransitionError> for RegistryError {
    fn from(err: TransitionError) -> Self {
        RegistryError::Transition(err)
    }
}

/// A by-name collection of [`FeatureStatus`] records.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FeatureRegistry {
    features: BTreeMap<&'static str, FeatureStatus>,
}

impl FeatureRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            features: BTreeMap::new(),
        }
    }

    /// Number of registered features.
    #[must_use]
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Returns `true` when no features are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Returns `true` when a feature is registered under `name`.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.features.contains_key(name)
    }

    /// Registers (or replaces) `status`, returning any prior record.
    pub fn register(&mut self, status: FeatureStatus) -> Option<FeatureStatus> {
        self.features.insert(status.name, status)
    }

    /// Registers a feature under `name` in the initial
    /// [`FeatureHealth::Unavailable`] state, returning any prior record.
    pub fn register_named(&mut self, name: &'static str) -> Option<FeatureStatus> {
        self.register(FeatureStatus::new(name))
    }

    /// Removes and returns the feature registered under `name`, if any.
    pub fn remove(&mut self, name: &str) -> Option<FeatureStatus> {
        self.features.remove(name)
    }

    /// Borrows the [`FeatureStatus`] registered under `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&FeatureStatus> {
        self.features.get(name)
    }

    /// Returns the health of the feature registered under `name`.
    #[must_use]
    pub fn health_of(&self, name: &str) -> Option<FeatureHealth> {
        self.features.get(name).map(|s| s.health)
    }

    /// Transitions the named feature to `health`, validating the lifecycle rule
    /// and replacing its reason with `reason`.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Unknown`] when `name` is not registered, or
    /// [`RegistryError::Transition`] when the current-to-`health` change is not
    /// permitted by [`is_valid_transition`]. On error the stored status is left
    /// unchanged.
    pub fn transition(
        &mut self,
        name: &'static str,
        health: FeatureHealth,
        reason: Option<String>,
    ) -> Result<FeatureHealth, RegistryError> {
        let status = self
            .features
            .get_mut(name)
            .ok_or(RegistryError::Unknown(name))?;
        if is_valid_transition(status.health, health) {
            status.health = health;
            status.reason = reason;
            Ok(health)
        } else {
            Err(RegistryError::Transition(TransitionError {
                from: status.health,
                to: health,
            }))
        }
    }

    /// Iterates registered statuses in deterministic (name-sorted) order.
    pub fn iter(&self) -> impl Iterator<Item = &FeatureStatus> + '_ {
        self.features.values()
    }

    /// Iterates registered feature names in deterministic (sorted) order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.features.keys().copied()
    }

    /// Aggregates an overall system health: the most severe registered feature
    /// health per [`FeatureHealth::severity`], or `None` when empty.
    ///
    /// Ties resolve to the first (name-sorted) feature at the top severity, but
    /// since severity fully determines the returned value the tie-break does
    /// not affect the result.
    #[must_use]
    pub fn overall_health(&self) -> Option<FeatureHealth> {
        let mut worst: Option<FeatureHealth> = None;
        for status in self.features.values() {
            match worst {
                Some(current) if status.health.severity() <= current.severity() => {}
                _ => worst = Some(status.health),
            }
        }
        worst
    }

    /// Returns `true` when the system is degraded: any feature is
    /// [`FeatureHealth::Degraded`] or [`FeatureHealth::Quarantined`].
    #[must_use]
    pub fn is_system_degraded(&self) -> bool {
        self.overall_health()
            .is_some_and(|h| h.severity() >= FeatureHealth::Degraded.severity())
    }

    /// Returns `true` when every registered feature is
    /// [`FeatureHealth::Active`]. An empty registry is not considered healthy.
    #[must_use]
    pub fn all_operational(&self) -> bool {
        !self.features.is_empty() && self.features.values().all(|s| s.health.is_operational())
    }

    /// Counts registered features whose health equals `health`.
    #[must_use]
    pub fn count_with(&self, health: FeatureHealth) -> usize {
        self.features
            .values()
            .filter(|s| s.health == health)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::FeatureHealth;

    fn registry_with(entries: &[(&'static str, FeatureHealth)]) -> FeatureRegistry {
        let mut r = FeatureRegistry::new();
        for &(name, health) in entries {
            r.register(FeatureStatus::new(name).with_health(health));
        }
        r
    }

    #[test]
    fn query_missing_returns_none() {
        let r = FeatureRegistry::new();
        assert!(r.is_empty());
        assert!(r.get("nope").is_none());
        assert!(r.health_of("nope").is_none());
        assert!(!r.contains("nope"));
        assert!(r.overall_health().is_none());
        assert!(!r.is_system_degraded());
        assert!(!r.all_operational());
    }

    #[test]
    fn register_query_and_replace() {
        let mut r = FeatureRegistry::new();
        assert!(r.register_named("hair").is_none());
        assert_eq!(r.len(), 1);
        assert_eq!(r.health_of("hair"), Some(FeatureHealth::Unavailable));

        // Re-registering returns the prior record.
        let prior = r.register(FeatureStatus::new("hair").with_health(FeatureHealth::Active));
        assert_eq!(prior.map(|s| s.health), Some(FeatureHealth::Unavailable));
        assert_eq!(r.health_of("hair"), Some(FeatureHealth::Active));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn remove_feature() {
        let mut r = registry_with(&[("a", FeatureHealth::Active)]);
        assert_eq!(r.remove("a").map(|s| s.health), Some(FeatureHealth::Active));
        assert!(r.remove("a").is_none());
        assert!(r.is_empty());
    }

    #[test]
    fn transition_validates_lifecycle() {
        let mut r = registry_with(&[("shadow", FeatureHealth::Unavailable)]);
        assert_eq!(
            r.transition("shadow", FeatureHealth::Initializing, None),
            Ok(FeatureHealth::Initializing)
        );
        assert_eq!(
            r.transition(
                "shadow",
                FeatureHealth::Warming,
                Some(String::from("compiling"))
            ),
            Ok(FeatureHealth::Warming)
        );
        assert_eq!(
            r.get("shadow").and_then(|s| s.reason.as_deref()),
            Some("compiling")
        );
    }

    #[test]
    fn illegal_transition_is_rejected_and_unchanged() {
        let mut r = registry_with(&[("shadow", FeatureHealth::Unavailable)]);
        let err = r
            .transition("shadow", FeatureHealth::Active, None)
            .unwrap_err();
        assert_eq!(
            err,
            RegistryError::Transition(TransitionError {
                from: FeatureHealth::Unavailable,
                to: FeatureHealth::Active,
            })
        );
        // Unchanged.
        assert_eq!(r.health_of("shadow"), Some(FeatureHealth::Unavailable));
    }

    #[test]
    fn transition_unknown_feature() {
        let mut r = FeatureRegistry::new();
        assert_eq!(
            r.transition("ghost", FeatureHealth::Initializing, None),
            Err(RegistryError::Unknown("ghost"))
        );
    }

    #[test]
    fn overall_health_takes_worst() {
        let r = registry_with(&[
            ("a", FeatureHealth::Active),
            ("b", FeatureHealth::Warming),
            ("c", FeatureHealth::Active),
        ]);
        // Warming is worse than Active.
        assert_eq!(r.overall_health(), Some(FeatureHealth::Warming));
        assert!(!r.is_system_degraded());
        assert!(!r.all_operational());
    }

    #[test]
    fn quarantine_dominates_system_health() {
        let r = registry_with(&[
            ("a", FeatureHealth::Active),
            ("b", FeatureHealth::Degraded),
            ("c", FeatureHealth::Quarantined),
        ]);
        assert_eq!(r.overall_health(), Some(FeatureHealth::Quarantined));
        assert!(r.is_system_degraded());
    }

    #[test]
    fn degraded_marks_system_degraded() {
        let r = registry_with(&[("a", FeatureHealth::Active), ("b", FeatureHealth::Degraded)]);
        assert_eq!(r.overall_health(), Some(FeatureHealth::Degraded));
        assert!(r.is_system_degraded());
    }

    #[test]
    fn all_operational_when_every_feature_active() {
        let r = registry_with(&[("a", FeatureHealth::Active), ("b", FeatureHealth::Active)]);
        assert!(r.all_operational());
        assert_eq!(r.overall_health(), Some(FeatureHealth::Active));
        assert!(!r.is_system_degraded());
    }

    #[test]
    fn count_with_and_deterministic_iteration() {
        let r = registry_with(&[
            ("zeta", FeatureHealth::Active),
            ("alpha", FeatureHealth::Degraded),
            ("mu", FeatureHealth::Active),
        ]);
        assert_eq!(r.count_with(FeatureHealth::Active), 2);
        assert_eq!(r.count_with(FeatureHealth::Degraded), 1);
        assert_eq!(r.count_with(FeatureHealth::Quarantined), 0);
        // BTreeMap iterates by sorted name regardless of insertion order.
        assert_eq!(r.names().collect::<Vec<_>>(), ["alpha", "mu", "zeta"]);
    }

    #[test]
    fn full_recovery_flow_through_registry() {
        let mut r = registry_with(&[("upscale", FeatureHealth::Active)]);
        // Fault -> quarantine.
        assert_eq!(
            r.transition("upscale", FeatureHealth::Quarantined, Some("oom".into())),
            Ok(FeatureHealth::Quarantined)
        );
        assert!(r.is_system_degraded());
        // Cannot jump back to Active.
        assert!(r
            .transition("upscale", FeatureHealth::Active, None)
            .is_err());
        // Controlled recovery.
        assert_eq!(
            r.transition("upscale", FeatureHealth::Initializing, None),
            Ok(FeatureHealth::Initializing)
        );
        assert_eq!(
            r.transition("upscale", FeatureHealth::Warming, None),
            Ok(FeatureHealth::Warming)
        );
        assert_eq!(
            r.transition("upscale", FeatureHealth::Active, None),
            Ok(FeatureHealth::Active)
        );
        assert!(r.all_operational());
    }

    #[test]
    fn registry_error_display() {
        let s = alloc::format!("{}", RegistryError::Unknown("x"));
        assert!(s.contains('x'));
    }
}
