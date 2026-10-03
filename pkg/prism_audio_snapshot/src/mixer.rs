//! The stateful orchestration core that runs snapshot transitions.
//!
//! A [`SnapshotMixer`] owns a [`SnapshotRegistry`], a [`SnapshotConfig`] of
//! defaults, the live resolved parameter map, and at most one in-flight
//! [`Transition`]. Callers ask it to move toward a snapshot (or a weighted
//! blend of snapshots), advance it by a timestep, and read the current blended
//! parameter values. It can also be driven directly from the global state
//! model via a set of [`StateSnapshotBindings`].
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Ties together [`crate::registry`], [`crate::config`], [`crate::stack`],
//! [`crate::transition`], and [`crate::state_binding`]. Reads
//! `prism_audio_content::state::StateManager` through the bindings and shapes
//! transitions with `prism_audio_content::curve::Interpolation`.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_audio_content::curve::Interpolation;
use prism_audio_content::state::StateManager;
use prism_audio_core::math::Sample;

use crate::config::SnapshotConfig;
use crate::parameter::{ParameterId, ParameterKind};
use crate::registry::SnapshotRegistry;
use crate::resolved::ResolvedParameters;
use crate::snapshot::SnapshotId;
use crate::stack;
use crate::state_binding::StateSnapshotBindings;
use crate::transition::Transition;

/// Owns the registry and runs one interpolated transition at a time.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotMixer {
    /// Snapshots this mixer can transition to.
    registry: SnapshotRegistry,
    /// Default duration and curve for transitions that omit them.
    config: SnapshotConfig,
    /// Live resolved parameter values.
    current: ResolvedParameters,
    /// The in-flight transition, if any.
    active: Option<Transition>,
}

impl SnapshotMixer {
    /// Creates a mixer with no current parameters and no active transition.
    #[must_use]
    pub fn new(registry: SnapshotRegistry, config: SnapshotConfig) -> Self {
        Self { registry, config, current: ResolvedParameters::new(), active: None }
    }

    /// Creates a mixer whose current parameters are the resolved values of
    /// `initial_snapshot`, if that snapshot is registered (otherwise empty).
    #[must_use]
    pub fn new_with(
        registry: SnapshotRegistry,
        config: SnapshotConfig,
        initial_snapshot: SnapshotId,
    ) -> Self {
        let current = registry
            .get(initial_snapshot)
            .map(ResolvedParameters::from_snapshot)
            .unwrap_or_default();
        Self { registry, config, current, active: None }
    }

    /// Starts a transition from the current values toward snapshot `id`.
    ///
    /// Returns `false` without changing anything when `id` is not registered.
    /// A `None` duration or curve falls back to the configured defaults.
    pub fn transition_to(
        &mut self,
        id: SnapshotId,
        duration_secs: Option<f32>,
        interp: Option<Interpolation>,
    ) -> bool {
        let (dest, kinds) = match self.registry.get(id) {
            Some(snapshot) => {
                let dest = ResolvedParameters::from_snapshot(snapshot);
                let kinds: BTreeMap<ParameterId, ParameterKind> =
                    snapshot.targets().map(|t| (t.id, t.kind)).collect();
                (dest, kinds)
            }
            None => return false,
        };
        self.begin(dest, kinds, duration_secs, interp);
        true
    }

    /// Starts a transition toward the weighted blend of several snapshots.
    ///
    /// Returns `false` without changing anything when the blend resolves to no
    /// parameters (empty input, zero weights, or only unknown snapshots).
    pub fn transition_to_blend(
        &mut self,
        weighted: &[(SnapshotId, Sample)],
        duration_secs: Option<f32>,
        interp: Option<Interpolation>,
    ) -> bool {
        let dest = stack::resolve_blend(&self.registry, weighted);
        if dest.is_empty() {
            return false;
        }
        let ids: Vec<SnapshotId> = weighted.iter().map(|&(id, _)| id).collect();
        let kinds = stack::blend_kinds(&self.registry, &ids);
        self.begin(dest, kinds, duration_secs, interp);
        true
    }

    /// Advances the active transition by `dt_secs`.
    ///
    /// Updates the current values to the live sample and, once complete, lands
    /// on the destination and clears the active transition. Does nothing when
    /// no transition is active.
    pub fn advance(&mut self, dt_secs: f32) {
        if let Some(transition) = self.active.as_mut() {
            let done = transition.advance(dt_secs);
            self.current = transition.sample();
            if done {
                self.active = None;
            }
        }
    }

    /// Returns the live resolved parameter map.
    #[must_use]
    pub fn current(&self) -> &ResolvedParameters {
        &self.current
    }

    /// Returns the current value of parameter `id`, if present.
    #[must_use]
    pub fn parameter(&self, id: ParameterId) -> Option<Sample> {
        self.current.get(id)
    }

    /// Returns `true` while a transition is in flight.
    #[must_use]
    pub fn is_transitioning(&self) -> bool {
        self.active.is_some()
    }

    /// Returns the registry backing this mixer.
    #[must_use]
    pub fn registry(&self) -> &SnapshotRegistry {
        &self.registry
    }

    /// Returns the configured defaults.
    #[must_use]
    pub fn config(&self) -> &SnapshotConfig {
        &self.config
    }

    /// Drives the mixer from the active game states.
    ///
    /// Resolves the active snapshots through `bindings`: an empty set leaves
    /// the mixer untouched, a single snapshot transitions directly, and
    /// several snapshots transition toward their equal-weight blend. Returns
    /// `true` when a transition was started.
    pub fn drive_from_states(
        &mut self,
        states: &StateManager,
        bindings: &StateSnapshotBindings,
        duration_secs: Option<f32>,
        interp: Option<Interpolation>,
    ) -> bool {
        let active = bindings.resolve_active(states);
        match active.as_slice() {
            [] => false,
            [only] => self.transition_to(*only, duration_secs, interp),
            many => {
                let weighted: Vec<(SnapshotId, Sample)> =
                    many.iter().map(|&id| (id, 1.0)).collect();
                self.transition_to_blend(&weighted, duration_secs, interp)
            }
        }
    }

    /// Installs a freshly built transition toward `dest`.
    fn begin(
        &mut self,
        dest: ResolvedParameters,
        kinds: BTreeMap<ParameterId, ParameterKind>,
        duration_secs: Option<f32>,
        interp: Option<Interpolation>,
    ) {
        let duration = duration_secs.unwrap_or(self.config.default_transition_secs);
        let interpolation = interp.unwrap_or(self.config.default_interpolation);
        self.active =
            Some(Transition::new(self.current.clone(), dest, kinds, duration, interpolation));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_audio_content::id::{StateGroupId, StateId};
    use prism_audio_content::state::StateGroup;

    use crate::snapshot::Snapshot;

    const EPS: Sample = 1e-6;

    fn snap(id: u32, targets: &[(u32, ParameterKind, Sample)]) -> Snapshot {
        let mut s = Snapshot::new(SnapshotId::new(id));
        for &(pid, kind, value) in targets {
            s.set(ParameterId::new(pid), kind, value);
        }
        s
    }

    fn registry() -> SnapshotRegistry {
        let mut reg = SnapshotRegistry::new();
        reg.register(snap(1, &[(1, ParameterKind::Linear, 0.0)]));
        reg.register(snap(2, &[(1, ParameterKind::Linear, 10.0)]));
        reg
    }

    #[test]
    fn transition_to_unknown_returns_false() {
        let mut m = SnapshotMixer::new(registry(), SnapshotConfig::default());
        assert!(!m.transition_to(SnapshotId::new(99), Some(1.0), Some(Interpolation::Linear)));
        assert!(!m.is_transitioning());
    }

    #[test]
    fn transition_completes_and_lands_on_dest() {
        let mut m =
            SnapshotMixer::new_with(registry(), SnapshotConfig::default(), SnapshotId::new(1));
        assert!(m.transition_to(SnapshotId::new(2), Some(2.0), Some(Interpolation::Linear)));
        assert!(m.is_transitioning());
        m.advance(1.0);
        assert!((m.parameter(ParameterId::new(1)).expect("present") - 5.0).abs() < EPS);
        assert!(m.is_transitioning());
        m.advance(1.0);
        assert!(!m.is_transitioning());
        assert!((m.parameter(ParameterId::new(1)).expect("present") - 10.0).abs() < EPS);
    }

    #[test]
    fn new_with_unknown_snapshot_starts_empty() {
        let m = SnapshotMixer::new_with(
            registry(),
            SnapshotConfig::default(),
            SnapshotId::new(99),
        );
        assert!(m.current().is_empty());
    }

    #[test]
    fn blend_transition_targets_weighted_mix() {
        let mut m =
            SnapshotMixer::new_with(registry(), SnapshotConfig::default(), SnapshotId::new(1));
        assert!(m.transition_to_blend(
            &[(SnapshotId::new(1), 1.0), (SnapshotId::new(2), 1.0)],
            Some(0.0),
            Some(Interpolation::Linear),
        ));
        // Zero duration lands immediately on the 50/50 blend of 0 and 10 = 5.
        m.advance(0.0);
        assert!((m.parameter(ParameterId::new(1)).expect("present") - 5.0).abs() < EPS);
    }

    #[test]
    fn empty_blend_returns_false() {
        let mut m = SnapshotMixer::new(registry(), SnapshotConfig::default());
        assert!(!m.transition_to_blend(&[], Some(1.0), None));
        assert!(!m.transition_to_blend(&[(SnapshotId::new(99), 1.0)], None, None));
    }

    #[test]
    fn advance_without_active_is_noop() {
        let mut m = SnapshotMixer::new(registry(), SnapshotConfig::default());
        m.advance(1.0);
        assert!(m.current().is_empty());
        assert!(!m.is_transitioning());
    }

    #[test]
    fn drive_from_states_single_snapshot() {
        let mut mgr = StateManager::new();
        mgr.register(StateGroup::new(
            StateGroupId::new(1),
            alloc::vec![StateId::new(10), StateId::new(11)],
            StateId::new(10),
        ));
        let mut bindings = StateSnapshotBindings::new();
        bindings.bind(StateGroupId::new(1), StateId::new(10), SnapshotId::new(1));
        bindings.bind(StateGroupId::new(1), StateId::new(11), SnapshotId::new(2));

        let mut m = SnapshotMixer::new(registry(), SnapshotConfig::default());
        assert!(mgr.set(StateGroupId::new(1), StateId::new(11)));
        assert!(m.drive_from_states(&mgr, &bindings, Some(0.0), Some(Interpolation::Linear)));
        m.advance(0.0);
        assert!((m.parameter(ParameterId::new(1)).expect("present") - 10.0).abs() < EPS);
    }

    #[test]
    fn drive_from_states_empty_is_noop() {
        let mgr = StateManager::new();
        let bindings = StateSnapshotBindings::new();
        let mut m = SnapshotMixer::new(registry(), SnapshotConfig::default());
        assert!(!m.drive_from_states(&mgr, &bindings, None, None));
        assert!(!m.is_transitioning());
    }

    #[test]
    fn drive_from_states_multiple_blends_equally() {
        let mut mgr = StateManager::new();
        mgr.register(StateGroup::new(
            StateGroupId::new(1),
            alloc::vec![StateId::new(10)],
            StateId::new(10),
        ));
        mgr.register(StateGroup::new(
            StateGroupId::new(2),
            alloc::vec![StateId::new(20)],
            StateId::new(20),
        ));
        let mut bindings = StateSnapshotBindings::new();
        bindings.bind(StateGroupId::new(1), StateId::new(10), SnapshotId::new(1));
        bindings.bind(StateGroupId::new(2), StateId::new(20), SnapshotId::new(2));

        let mut m = SnapshotMixer::new(registry(), SnapshotConfig::default());
        assert!(m.drive_from_states(&mgr, &bindings, Some(0.0), Some(Interpolation::Linear)));
        m.advance(0.0);
        // Equal-weight blend of 0 and 10 is 5.
        assert!((m.parameter(ParameterId::new(1)).expect("present") - 5.0).abs() < EPS);
    }

    #[test]
    fn accessors_expose_registry_and_config() {
        let m = SnapshotMixer::new(registry(), SnapshotConfig::new(1.0, Interpolation::Linear));
        assert_eq!(m.registry().len(), 2);
        assert_eq!(m.config().default_interpolation, Interpolation::Linear);
    }
}
