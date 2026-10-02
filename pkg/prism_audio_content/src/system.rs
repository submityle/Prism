//! The live **event system**: the runtime half of the content model.
//!
//! Where [`crate::model::ContentModel`] is immutable authored data, an
//! [`EventSystem`] holds the *mutable* game-sync state (active states,
//! per-object switches, live RTPC values, and per-container cursors) and turns
//! a posted [`crate::event::Event`] into a flat stream of
//! [`crate::action::ResolvedAction`]s naming concrete leaf sounds and
//! parameter writes. One model can back many independent systems (split-screen,
//! server-side mixdown, offline bounce); each system is a self-contained,
//! deterministic state machine seeded by a single `u64`.
//!
//! Posting is **control-rate** (invoked when the game raises an event), so this
//! layer may allocate; it does no per-sample DSP. The lower runtime
//! (`prism_audio_rt`) consumes the resolved stream and manages voices.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The
//! "post event → mutate game-sync state → flatten container trees" pipeline is
//! reconstructed from first principles over the crate's own plain data and the
//! deterministic [`crate::rng::Rng`]. No AI/ML.
//!
//! # Relationship
//!
//! [`EventSystem::new`] builds its [`StateManager`], [`SwitchManager`], and
//! [`RtpcRegistry`] from a [`ContentModel`]. [`EventSystem::post_event`] walks
//! [`crate::container::Container`] trees (reading the live switch/RTPC context),
//! evaluates [`crate::rtpc::RtpcBinding`]s into
//! [`crate::parameter::ParameterSetting`]s, and emits
//! [`crate::action::ResolvedAction`]s for the voice/graph layer to act on.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::action::{Action, ResolvedAction};
use crate::container::ContainerState;
use crate::id::{
    ContainerId, EventId, GameObjectId, Playable, RtpcId, SoundId, StateGroupId, StateId,
    SwitchGroupId, SwitchId,
};
use crate::model::ContentModel;
use crate::parameter::ParameterSetting;
use crate::rng::Rng;
use crate::rtpc::RtpcRegistry;
use crate::state::StateManager;
use crate::switch::SwitchManager;

/// Hard ceiling on container recursion depth.
///
/// Authoring data can accidentally (or maliciously) reference containers in a
/// cycle; this bound guarantees every traversal terminates without a stack
/// overflow or an unbounded resolved stream. It is deliberately generous: real
/// container trees are only a handful of levels deep.
pub const MAX_CONTAINER_DEPTH: usize = 32;

/// Live, mutable runtime state driving one deterministic stream of resolved
/// actions from an authored [`ContentModel`].
#[derive(Debug, Clone)]
pub struct EventSystem {
    states: StateManager,
    switches: SwitchManager,
    rtpc: RtpcRegistry,
    rtpc_values: BTreeMap<(GameObjectId, RtpcId), Sample>,
    container_states: BTreeMap<ContainerId, ContainerState>,
    rng: Rng,
}

impl EventSystem {
    /// Builds a system for `model`, seeding its managers from the model's
    /// registered groups/definitions and its RNG from `seed`.
    ///
    /// The resulting system is fully deterministic: two systems built from the
    /// same model and seed produce identical resolved streams for identical
    /// posting sequences.
    #[must_use]
    pub fn new(model: &ContentModel, seed: u64) -> Self {
        Self {
            states: model.build_state_manager(),
            switches: model.build_switch_manager(),
            rtpc: model.build_rtpc_registry(),
            rtpc_values: BTreeMap::new(),
            container_states: BTreeMap::new(),
            rng: Rng::new(seed),
        }
    }

    /// Posts the event `event` for game object `object`, appending every
    /// resulting [`ResolvedAction`] to `out`.
    ///
    /// Returns `true` when the event existed and its actions ran, `false` when
    /// no event with that id is registered (in which case `out` is untouched).
    /// Actions run in authored order; container trees are flattened to concrete
    /// leaf sounds, and RTPC changes are expanded into parameter writes.
    pub fn post_event(
        &mut self,
        model: &ContentModel,
        event: EventId,
        object: GameObjectId,
        out: &mut Vec<ResolvedAction>,
    ) -> bool {
        let Some(ev) = model.event(event) else {
            return false;
        };
        // `Action` is `Copy`, so collecting the ids up front releases the borrow
        // of `model` through `ev` before the mutable `self` calls below.
        for action in ev.actions().to_vec() {
            self.apply_action(model, action, object, out);
        }
        true
    }

    /// Applies one authored action, mutating live state and/or emitting
    /// resolved actions.
    fn apply_action(
        &mut self,
        model: &ContentModel,
        action: Action,
        object: GameObjectId,
        out: &mut Vec<ResolvedAction>,
    ) {
        match action {
            Action::Play(playable) => self.flatten_play(model, object, playable, 0.0, 0, out),
            Action::Stop(playable) => {
                let mut sounds = Vec::new();
                self.collect_sounds(model, playable, 0, &mut sounds);
                for sound in sounds {
                    out.push(ResolvedAction::StopSound { object, sound });
                }
            }
            Action::StopAll => out.push(ResolvedAction::StopAll { object }),
            Action::SetState { group, state } => {
                self.states.set(group, state);
            }
            Action::SetSwitch { group, switch } => {
                self.switches.set(object, group, switch);
            }
            Action::SetRtpc { rtpc, value } => self.set_rtpc(object, rtpc, value, out),
            Action::SetBusVolumeDb { bus, volume_db } => {
                out.push(ResolvedAction::SetBusVolumeDb { bus, volume_db });
            }
        }
    }

    /// Recursively flattens a playable into concrete [`ResolvedAction::PlaySound`]
    /// entries, accumulating per-level gain down the tree.
    fn flatten_play(
        &mut self,
        model: &ContentModel,
        object: GameObjectId,
        playable: Playable,
        gain_db: Sample,
        depth: usize,
        out: &mut Vec<ResolvedAction>,
    ) {
        if depth > MAX_CONTAINER_DEPTH {
            return;
        }
        match playable {
            Playable::Sound(sound) => {
                out.push(ResolvedAction::PlaySound { object, sound, gain_db });
            }
            Playable::Container(cid) => {
                let Some(container) = model.container(cid) else {
                    return;
                };
                let kind = &container.kind;
                // Resolve the live switch/blend context into locals *before*
                // borrowing `self`'s fields mutably below.
                let active_switch =
                    kind.switch_group().and_then(|group| self.switches.resolve(object, group));
                let blend_position =
                    kind.blend_rtpc().map_or(0.0, |rtpc| self.normalized_rtpc(object, rtpc));
                // Resolve one level into an owned pick list in an inner scope so
                // the `container_states`/`rng` field borrows end before we
                // recurse back through `&mut self`.
                let picks = {
                    let state =
                        self.container_states.entry(cid).or_default();
                    let mut picks = Vec::new();
                    kind.resolve_into(state, &mut self.rng, active_switch, blend_position, &mut picks);
                    picks
                };
                for pick in picks {
                    self.flatten_play(
                        model,
                        object,
                        pick.playable,
                        gain_db + pick.gain_db,
                        depth + 1,
                        out,
                    );
                }
            }
        }
    }

    /// Collects every distinct leaf [`SoundId`] statically reachable from
    /// `playable` into `out` (used by `Stop` traversal).
    ///
    /// This enumerates the whole static subtree regardless of runtime cursor or
    /// switch, so stopping a container stops every leaf it could have started.
    fn collect_sounds(
        &self,
        model: &ContentModel,
        playable: Playable,
        depth: usize,
        out: &mut Vec<SoundId>,
    ) {
        if depth > MAX_CONTAINER_DEPTH {
            return;
        }
        match playable {
            Playable::Sound(sound) => {
                if !out.contains(&sound) {
                    out.push(sound);
                }
            }
            Playable::Container(cid) => {
                if let Some(container) = model.container(cid) {
                    let mut children = Vec::new();
                    container.kind.collect_children(&mut children);
                    for child in children {
                        self.collect_sounds(model, child, depth + 1, out);
                    }
                }
            }
        }
    }

    /// Sets an RTPC value for `object`, storing the clamped value and emitting a
    /// [`ResolvedAction::SetParameter`] for every binding the RTPC drives.
    fn set_rtpc(
        &mut self,
        object: GameObjectId,
        rtpc: RtpcId,
        value: Sample,
        out: &mut Vec<ResolvedAction>,
    ) {
        let clamped = self.rtpc.definition(rtpc).map_or(value, |def| def.clamp(value));
        self.rtpc_values.insert((object, rtpc), clamped);
        let mut settings = Vec::new();
        self.rtpc.evaluate_into(rtpc, clamped, &mut settings);
        for setting in settings {
            out.push(ResolvedAction::SetParameter { object, setting });
        }
    }

    /// Returns the live value of `rtpc` for `object`: the explicitly-set value
    /// if any, else the definition default, else `0.0` for an unknown RTPC.
    #[must_use]
    pub fn rtpc_value(&self, object: GameObjectId, rtpc: RtpcId) -> Sample {
        if let Some(value) = self.rtpc_values.get(&(object, rtpc)) {
            return *value;
        }
        self.rtpc.definition(rtpc).map_or(0.0, |def| def.default)
    }

    /// Returns the live RTPC value normalised to `[0, 1]` against its declared
    /// range (used as a blend-container position). Degenerate ranges and
    /// unknown RTPCs normalise to `0.0`.
    #[must_use]
    pub fn normalized_rtpc(&self, object: GameObjectId, rtpc: RtpcId) -> Sample {
        let value = self.rtpc_value(object, rtpc);
        match self.rtpc.definition(rtpc) {
            Some(def) if def.max > def.min => {
                ((value - def.min) / (def.max - def.min)).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }

    /// Directly sets a global state (outside an event). Returns `false` when
    /// the group is unknown or the state is not a member of it.
    pub fn set_state(&mut self, group: StateGroupId, state: StateId) -> bool {
        self.states.set(group, state)
    }

    /// Returns the active state of `group`, if the group is registered.
    #[must_use]
    pub fn active_state(&self, group: StateGroupId) -> Option<StateId> {
        self.states.active(group)
    }

    /// Directly sets a per-object switch (outside an event). Returns `false`
    /// when the group is unknown or the switch is not a member of it.
    pub fn set_switch(
        &mut self,
        object: GameObjectId,
        group: SwitchGroupId,
        switch: SwitchId,
    ) -> bool {
        self.switches.set(object, group, switch)
    }

    /// Resolves the active switch for `(object, group)`, falling back to the
    /// group default. `None` only when the group is unknown.
    #[must_use]
    pub fn resolve_switch(
        &self,
        object: GameObjectId,
        group: SwitchGroupId,
    ) -> Option<SwitchId> {
        self.switches.resolve(object, group)
    }

    /// Directly sets an RTPC value (outside an event), returning the resolved
    /// parameter writes it produces.
    pub fn apply_rtpc(
        &mut self,
        object: GameObjectId,
        rtpc: RtpcId,
        value: Sample,
        out: &mut Vec<ResolvedAction>,
    ) {
        self.set_rtpc(object, rtpc, value, out);
    }

    /// Resets every container cursor/history to its initial condition without
    /// disturbing states, switches, or RTPC values.
    pub fn reset_containers(&mut self) {
        for state in self.container_states.values_mut() {
            state.reset();
        }
    }

    /// Resets the whole system to a freshly-constructed condition for the given
    /// `seed`: states return to defaults, per-object switches and live RTPC
    /// values are cleared, and all container cursors reset.
    pub fn reset(&mut self, seed: u64) {
        self.states.reset_to_defaults();
        self.rtpc_values.clear();
        self.container_states.clear();
        self.rng = Rng::new(seed);
    }

    /// Borrows the live state manager (diagnostics/inspection).
    #[must_use]
    pub fn states(&self) -> &StateManager {
        &self.states
    }

    /// Borrows the live switch manager (diagnostics/inspection).
    #[must_use]
    pub fn switches(&self) -> &SwitchManager {
        &self.switches
    }

    /// Borrows the RTPC registry (diagnostics/inspection).
    #[must_use]
    pub fn rtpc_registry(&self) -> &RtpcRegistry {
        &self.rtpc
    }
}

/// Convenience: resolve a single [`ParameterSetting`] write into a
/// [`ResolvedAction::SetParameter`] for `object`.
#[must_use]
pub fn parameter_action(object: GameObjectId, setting: ParameterSetting) -> ResolvedAction {
    ResolvedAction::SetParameter { object, setting }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::container::{BlendLayer, Container, ContainerKind, RandomMode, WeightedChild};
    use crate::curve::ParameterCurve;
    use crate::event::Event;
    use crate::parameter::ParameterTarget;
    use crate::rtpc::{RtpcBinding, RtpcDefinition};
    use crate::state::StateGroup;
    use crate::switch::SwitchGroup;

    const EPS: Sample = 1e-4;

    fn close(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= EPS
    }

    #[test]
    fn unknown_event_returns_false_and_leaves_out_untouched() {
        let model = ContentModel::new();
        let mut system = EventSystem::new(&model, 1);
        let mut out = Vec::new();
        let ran = system.post_event(&model, EventId::new(999), GameObjectId::new(1), &mut out);
        assert!(!ran);
        assert!(out.is_empty());
    }

    #[test]
    fn nested_blend_containers_accumulate_gain_down_the_tree() {
        // outer blend (-6 dB) -> inner blend (-3 dB) -> leaf sound.
        let inner = Container::new(
            ContainerId::new(2),
            ContainerKind::Blend {
                rtpc: RtpcId::new(100),
                layers: vec![BlendLayer::new(
                    Playable::Sound(SoundId::new(7)),
                    ParameterCurve::constant(-3.0),
                )],
            },
        );
        let outer = Container::new(
            ContainerId::new(1),
            ContainerKind::Blend {
                rtpc: RtpcId::new(100),
                layers: vec![BlendLayer::new(
                    Playable::Container(ContainerId::new(2)),
                    ParameterCurve::constant(-6.0),
                )],
            },
        );
        let mut model = ContentModel::new();
        model.add_container(inner);
        model.add_container(outer);
        model.add_event(
            Event::new(EventId::new(1))
                .with(Action::Play(Playable::Container(ContainerId::new(1)))),
        );

        let mut system = EventSystem::new(&model, 1);
        let mut out = Vec::new();
        assert!(system.post_event(&model, EventId::new(1), GameObjectId::new(1), &mut out));
        assert_eq!(out.len(), 1);
        match out[0] {
            ResolvedAction::PlaySound { object, sound, gain_db } => {
                assert_eq!(object, GameObjectId::new(1));
                assert_eq!(sound, SoundId::new(7));
                assert!(close(gain_db, -9.0), "gain {gain_db} != -9");
            }
            ref other => panic!("expected PlaySound, got {other:?}"),
        }
    }

    #[test]
    fn stop_traverses_subtree_and_dedups_leaf_sounds() {
        // Container with two children, one duplicated, so Stop emits each once.
        let container = Container::new(
            ContainerId::new(1),
            ContainerKind::Sequence {
                children: vec![
                    Playable::Sound(SoundId::new(5)),
                    Playable::Sound(SoundId::new(6)),
                    Playable::Sound(SoundId::new(5)),
                ],
                mode: crate::container::SequenceMode::Loop,
            },
        );
        let mut model = ContentModel::new();
        model.add_container(container);
        model.add_event(
            Event::new(EventId::new(1))
                .with(Action::Stop(Playable::Container(ContainerId::new(1)))),
        );

        let mut system = EventSystem::new(&model, 1);
        let mut out = Vec::new();
        assert!(system.post_event(&model, EventId::new(1), GameObjectId::new(3), &mut out));
        assert_eq!(out.len(), 2);
        let stopped: Vec<SoundId> = out
            .iter()
            .map(|a| match *a {
                ResolvedAction::StopSound { sound, .. } => sound,
                ref other => panic!("expected StopSound, got {other:?}"),
            })
            .collect();
        assert_eq!(stopped, vec![SoundId::new(5), SoundId::new(6)]);
    }

    #[test]
    fn stop_all_emits_scoped_stop_all() {
        let mut model = ContentModel::new();
        model.add_event(Event::new(EventId::new(1)).with(Action::StopAll));
        let mut system = EventSystem::new(&model, 1);
        let mut out = Vec::new();
        assert!(system.post_event(&model, EventId::new(1), GameObjectId::new(9), &mut out));
        assert_eq!(out.len(), 1);
        match out[0] {
            ResolvedAction::StopAll { object } => assert_eq!(object, GameObjectId::new(9)),
            ref other => panic!("expected StopAll, got {other:?}"),
        }
    }

    #[test]
    fn set_rtpc_emits_one_set_parameter_per_binding() {
        let mut model = ContentModel::new();
        model.add_rtpc(RtpcDefinition::new(RtpcId::new(1), 0.0, 100.0, 0.0));
        model.add_rtpc_binding(RtpcBinding::new(
            RtpcId::new(1),
            ParameterTarget::VolumeDb,
            ParameterCurve::line(0.0, -60.0, 100.0, 0.0),
        ));
        model.add_rtpc_binding(RtpcBinding::new(
            RtpcId::new(1),
            ParameterTarget::LowpassCutoffHz,
            ParameterCurve::line(0.0, 1000.0, 100.0, 20000.0),
        ));
        model.add_event(
            Event::new(EventId::new(1))
                .with(Action::SetRtpc { rtpc: RtpcId::new(1), value: 50.0 }),
        );

        let mut system = EventSystem::new(&model, 1);
        let mut out = Vec::new();
        assert!(system.post_event(&model, EventId::new(1), GameObjectId::new(2), &mut out));
        assert_eq!(out.len(), 2);
        for action in &out {
            match *action {
                ResolvedAction::SetParameter { object, setting } => {
                    assert_eq!(object, GameObjectId::new(2));
                    match setting.target {
                        ParameterTarget::VolumeDb => assert!(close(setting.value, -30.0)),
                        ParameterTarget::LowpassCutoffHz => {
                            assert!(close(setting.value, 10500.0));
                        }
                        other => panic!("unexpected target {other:?}"),
                    }
                }
                ref other => panic!("expected SetParameter, got {other:?}"),
            }
        }
        // The live value is clamped/stored for later reads.
        assert!(close(system.rtpc_value(GameObjectId::new(2), RtpcId::new(1)), 50.0));
    }

    #[test]
    fn set_state_action_mutates_live_state() {
        let mut model = ContentModel::new();
        model.add_state_group(StateGroup::new(
            StateGroupId::new(1),
            vec![StateId::new(10), StateId::new(11)],
            StateId::new(10),
        ));
        model.add_event(
            Event::new(EventId::new(1))
                .with(Action::SetState { group: StateGroupId::new(1), state: StateId::new(11) }),
        );
        let mut system = EventSystem::new(&model, 1);
        assert_eq!(system.active_state(StateGroupId::new(1)), Some(StateId::new(10)));
        let mut out = Vec::new();
        assert!(system.post_event(&model, EventId::new(1), GameObjectId::new(1), &mut out));
        assert!(out.is_empty());
        assert_eq!(system.active_state(StateGroupId::new(1)), Some(StateId::new(11)));
    }

    #[test]
    fn set_switch_action_mutates_per_object_switch() {
        let mut model = ContentModel::new();
        model.add_switch_group(SwitchGroup::new(
            SwitchGroupId::new(1),
            vec![SwitchId::new(20), SwitchId::new(21)],
            SwitchId::new(20),
        ));
        model.add_event(
            Event::new(EventId::new(1))
                .with(Action::SetSwitch { group: SwitchGroupId::new(1), switch: SwitchId::new(21) }),
        );
        let mut system = EventSystem::new(&model, 1);
        let obj = GameObjectId::new(4);
        assert_eq!(system.resolve_switch(obj, SwitchGroupId::new(1)), Some(SwitchId::new(20)));
        let mut out = Vec::new();
        assert!(system.post_event(&model, EventId::new(1), obj, &mut out));
        assert_eq!(system.resolve_switch(obj, SwitchGroupId::new(1)), Some(SwitchId::new(21)));
        // Another object still sees the default.
        assert_eq!(
            system.resolve_switch(GameObjectId::new(5), SwitchGroupId::new(1)),
            Some(SwitchId::new(20))
        );
    }

    #[test]
    fn self_referencing_container_terminates_via_depth_guard() {
        // A container whose only child is itself must not overflow the stack
        // or produce an unbounded stream.
        let container = Container::new(
            ContainerId::new(1),
            ContainerKind::Random {
                children: vec![WeightedChild::uniform(Playable::Container(ContainerId::new(1)))],
                mode: RandomMode::Standard,
                avoid_repeat: 0,
            },
        );
        let mut model = ContentModel::new();
        model.add_container(container);
        model.add_event(
            Event::new(EventId::new(1))
                .with(Action::Play(Playable::Container(ContainerId::new(1)))),
        );
        let mut system = EventSystem::new(&model, 1);
        let mut out = Vec::new();
        assert!(system.post_event(&model, EventId::new(1), GameObjectId::new(1), &mut out));
        // No leaf is ever reached, so nothing is played; crucially it returns.
        assert!(out.is_empty());
    }

    #[test]
    fn same_model_and_seed_yields_identical_resolved_streams() {
        let container = Container::new(
            ContainerId::new(1),
            ContainerKind::Random {
                children: vec![
                    WeightedChild::uniform(Playable::Sound(SoundId::new(1))),
                    WeightedChild::uniform(Playable::Sound(SoundId::new(2))),
                    WeightedChild::uniform(Playable::Sound(SoundId::new(3))),
                ],
                mode: RandomMode::Standard,
                avoid_repeat: 1,
            },
        );
        let mut model = ContentModel::new();
        model.add_container(container);
        model.add_event(
            Event::new(EventId::new(1))
                .with(Action::Play(Playable::Container(ContainerId::new(1)))),
        );

        let post_many = |seed: u64| {
            let mut system = EventSystem::new(&model, seed);
            let mut out = Vec::new();
            for _ in 0..16 {
                system.post_event(&model, EventId::new(1), GameObjectId::new(1), &mut out);
            }
            out
        };
        assert_eq!(post_many(42), post_many(42));
    }
}
