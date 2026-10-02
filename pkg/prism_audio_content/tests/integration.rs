//! End-to-end integration tests for the Prism audio content model.
//!
//! These exercise the public API surface only (as a downstream crate would):
//! authoring a [`ContentModel`], driving it through an [`EventSystem`], and
//! asserting on the flattened [`ResolvedAction`] stream. They complement the
//! per-module unit tests by covering cross-module interactions:
//! nested-container flattening, Stop subtree traversal with de-duplication,
//! StopAll, RTPC fan-out, State/Switch mutation, cyclic-container depth
//! guarding, and determinism across independent systems.
//!
//! # Provenance
//!
//! Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code. Pure black-box exercise of
//! this crate's own public API. No AI/ML.
//!
//! # Relationship
//!
//! Links against `prism_audio_content` with the `serialize` feature enabled
//! (see `[dev-dependencies]`), so it also confirms the public types compile
//! and resolve under that feature combination.

use prism_audio_content::action::{Action, ResolvedAction};
use prism_audio_content::container::{
    BlendLayer, Container, ContainerKind, RandomMode, SequenceMode, SwitchBranch, WeightedChild,
};
use prism_audio_content::curve::ParameterCurve;
use prism_audio_content::event::Event;
use prism_audio_content::id::{
    ContainerId, EventId, GameObjectId, Playable, RtpcId, SoundId, StateGroupId, StateId,
    SwitchGroupId, SwitchId,
};
use prism_audio_content::model::ContentModel;
use prism_audio_content::parameter::ParameterTarget;
use prism_audio_content::rtpc::{RtpcBinding, RtpcDefinition};
use prism_audio_content::state::StateGroup;
use prism_audio_content::switch::SwitchGroup;
use prism_audio_content::system::EventSystem;
use prism_audio_core::math::Sample;

const EPS: Sample = 1e-4;

fn close(a: Sample, b: Sample) -> bool {
    (a - b).abs() <= EPS
}

/// A sequence container wrapping a leaf, nested inside a blend container, is
/// flattened to a single `PlaySound` carrying the blend layer's gain.
#[test]
fn nested_containers_flatten_to_leaf_with_accumulated_gain() {
    let inner = Container::new(
        ContainerId::new(2),
        ContainerKind::Sequence {
            children: vec![Playable::Sound(SoundId::new(42))],
            mode: SequenceMode::Once,
        },
    );
    let outer = Container::new(
        ContainerId::new(1),
        ContainerKind::Blend {
            rtpc: RtpcId::new(1),
            layers: vec![BlendLayer::new(
                Playable::Container(ContainerId::new(2)),
                ParameterCurve::constant(-12.0),
            )],
        },
    );
    let mut model = ContentModel::new();
    model.add_container(inner);
    model.add_container(outer);
    model.add_event(
        Event::new(EventId::new(1)).with(Action::Play(Playable::Container(ContainerId::new(1)))),
    );

    let mut system = EventSystem::new(&model, 7);
    let mut out = Vec::new();
    assert!(system.post_event(&model, EventId::new(1), GameObjectId::new(1), &mut out));
    assert_eq!(out.len(), 1);
    match out[0] {
        ResolvedAction::PlaySound { sound, gain_db, .. } => {
            assert_eq!(sound, SoundId::new(42));
            assert!(close(gain_db, -12.0));
        }
        ref other => panic!("expected PlaySound, got {other:?}"),
    }
}

/// Stopping a container enumerates its whole static subtree once per distinct
/// leaf, independent of the runtime cursor.
#[test]
fn stop_container_dedups_leaves_across_whole_subtree() {
    let child = Container::new(
        ContainerId::new(2),
        ContainerKind::Sequence {
            children: vec![Playable::Sound(SoundId::new(1)), Playable::Sound(SoundId::new(2))],
            mode: SequenceMode::Loop,
        },
    );
    let root = Container::new(
        ContainerId::new(1),
        ContainerKind::Sequence {
            children: vec![
                Playable::Container(ContainerId::new(2)),
                Playable::Sound(SoundId::new(1)),
                Playable::Sound(SoundId::new(3)),
            ],
            mode: SequenceMode::Loop,
        },
    );
    let mut model = ContentModel::new();
    model.add_container(child);
    model.add_container(root);
    model.add_event(
        Event::new(EventId::new(1)).with(Action::Stop(Playable::Container(ContainerId::new(1)))),
    );

    let mut system = EventSystem::new(&model, 1);
    let mut out = Vec::new();
    assert!(system.post_event(&model, EventId::new(1), GameObjectId::new(1), &mut out));
    let stopped: Vec<SoundId> = out
        .iter()
        .map(|a| match *a {
            ResolvedAction::StopSound { sound, .. } => sound,
            ref other => panic!("expected StopSound, got {other:?}"),
        })
        .collect();
    assert_eq!(stopped, vec![SoundId::new(1), SoundId::new(2), SoundId::new(3)]);
}

/// A switch container selects the branch matching the active switch, and falls
/// back to its default when the active switch has no branch.
#[test]
fn switch_container_selects_branch_then_falls_back_to_default() {
    let container = Container::new(
        ContainerId::new(1),
        ContainerKind::Switch {
            group: SwitchGroupId::new(1),
            branches: vec![
                SwitchBranch::new(SwitchId::new(10), Playable::Sound(SoundId::new(100))),
                SwitchBranch::new(SwitchId::new(11), Playable::Sound(SoundId::new(101))),
            ],
            default: Some(Playable::Sound(SoundId::new(999))),
        },
    );
    let mut model = ContentModel::new();
    model.add_switch_group(SwitchGroup::new(
        SwitchGroupId::new(1),
        vec![SwitchId::new(10), SwitchId::new(11), SwitchId::new(12)],
        SwitchId::new(10),
    ));
    model.add_container(container);
    model.add_event(
        Event::new(EventId::new(1)).with(Action::Play(Playable::Container(ContainerId::new(1)))),
    );

    let obj = GameObjectId::new(1);
    let mut system = EventSystem::new(&model, 1);

    // Default active switch 10 -> sound 100.
    let mut out = Vec::new();
    assert!(system.post_event(&model, EventId::new(1), obj, &mut out));
    assert!(matches!(out[0], ResolvedAction::PlaySound { sound, .. } if sound == SoundId::new(100)));

    // Switch to 11 -> sound 101.
    assert!(system.set_switch(obj, SwitchGroupId::new(1), SwitchId::new(11)));
    out.clear();
    assert!(system.post_event(&model, EventId::new(1), obj, &mut out));
    assert!(matches!(out[0], ResolvedAction::PlaySound { sound, .. } if sound == SoundId::new(101)));

    // Switch to 12 (no branch) -> default sound 999.
    assert!(system.set_switch(obj, SwitchGroupId::new(1), SwitchId::new(12)));
    out.clear();
    assert!(system.post_event(&model, EventId::new(1), obj, &mut out));
    assert!(matches!(out[0], ResolvedAction::PlaySound { sound, .. } if sound == SoundId::new(999)));
}

/// Driving an RTPC through an event fans out to every bound parameter target
/// and records the clamped live value.
#[test]
fn rtpc_event_fans_out_to_all_bindings() {
    let mut model = ContentModel::new();
    model.add_rtpc(RtpcDefinition::new(RtpcId::new(1), 0.0, 100.0, 0.0));
    model.add_rtpc_binding(RtpcBinding::new(
        RtpcId::new(1),
        ParameterTarget::VolumeDb,
        ParameterCurve::line(0.0, -60.0, 100.0, 0.0),
    ));
    model.add_rtpc_binding(RtpcBinding::new(
        RtpcId::new(1),
        ParameterTarget::Pan,
        ParameterCurve::line(0.0, -1.0, 100.0, 1.0),
    ));
    model.add_event(
        Event::new(EventId::new(1)).with(Action::SetRtpc { rtpc: RtpcId::new(1), value: 150.0 }),
    );

    let obj = GameObjectId::new(2);
    let mut system = EventSystem::new(&model, 1);
    let mut out = Vec::new();
    assert!(system.post_event(&model, EventId::new(1), obj, &mut out));
    assert_eq!(out.len(), 2);
    // 150 clamps to 100 -> VolumeDb = 0, Pan = 1.
    for action in &out {
        match *action {
            ResolvedAction::SetParameter { setting, .. } => match setting.target {
                ParameterTarget::VolumeDb => assert!(close(setting.value, 0.0)),
                ParameterTarget::Pan => assert!(close(setting.value, 1.0)),
                other => panic!("unexpected target {other:?}"),
            },
            ref other => panic!("expected SetParameter, got {other:?}"),
        }
    }
    assert!(close(system.rtpc_value(obj, RtpcId::new(1)), 100.0));
}

/// State and switch set actions mutate live game-sync state, and StopAll is
/// scoped to the posting object.
#[test]
fn state_switch_and_stop_all_mutations() {
    let mut model = ContentModel::new();
    model.add_state_group(StateGroup::new(
        StateGroupId::new(1),
        vec![StateId::new(1), StateId::new(2)],
        StateId::new(1),
    ));
    model.add_event(
        Event::new(EventId::new(1))
            .with(Action::SetState { group: StateGroupId::new(1), state: StateId::new(2) })
            .with(Action::StopAll),
    );

    let obj = GameObjectId::new(8);
    let mut system = EventSystem::new(&model, 1);
    let mut out = Vec::new();
    assert!(system.post_event(&model, EventId::new(1), obj, &mut out));
    assert_eq!(system.active_state(StateGroupId::new(1)), Some(StateId::new(2)));
    assert_eq!(out.len(), 1);
    assert!(matches!(out[0], ResolvedAction::StopAll { object } if object == obj));
}

/// A container that references itself must terminate thanks to the depth guard.
#[test]
fn cyclic_container_is_depth_guarded() {
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
        Event::new(EventId::new(1)).with(Action::Play(Playable::Container(ContainerId::new(1)))),
    );
    let mut system = EventSystem::new(&model, 1);
    let mut out = Vec::new();
    assert!(system.post_event(&model, EventId::new(1), GameObjectId::new(1), &mut out));
    assert!(out.is_empty());
}

/// Two independent systems built from the same model and seed produce byte-for
/// byte identical resolved streams over a long posting sequence.
#[test]
fn determinism_across_independent_systems() {
    let container = Container::new(
        ContainerId::new(1),
        ContainerKind::Random {
            children: vec![
                WeightedChild::new(Playable::Sound(SoundId::new(1)), 1.0),
                WeightedChild::new(Playable::Sound(SoundId::new(2)), 2.0),
                WeightedChild::new(Playable::Sound(SoundId::new(3)), 3.0),
            ],
            mode: RandomMode::Shuffle,
            avoid_repeat: 0,
        },
    );
    let mut model = ContentModel::new();
    model.add_container(container);
    model.add_event(
        Event::new(EventId::new(1)).with(Action::Play(Playable::Container(ContainerId::new(1)))),
    );

    let drive = |seed: u64| {
        let mut system = EventSystem::new(&model, seed);
        let mut out = Vec::new();
        for _ in 0..64 {
            system.post_event(&model, EventId::new(1), GameObjectId::new(1), &mut out);
        }
        out
    };
    assert_eq!(drive(2026), drive(2026));
}
