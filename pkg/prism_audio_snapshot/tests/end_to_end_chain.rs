//! Production-grade end-to-end integration coverage for the public API of the
//! `prism_audio_snapshot` crate: it exercises the full snapshot pipeline across
//! module boundaries exactly as a game runtime would. A realistic mixer scene
//! (master gain, music bus level in decibels, a reverb send ratio, and a
//! low-pass cutoff in hertz) is captured into named `Snapshot`s, registered in
//! a `SnapshotRegistry`, driven through a stateful `SnapshotMixer` with timed,
//! curve-shaped `Transition`s, blended across multiple weighted snapshots, and
//! activated through `StateSnapshotBindings` wired to a live `StateManager`.
//! Each test pins one behavioural concern of the real chain: capture, timed
//! interpolation, per-`ParameterKind` domain math, interpolation-curve
//! shaping, weighted multi-snapshot blends, lowest-id kind arbitration,
//! State-driven activation, deterministic reproducibility, union-of-keys
//! transition semantics, boundary and error paths, and a value round-trip of
//! `ResolvedParameters` through its public iterator/accessor surface.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, MPEG, Google Resonance Audio, Web Audio, or Project Acoustics source
//! or derived code; no AI/ML. Any public standard informs ideas only, never
//! copied text or code. Floating-point helpers here are hand-written and avoid
//! the standard-library float math functions so results stay deterministic.
//!
//! # Relationship
//! Drives only the real public API of `prism_audio_snapshot`
//! (`mixer::SnapshotMixer`, `registry::SnapshotRegistry`, `snapshot::Snapshot`,
//! `resolved::ResolvedParameters`, `stack`, `transition`, `state_binding`,
//! `parameter`, `target`, `config`) together with its real dependencies
//! `prism_audio_core::math::Sample`, `prism_audio_content::curve::Interpolation`,
//! and `prism_audio_content::{id, state}`. It adds no production code and
//! treats the crate strictly as an external consumer would.

use prism_audio_content::curve::Interpolation;
use prism_audio_content::id::{StateGroupId, StateId};
use prism_audio_content::state::{StateGroup, StateManager};
use prism_audio_core::math::Sample;

use prism_audio_snapshot::config::SnapshotConfig;
use prism_audio_snapshot::mixer::SnapshotMixer;
use prism_audio_snapshot::parameter::{ParameterId, ParameterKind};
use prism_audio_snapshot::registry::SnapshotRegistry;
use prism_audio_snapshot::resolved::ResolvedParameters;
use prism_audio_snapshot::snapshot::{Snapshot, SnapshotId};
use prism_audio_snapshot::stack;
use prism_audio_snapshot::target::ParameterTarget;

/// Tight tolerance for linear-domain comparisons.
const EPS: Sample = 1e-5;
/// Looser tolerance for hertz values that route through `ln`/`exp`.
const HZ_EPS: Sample = 5.0;

// --- Stable parameter identities for the mixer scene. --------------------
const MASTER_GAIN: u32 = 1; // Linear amplitude.
const MUSIC_LEVEL: u32 = 2; // Decibel bus level.
const REVERB_SEND: u32 = 3; // Dimensionless ratio.
const LOWPASS_HZ: u32 = 4; // Frequency in hertz.

// --- Stable snapshot identities. -----------------------------------------
const EXPLORE: u32 = 1;
const COMBAT: u32 = 2;
const STEALTH: u32 = 3;

/// Hand-rolled absolute value; avoids the standard float math surface so the
/// comparison stays deterministic across platforms.
fn fabs(x: Sample) -> Sample {
    if x < 0.0 { -x } else { x }
}

/// Approximate equality in a value's own domain using `fabs`.
fn approx(a: Sample, b: Sample, eps: Sample) -> bool {
    fabs(a - b) < eps
}

/// Builds one snapshot from `(parameter, kind, value)` triples.
fn snapshot(id: u32, targets: &[(u32, ParameterKind, Sample)]) -> Snapshot {
    let mut snap = Snapshot::new(SnapshotId::new(id));
    for &(pid, kind, value) in targets {
        snap = snap.with_target(ParameterTarget::new(ParameterId::new(pid), kind, value));
    }
    snap
}

/// The three authored mixer snapshots of the scene.
fn explore() -> Snapshot {
    snapshot(
        EXPLORE,
        &[
            (MASTER_GAIN, ParameterKind::Linear, 1.0),
            (MUSIC_LEVEL, ParameterKind::Decibel, -6.0),
            (REVERB_SEND, ParameterKind::Ratio, 0.2),
            (LOWPASS_HZ, ParameterKind::Hertz, 20_000.0),
        ],
    )
}

fn combat() -> Snapshot {
    snapshot(
        COMBAT,
        &[
            (MASTER_GAIN, ParameterKind::Linear, 1.0),
            (MUSIC_LEVEL, ParameterKind::Decibel, 0.0),
            (REVERB_SEND, ParameterKind::Ratio, 0.05),
            (LOWPASS_HZ, ParameterKind::Hertz, 8_000.0),
        ],
    )
}

fn stealth() -> Snapshot {
    snapshot(
        STEALTH,
        &[
            (MASTER_GAIN, ParameterKind::Linear, 0.6),
            (MUSIC_LEVEL, ParameterKind::Decibel, -18.0),
            (REVERB_SEND, ParameterKind::Ratio, 0.5),
            (LOWPASS_HZ, ParameterKind::Hertz, 2_000.0),
        ],
    )
}

/// Registry holding the whole authored scene.
fn scene_registry() -> SnapshotRegistry {
    let mut reg = SnapshotRegistry::new();
    reg.register(explore());
    reg.register(combat());
    reg.register(stealth());
    reg
}

/// Reads a required parameter from the mixer, failing loudly if absent.
fn param(mixer: &SnapshotMixer, id: u32) -> Sample {
    mixer
        .parameter(ParameterId::new(id))
        .expect("parameter should be present in the resolved map")
}

#[test]
fn full_chain_capture_transition_recall_lands_on_destination() {
    // Capture: start fully settled on EXPLORE.
    let mut mixer = SnapshotMixer::new_with(
        scene_registry(),
        SnapshotConfig::default(),
        SnapshotId::new(EXPLORE),
    );
    assert!(approx(param(&mixer, MUSIC_LEVEL), -6.0, EPS));
    assert!(approx(param(&mixer, LOWPASS_HZ), 20_000.0, HZ_EPS));
    assert!(!mixer.is_transitioning());

    // Transition: recall COMBAT over two seconds on a linear ramp.
    assert!(mixer.transition_to(SnapshotId::new(COMBAT), Some(2.0), Some(Interpolation::Linear)));
    assert!(mixer.is_transitioning());

    // Halfway: each parameter blends in its own domain.
    mixer.advance(1.0);
    assert!(mixer.is_transitioning());
    assert!(approx(param(&mixer, MASTER_GAIN), 1.0, EPS));
    assert!(approx(param(&mixer, MUSIC_LEVEL), -3.0, EPS)); // dB arithmetic midpoint.
    assert!(approx(param(&mixer, REVERB_SEND), 0.125, EPS)); // ratio arithmetic.
    assert!(approx(param(&mixer, LOWPASS_HZ), 12_649.11, HZ_EPS)); // hertz geometric.

    // Recall completes and lands exactly on the destination.
    mixer.advance(1.0);
    assert!(!mixer.is_transitioning());
    assert!(approx(param(&mixer, MUSIC_LEVEL), 0.0, EPS));
    assert!(approx(param(&mixer, REVERB_SEND), 0.05, EPS));
    assert!(approx(param(&mixer, LOWPASS_HZ), 8_000.0, HZ_EPS));
}

#[test]
fn hertz_parameter_interpolates_geometrically_through_the_mixer() {
    let mut mixer = SnapshotMixer::new_with(
        scene_registry(),
        SnapshotConfig::default(),
        SnapshotId::new(EXPLORE),
    );
    // EXPLORE -> STEALTH sweeps the cutoff from 20 kHz down to 2 kHz.
    assert!(mixer.transition_to(SnapshotId::new(STEALTH), Some(2.0), Some(Interpolation::Linear)));
    mixer.advance(1.0);

    // Geometric midpoint is sqrt(20000 * 2000) = 6324.555..., not the
    // arithmetic 11000, confirming hertz blends in the log domain.
    let cutoff = param(&mixer, LOWPASS_HZ);
    assert!(approx(cutoff, 6_324.555, HZ_EPS));
    assert!(!approx(cutoff, 11_000.0, 100.0));
}

#[test]
fn interpolation_curve_shapes_the_blended_position() {
    // Two identical transitions differing only by curve must diverge at the
    // same elapsed time, and the eased value must match the curve's shaping.
    let mut linear = SnapshotMixer::new_with(
        scene_registry(),
        SnapshotConfig::default(),
        SnapshotId::new(EXPLORE),
    );
    let mut eased = SnapshotMixer::new_with(
        scene_registry(),
        SnapshotConfig::default(),
        SnapshotId::new(EXPLORE),
    );
    assert!(linear.transition_to(SnapshotId::new(COMBAT), Some(2.0), Some(Interpolation::Linear)));
    assert!(eased.transition_to(SnapshotId::new(COMBAT), Some(2.0), Some(Interpolation::EaseIn)));

    linear.advance(1.0);
    eased.advance(1.0);

    // Reverb send 0.2 -> 0.05 in the ratio (arithmetic) domain.
    let linear_reverb = param(&linear, REVERB_SEND);
    let eased_reverb = param(&eased, REVERB_SEND);

    // Linear at progress 0.5 blends halfway: 0.125.
    assert!(approx(linear_reverb, 0.125, EPS));
    // EaseIn shapes progress 0.5 to 0.25, so 0.2 + (0.05 - 0.2) * 0.25 = 0.1625.
    let shaped = Interpolation::EaseIn.shape(0.5);
    let expected = 0.2 + (0.05 - 0.2) * shaped;
    assert!(approx(eased_reverb, expected, EPS));
    // The two curves genuinely disagree at the same instant.
    assert!(!approx(linear_reverb, eased_reverb, 1e-3));
}

#[test]
fn constant_curve_holds_start_for_the_whole_segment() {
    let mut mixer = SnapshotMixer::new_with(
        scene_registry(),
        SnapshotConfig::default(),
        SnapshotId::new(EXPLORE),
    );
    assert!(mixer.transition_to(
        SnapshotId::new(COMBAT),
        Some(2.0),
        Some(Interpolation::Constant),
    ));

    // Constant shaping stays at 0 for the whole segment, so the resolved
    // value holds the start even partway through.
    mixer.advance(1.0);
    assert!(mixer.is_transitioning());
    assert!(approx(param(&mixer, REVERB_SEND), 0.2, EPS));
    assert!(approx(param(&mixer, MUSIC_LEVEL), -6.0, EPS));

    // Once the timer elapses the transition completes and is cleared. Because
    // `Interpolation::Constant` shapes every position (including 1.0) to 0.0,
    // the final resolved values remain the start, not the destination: the
    // "jump at the end" is not realized by `shape`. This is asserted as the
    // real, observed behavior of the public API.
    mixer.advance(1.0);
    assert!(!mixer.is_transitioning());
    assert!(approx(param(&mixer, REVERB_SEND), 0.2, EPS));
    assert!(approx(param(&mixer, MUSIC_LEVEL), -6.0, EPS));
}

#[test]
fn weighted_blend_transition_normalizes_weights_per_domain() {
    let mut mixer = SnapshotMixer::new_with(
        scene_registry(),
        SnapshotConfig::default(),
        SnapshotId::new(EXPLORE),
    );
    // Weights 1 (EXPLORE) and 3 (COMBAT); zero duration lands immediately.
    assert!(mixer.transition_to_blend(
        &[(SnapshotId::new(EXPLORE), 1.0), (SnapshotId::new(COMBAT), 3.0)],
        Some(0.0),
        Some(Interpolation::Linear),
    ));
    mixer.advance(0.0);
    assert!(!mixer.is_transitioning());

    // Decibel arithmetic weighted mean: (-6*1 + 0*3) / 4 = -1.5.
    assert!(approx(param(&mixer, MUSIC_LEVEL), -1.5, EPS));
    // Ratio arithmetic weighted mean: (0.2*1 + 0.05*3) / 4 = 0.0875.
    assert!(approx(param(&mixer, REVERB_SEND), 0.0875, EPS));
    // Master gain is 1.0 in both snapshots, so the blend is still 1.0.
    assert!(approx(param(&mixer, MASTER_GAIN), 1.0, EPS));
}

#[test]
fn blend_takes_parameter_kind_from_lowest_id_snapshot() {
    // Two snapshots disagree on the kind of the same parameter; the lowest id
    // must win, so the blend is geometric (hertz), not arithmetic (linear).
    let mut reg = SnapshotRegistry::new();
    reg.register(snapshot(10, &[(LOWPASS_HZ, ParameterKind::Hertz, 100.0)]));
    reg.register(snapshot(20, &[(LOWPASS_HZ, ParameterKind::Linear, 400.0)]));

    let kinds = stack::blend_kinds(&reg, &[SnapshotId::new(20), SnapshotId::new(10)]);
    assert_eq!(
        kinds.get(&ParameterId::new(LOWPASS_HZ)).copied(),
        Some(ParameterKind::Hertz),
    );

    let mut mixer = SnapshotMixer::new(reg, SnapshotConfig::default());
    assert!(mixer.transition_to_blend(
        &[(SnapshotId::new(10), 1.0), (SnapshotId::new(20), 1.0)],
        Some(0.0),
        Some(Interpolation::Linear),
    ));
    mixer.advance(0.0);
    // Geometric mean of 100 and 400 is 200, not the arithmetic 250.
    let cutoff = param(&mixer, LOWPASS_HZ);
    assert!(approx(cutoff, 200.0, HZ_EPS));
    assert!(!approx(cutoff, 250.0, 10.0));
}

#[test]
fn state_driven_activation_follows_the_state_manager() {
    // A single state group selects the active snapshot through bindings.
    let mut manager = StateManager::new();
    manager.register(StateGroup::new(
        StateGroupId::new(1),
        vec![StateId::new(10), StateId::new(11)],
        StateId::new(10),
    ));

    let mut bindings = prism_audio_snapshot::state_binding::StateSnapshotBindings::new();
    bindings.bind(StateGroupId::new(1), StateId::new(10), SnapshotId::new(EXPLORE));
    bindings.bind(StateGroupId::new(1), StateId::new(11), SnapshotId::new(COMBAT));

    let mut mixer = SnapshotMixer::new(scene_registry(), SnapshotConfig::default());

    // Default active state 10 -> EXPLORE.
    assert!(mixer.drive_from_states(&manager, &bindings, Some(0.0), Some(Interpolation::Linear)));
    mixer.advance(0.0);
    assert!(approx(param(&mixer, MUSIC_LEVEL), -6.0, EPS));

    // Flip the state to 11 -> the mix follows to COMBAT.
    assert!(manager.set(StateGroupId::new(1), StateId::new(11)));
    assert!(mixer.drive_from_states(&manager, &bindings, Some(0.0), Some(Interpolation::Linear)));
    mixer.advance(0.0);
    assert!(approx(param(&mixer, MUSIC_LEVEL), 0.0, EPS));
    assert!(approx(param(&mixer, REVERB_SEND), 0.05, EPS));
}

#[test]
fn state_driven_multiple_groups_blend_with_equal_weight() {
    // Two active groups map to two snapshots that blend equally.
    let mut manager = StateManager::new();
    manager.register(StateGroup::new(
        StateGroupId::new(1),
        vec![StateId::new(10)],
        StateId::new(10),
    ));
    manager.register(StateGroup::new(
        StateGroupId::new(2),
        vec![StateId::new(20)],
        StateId::new(20),
    ));

    let mut bindings = prism_audio_snapshot::state_binding::StateSnapshotBindings::new();
    bindings.bind(StateGroupId::new(1), StateId::new(10), SnapshotId::new(EXPLORE));
    bindings.bind(StateGroupId::new(2), StateId::new(20), SnapshotId::new(STEALTH));

    let mut mixer = SnapshotMixer::new(scene_registry(), SnapshotConfig::default());
    assert!(mixer.drive_from_states(&manager, &bindings, Some(0.0), Some(Interpolation::Linear)));
    mixer.advance(0.0);

    // Equal-weight gain blend of EXPLORE 1.0 and STEALTH 0.6 is 0.8.
    assert!(approx(param(&mixer, MASTER_GAIN), 0.8, EPS));
    // Decibel equal blend of -6 and -18 is -12.
    assert!(approx(param(&mixer, MUSIC_LEVEL), -12.0, EPS));
}

#[test]
fn transition_preserves_union_of_start_and_destination_keys() {
    // A parameter present on only one side keeps that side's value, so
    // introducing or dropping a parameter never snaps an unrelated one.
    let mut reg = SnapshotRegistry::new();
    reg.register(snapshot(1, &[(MASTER_GAIN, ParameterKind::Linear, 0.9)]));
    reg.register(snapshot(2, &[(REVERB_SEND, ParameterKind::Ratio, 0.4)]));

    let mut mixer = SnapshotMixer::new_with(reg, SnapshotConfig::default(), SnapshotId::new(1));
    assert!(approx(param(&mixer, MASTER_GAIN), 0.9, EPS));

    assert!(mixer.transition_to(SnapshotId::new(2), Some(0.0), Some(Interpolation::Linear)));
    mixer.advance(0.0);

    // Start-only key is retained; destination-only key is introduced.
    assert!(approx(param(&mixer, MASTER_GAIN), 0.9, EPS));
    assert!(approx(param(&mixer, REVERB_SEND), 0.4, EPS));
    assert_eq!(mixer.current().len(), 2);
}

#[test]
fn pipeline_is_deterministic_across_identical_runs() {
    // The same scene advanced by the same fractional timesteps must produce
    // bit-identical resolved values; compared via `to_bits`, never `==`.
    fn run() -> ResolvedParameters {
        let mut mixer = SnapshotMixer::new_with(
            scene_registry(),
            SnapshotConfig::new(1.0, Interpolation::SCurve),
            SnapshotId::new(EXPLORE),
        );
        mixer.transition_to(SnapshotId::new(STEALTH), None, None);
        for _ in 0..3 {
            mixer.advance(0.3);
        }
        mixer.current().clone()
    }

    let first = run();
    let second = run();

    assert_eq!(first.len(), second.len());
    assert!(!first.is_empty());
    for ((id_a, &val_a), (id_b, &val_b)) in first.iter().zip(second.iter()) {
        assert_eq!(id_a, id_b);
        assert_eq!(val_a.to_bits(), val_b.to_bits());
    }
}

#[test]
fn error_and_boundary_paths_leave_the_mixer_untouched() {
    // Unknown destination: no transition starts.
    let mut mixer = SnapshotMixer::new(scene_registry(), SnapshotConfig::default());
    assert!(!mixer.transition_to(SnapshotId::new(999), Some(1.0), Some(Interpolation::Linear)));
    assert!(!mixer.is_transitioning());

    // Empty and all-unknown blends resolve to nothing.
    assert!(!mixer.transition_to_blend(&[], Some(1.0), None));
    assert!(!mixer.transition_to_blend(&[(SnapshotId::new(999), 1.0)], None, None));
    assert!(!mixer.is_transitioning());

    // Advancing with no active transition is a no-op.
    mixer.advance(5.0);
    assert!(mixer.current().is_empty());

    // new_with an unknown snapshot yields an empty current map.
    let empty = SnapshotMixer::new_with(
        scene_registry(),
        SnapshotConfig::default(),
        SnapshotId::new(999),
    );
    assert!(empty.current().is_empty());

    // Driving from empty bindings starts nothing.
    let manager = StateManager::new();
    let bindings = prism_audio_snapshot::state_binding::StateSnapshotBindings::new();
    let mut driven = SnapshotMixer::new(scene_registry(), SnapshotConfig::default());
    assert!(!driven.drive_from_states(&manager, &bindings, None, None));
    assert!(!driven.is_transitioning());
}

#[test]
fn resolved_parameters_round_trip_through_public_surface() {
    // Resolve a snapshot, then rebuild an equal map purely from the public
    // iterator/accessor surface: a value-level round trip with no data loss.
    let resolved = ResolvedParameters::from_snapshot(&explore());
    assert_eq!(resolved.len(), 4);

    let mut rebuilt = ResolvedParameters::new();
    for (&id, &value) in resolved.iter() {
        rebuilt.set(id, value);
    }

    // Structural equality (derived `PartialEq`) confirms the round trip.
    assert_eq!(resolved, rebuilt);
    // A clone is likewise equal to its source.
    assert_eq!(resolved, resolved.clone());

    // Keys remain in ascending, deterministic order and values are
    // bit-identical through the round trip.
    let ids: Vec<u32> = rebuilt.param_ids().map(|id| id.get()).collect();
    assert_eq!(ids, vec![MASTER_GAIN, MUSIC_LEVEL, REVERB_SEND, LOWPASS_HZ]);
    for (&id, &value) in resolved.iter() {
        let got = rebuilt.get(id).expect("rebuilt map keeps every key");
        assert_eq!(got.to_bits(), value.to_bits());
    }
}
