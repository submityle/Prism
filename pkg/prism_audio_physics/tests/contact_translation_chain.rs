//! Whole-chain integration tests for the `prism_audio_physics` contact bridge.
//!
//! These tests drive the public surface of `prism_audio_physics` exactly as a
//! host integration layer would: they build per-body kinematic snapshots
//! (`BodyAudioState`), feed engine-agnostic `ContactManifoldView` facts into the
//! stateful `ContactAudioTranslator` tagged with a `ContactPhase`, and inspect
//! the drained `ContactDrive` of `ImpactEvent` / `SustainEvent` /
//! `SeparationEvent` values. The chain under test is
//! kinematics -> impulse -> offset -> material -> roughness -> merge -> cluster
//! -> budget, plus the stable-identity (`ContactKey` / `procedural_id`) and
//! acoustic-material (`MaterialResolver`) edges.
//!
//! # Provenance
//! Original work authored for Prism; it contains no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, Dolby, MPEG, Google Resonance Audio, Web
//! Audio, or Microsoft Project Acoustics source or derived code, and no AI/ML.
//! Any public acoustics standards are used only as ideas, never as copied text
//! or data.
//!
//! # Relationship
//! Exercises design section 47.1 (the physics-engine coupling bridge): the
//! contact event bus, impulse/energy mapping, near-coincident merge, far-field
//! cluster, and per-block budget. It depends only on the public API of
//! `prism_audio_physics` (and the `Vec3` vector type the API itself exposes in
//! its signatures); it never touches crate internals.

use bevy_math::Vec3;

use prism_audio_physics::body::{BodyAudioId, BodyAudioState};
use prism_audio_physics::config::TranslatorConfig;
use prism_audio_physics::contact_id::{procedural_id, ContactKey};
use prism_audio_physics::contact_input::{ContactManifoldView, ContactPhase, ContactPointView};
use prism_audio_physics::material::{AudioMaterialId, MaterialResolver};
use prism_audio_physics::offset::BlockClock;
use prism_audio_physics::roughness::surface_roughness;
use prism_audio_physics::translator::ContactAudioTranslator;

/// Absolute value without pulling in `f32::abs` (kept off std float math).
fn fabs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// Builds a dynamic/static body snapshot from scalar facts.
fn make_body(id: u64, linear: Vec3, inverse_mass: f32, material: u32) -> BodyAudioState {
    BodyAudioState::new(
        BodyAudioId(id),
        linear,
        Vec3::ZERO,
        Vec3::ZERO,
        inverse_mass,
        AudioMaterialId(material),
    )
}

/// Builds a one-point manifold with the physics-core `a`->`b` normal of `+Y`.
fn make_manifold(a: u64, b: u64, point: Vec3, penetration: f32) -> ContactManifoldView {
    ContactManifoldView::new(
        BodyAudioId(a),
        BodyAudioId(b),
        Vec3::Y,
        vec![ContactPointView::new(point, penetration)],
    )
}

/// A translator with default config and an empty (fallback-only) resolver.
fn make_translator() -> ContactAudioTranslator {
    ContactAudioTranslator::new(TranslatorConfig::default(), MaterialResolver::new())
}

#[test]
fn impact_chain_is_sample_accurate() {
    // Body `a` descends onto static body `b` along the `+Y` normal at 2 m/s.
    // The started contact must become exactly one impact, placed on the sample
    // matching the sub-step fraction, carrying the resolved identity and pair.
    let a = make_body(1, Vec3::new(0.0, 2.0, 0.0), 1.0, 10);
    let b = make_body(2, Vec3::ZERO, 0.0, 20);

    let run = || {
        let mut t = make_translator();
        t.begin_block(BlockClock::new(48_000, 128));
        let view = make_manifold(1, 2, Vec3::ZERO, 0.01);
        t.ingest(&view, &a, &b, ContactPhase::Started, 0.5);
        (t.tracked_contacts(), t.drain())
    };

    let (tracked, drive) = run();
    assert_eq!(drive.impacts.len(), 1);
    assert_eq!(drive.sustains.len(), 0);
    assert_eq!(drive.separations.len(), 0);
    assert_eq!(tracked, 1);

    let impact = drive.impacts[0];
    assert!(impact.impulse > 0.0);
    // Sub-step 0.5 of a 128-frame block lands on sample 64.
    assert_eq!(impact.sample_offset, 64);
    // The identity is the stable, order-independent procedural id of the pair.
    assert_eq!(impact.contact, procedural_id(ContactKey::new(BodyAudioId(1), BodyAudioId(2))));
    // The material pair matches an independent fallback resolution of (10, 20).
    let expected_pair = MaterialResolver::new().resolve(AudioMaterialId(10), AudioMaterialId(20));
    assert_eq!(impact.material_pair, expected_pair);

    // Same reset + same input must reproduce an identical drive bit-for-bit.
    let (_, drive_again) = run();
    assert_eq!(drive, drive_again);
}

#[test]
fn restitution_scales_impact_impulse() {
    // A perfectly elastic contact (e = 1) carries exactly twice the impulse of
    // a fully inelastic one (e = 0) for the same closing speed and masses.
    let a = make_body(1, Vec3::new(0.0, 2.0, 0.0), 1.0, 0);
    let b = make_body(2, Vec3::ZERO, 0.0, 0);

    let impulse_for = |restitution: f32| {
        let config = TranslatorConfig {
            restitution,
            ..Default::default()
        };
        let mut t = ContactAudioTranslator::new(config, MaterialResolver::new());
        t.begin_block(BlockClock::new(48_000, 128));
        let view = make_manifold(1, 2, Vec3::ZERO, 0.01);
        t.ingest(&view, &a, &b, ContactPhase::Started, 0.0);
        let drive = t.drain();
        assert_eq!(drive.impacts.len(), 1);
        drive.impacts[0].impulse
    };

    let inelastic = impulse_for(0.0);
    let elastic = impulse_for(1.0);
    // reduced mass 1, closing speed 2 => inelastic 2.0, elastic 4.0.
    assert!(fabs(inelastic - 2.0) < 1e-4);
    assert!(fabs(elastic - 4.0) < 1e-4);
    assert!(fabs(elastic - 2.0 * inelastic) < 1e-4);
}

#[test]
fn persisting_contact_reports_tangential_sustain() {
    // A body sliding tangentially (3 m/s along `+X`) across a `+Y` contact must
    // produce a sustain whose tangential speed is that grazing speed, and the
    // contact must stay tracked across the block boundary.
    let a = make_body(1, Vec3::new(3.0, 0.0, 0.0), 1.0, 5);
    let b = make_body(2, Vec3::ZERO, 0.0, 9);

    let mut t = make_translator();

    t.begin_block(BlockClock::new(48_000, 128));
    t.ingest(&make_manifold(1, 2, Vec3::ZERO, 0.02), &a, &b, ContactPhase::Started, 0.0);
    let _ = t.drain();

    t.begin_block(BlockClock::new(48_000, 128));
    t.ingest(&make_manifold(1, 2, Vec3::ZERO, 0.02), &a, &b, ContactPhase::Persisting, 0.0);
    let drive = t.drain();

    assert_eq!(drive.sustains.len(), 1);
    assert_eq!(drive.impacts.len(), 0);
    let sustain = drive.sustains[0];
    assert!(fabs(sustain.tangential_speed - 3.0) < 1e-4);
    assert!((0.0..=1.0).contains(&sustain.roughness));
    assert_eq!(t.tracked_contacts(), 1);
}

#[test]
fn separation_forgets_contact() {
    // Starting then ending a contact must emit one separation carrying the
    // contact's stable id, and the translator must drop it from tracking.
    let a = make_body(1, Vec3::new(0.0, 2.0, 0.0), 1.0, 1);
    let b = make_body(2, Vec3::ZERO, 0.0, 2);

    let mut t = make_translator();

    t.begin_block(BlockClock::new(48_000, 128));
    t.ingest(&make_manifold(1, 2, Vec3::ZERO, 0.01), &a, &b, ContactPhase::Started, 0.0);
    let _ = t.drain();
    assert_eq!(t.tracked_contacts(), 1);

    t.begin_block(BlockClock::new(48_000, 128));
    t.ingest(&make_manifold(1, 2, Vec3::ZERO, 0.0), &a, &b, ContactPhase::Ended, 0.0);
    let drive = t.drain();

    assert_eq!(drive.separations.len(), 1);
    assert_eq!(
        drive.separations[0].contact,
        procedural_id(ContactKey::new(BodyAudioId(1), BodyAudioId(2)))
    );
    assert_eq!(t.tracked_contacts(), 0);
}

#[test]
fn near_coincident_impacts_merge_to_one() {
    // Several sub-step strikes of the *same* contact within the merge window
    // (the frame-rate "machine gun") must collapse to a single impact.
    let a = make_body(1, Vec3::new(0.0, 2.0, 0.0), 1.0, 3);
    let b = make_body(2, Vec3::ZERO, 0.0, 4);

    let mut t = make_translator();
    t.begin_block(BlockClock::new(48_000, 512));
    for _ in 0..4 {
        t.ingest(&make_manifold(1, 2, Vec3::ZERO, 0.01), &a, &b, ContactPhase::Started, 0.0);
    }
    let drive = t.drain();
    assert_eq!(drive.impacts.len(), 1);
}

#[test]
fn far_crowd_clusters_but_near_detail_survives() {
    // Three distinct contacts far from the listener and mutually close collapse
    // into one summed-energy group impact, while a distinct near contact keeps
    // its own voice. Default cluster: radius 20 m, min group size 3.
    let mut t = make_translator();
    t.begin_block(BlockClock::new(48_000, 512));

    // Far, tight crowd around x = 100 (distinct body pairs so merge leaves them).
    let drop = Vec3::new(0.0, 2.0, 0.0);
    for i in 0..3u64 {
        let a = make_body(i * 2 + 1, drop, 1.0, 10);
        let b = make_body(i * 2 + 2, Vec3::ZERO, 0.0, 20);
        let point = Vec3::new(100.0 + i as f32, 0.0, 0.0);
        t.ingest(&make_manifold(i * 2 + 1, i * 2 + 2, point, 0.01), &a, &b, ContactPhase::Started, 0.0);
    }
    // One near contact at x = 1 (well inside the radius): must survive alone.
    let near_a = make_body(101, drop, 1.0, 10);
    let near_b = make_body(102, Vec3::ZERO, 0.0, 20);
    t.ingest(
        &make_manifold(101, 102, Vec3::new(1.0, 0.0, 0.0), 0.01),
        &near_a,
        &near_b,
        ContactPhase::Started,
        0.0,
    );

    let drive = t.drain();
    // One collapsed group + one surviving near impact.
    assert_eq!(drive.impacts.len(), 2);

    // Energy is conserved: three 2.6 impulses sum into the group's 7.8.
    let mut max_impulse = 0.0f32;
    for impact in &drive.impacts {
        if impact.impulse > max_impulse {
            max_impulse = impact.impulse;
        }
    }
    assert!(fabs(max_impulse - 7.8) < 1e-2);
}

#[test]
fn per_block_budget_keeps_loudest() {
    // With clustering disabled and a hard cap of two, five distinct contacts of
    // increasing closing speed must truncate to the two loudest, sorted down.
    let mut config = TranslatorConfig {
        max_impacts_per_block: 2,
        ..Default::default()
    };
    config.cluster.min_cluster_size = usize::MAX;

    let mut t = ContactAudioTranslator::new(config, MaterialResolver::new());
    t.begin_block(BlockClock::new(48_000, 512));
    for i in 0..5u64 {
        let a = make_body(i * 2 + 1, Vec3::new(0.0, 1.0 + i as f32, 0.0), 1.0, 10);
        let b = make_body(i * 2 + 2, Vec3::ZERO, 0.0, 20);
        let point = Vec3::new(i as f32 * 100.0, 0.0, 0.0);
        t.ingest(&make_manifold(i * 2 + 1, i * 2 + 2, point, 0.01), &a, &b, ContactPhase::Started, 0.0);
    }
    let drive = t.drain();
    assert_eq!(drive.impacts.len(), 2);
    assert!(drive.impacts[0].impulse >= drive.impacts[1].impulse);
    assert!(drive.impacts[1].impulse > 0.0);
}

#[test]
fn static_pair_is_silent() {
    // Two immovable bodies (both inverse mass 0) have zero reduced mass, so even
    // a reported started contact exchanges no momentum: a silent impact.
    let a = make_body(1, Vec3::new(0.0, 2.0, 0.0), 0.0, 10);
    let b = make_body(2, Vec3::ZERO, 0.0, 20);

    let mut t = make_translator();
    t.begin_block(BlockClock::new(48_000, 128));
    t.ingest(&make_manifold(1, 2, Vec3::ZERO, 0.01), &a, &b, ContactPhase::Started, 0.0);
    let drive = t.drain();

    assert_eq!(drive.impacts.len(), 1);
    assert!(fabs(drive.impacts[0].impulse) < 1e-6);
    assert!(fabs(drive.impacts[0].normal) < 1e-6);
    assert!(fabs(drive.impacts[0].tangential) < 1e-6);
}

#[test]
fn material_and_contact_identity_are_deterministic() {
    // The identity and material edges must be symmetric, stable, and total so a
    // persistent manifold keeps one voice and no pair is ever silent.
    let a = BodyAudioId(5);
    let b = BodyAudioId(42);
    assert_eq!(ContactKey::new(a, b), ContactKey::new(b, a));
    let id_ab = procedural_id(ContactKey::new(a, b));
    assert_eq!(id_ab, procedural_id(ContactKey::new(b, a)));
    assert_eq!(id_ab, procedural_id(ContactKey::new(a, b)));
    assert_ne!(
        id_ab,
        procedural_id(ContactKey::new(BodyAudioId(5), BodyAudioId(43)))
    );

    let mut resolver = MaterialResolver::new();
    let m0 = AudioMaterialId(11);
    let m1 = AudioMaterialId(97);
    // Fallback resolution is symmetric and repeatable.
    assert_eq!(resolver.resolve(m0, m1), resolver.resolve(m1, m0));
    // An explicit registration overrides the fallback for that unordered pair.
    let explicit = resolver.resolve(AudioMaterialId(1), AudioMaterialId(1));
    resolver.register(m0, m1, explicit);
    assert_eq!(resolver.resolve(m0, m1), explicit);
    assert_eq!(resolver.resolve(m1, m0), explicit);
    assert_eq!(resolver.registered_len(), 1);

    // Roughness derived from a distinguishing pair is deterministic and bounded.
    let pair = MaterialResolver::new().resolve(AudioMaterialId(3), AudioMaterialId(7));
    let r0 = surface_roughness(pair, 0.5);
    let r1 = surface_roughness(pair, 0.5);
    assert!(fabs(r0 - r1) < 1e-6);
    assert!((0.0..=1.0).contains(&r0));
}
