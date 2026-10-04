//! Full-chain integration tests for the `prism_audio_procedural` crate.
//!
//! These tests drive the public surface end to end: physics contact facts flow
//! through the [`ContactEventBus`] (merge, ordering, budget eviction), into a
//! configured [`ContactVoice`] whose modal/friction output is checked for
//! finiteness, bounded level, determinism, compressive loudness, and
//! silence-in/silence-out behaviour. They also resolve a [`MaterialLibrary`]
//! profile for every material-category pair, exercise the [`GranularEngine`]
//! scatter cloud, and gate the [`SoundscapeScheduler`] by radius, concurrency
//! cap, spatial filter, and indoor/outdoor permission. Signals avoid
//! transcendental math: amplitudes are compared with a hand-rolled absolute
//! value and energies with sum-of-squares, so the tests themselves introduce no
//! floating-point library dependence.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, MPEG, Google Resonance Audio, Web Audio, or Project Acoustics source
//! or derived code; no AI/ML. Public standards contribute ideas only.
//!
//! # Relationship
//! Exercises design section 47 (contact/modal/granular physics-coupled audio)
//! and section 37 (procedural soundscape/ambience) through the crate's public
//! API, depending only on the crate under test and the standard library made
//! available by its `std` feature.

use prism_audio_procedural::soundscape::NullSpatialFilter;
use prism_audio_procedural::{
    ContactEvent, ContactEventBus, ContactId, ContactPhase, ContactPoint, ContactVoice,
    GrainParams, GranularEngine, ImpactEvent, MaterialCategory, MaterialId, MaterialLibrary,
    MaterialPairId, MergeConfig, SoundscapeElement, SoundscapePalette, SoundscapeScheduler,
    SoundscapeState, SpatialFilter, SustainEvent,
};

/// Branchless absolute value: avoids the banned `f32::abs` std float method.
fn fabs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Peak absolute sample of a buffer.
fn peak(buffer: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for &s in buffer {
        let a = fabs(s);
        if a > m {
            m = a;
        }
    }
    m
}

/// Sum-of-squares energy of a buffer.
fn energy(buffer: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &s in buffer {
        sum += s * s;
    }
    sum
}

/// Returns `true` when every sample is a finite number.
fn all_finite(buffer: &[f32]) -> bool {
    for &s in buffer {
        if !s.is_finite() {
            return false;
        }
    }
    true
}

/// A spatial filter that rejects every placement, used to prove the scheduler
/// honours a gating filter.
struct RejectAll;

impl SpatialFilter for RejectAll {
    fn accept(&self, _position: [f32; 3]) -> Option<f32> {
        None
    }
}

/// Renders a single configured strike into a fresh voice with the given seed.
fn render_impact_voice(seed: u64) -> Vec<f32> {
    let lib = MaterialLibrary::default();
    let profile = lib.profile(MaterialPairId::new(1, 2));
    let mut voice = ContactVoice::new(48_000, 24, 8, 512, seed);
    voice.configure(&profile);
    voice.trigger_impact_raw(5.0, 3.0, 0.5, ContactPoint::new(0.3), 32);
    let mut out = vec![0.0f32; 512];
    voice.render(512, &mut out);
    out
}

/// Renders a mono granular cloud at a given grain rate and seed.
fn render_cloud(rate: f32, seed: u64) -> Vec<f32> {
    let mut eng = GranularEngine::new(48_000, 64, seed);
    eng.set_params(GrainParams {
        grain_rate_hz: rate,
        base_freq_hz: 220.0,
        amplitude: 0.5,
        ..GrainParams::default()
    });
    let mut out = vec![0.0f32; 4096];
    eng.render_mono(4096, &mut out);
    out
}

/// Runs the soundscape scheduler over many blocks and collects every emitted
/// one-shot as `(element_id, position, gain)` tuples.
fn collect_shots(seed: u64) -> Vec<(u32, [f32; 3], f32)> {
    let radius = 3.0f32;
    let element = SoundscapeElement::new(1)
        .with_interval(0.05, 0.0)
        .with_trigger_prob(1.0)
        .with_scatter_radius(radius)
        .with_gain_range(0.5, 1.0)
        .with_concurrency_cap(1000);
    let palette = SoundscapePalette::new().with_element(element);
    let mut sched = SoundscapeScheduler::new(48_000, 4, seed);
    sched.set_max_concurrent(1000);
    sched.set_listener([5.0, 1.0, -2.0]);
    let state = SoundscapeState::outdoor_noon();
    let filter = NullSpatialFilter;
    let mut shots = Vec::new();
    for _ in 0..40 {
        let produced = sched.process(&palette, state, 4800, &filter);
        for s in produced {
            shots.push((s.element_id, s.position, s.gain));
        }
    }
    shots
}

#[test]
fn full_contact_chain_through_bus_produces_bounded_sound() {
    let lib = MaterialLibrary::default();
    let profile = lib.profile(MaterialPairId::new(1, 2));
    let mut voice = ContactVoice::new(48_000, 24, 8, 1024, 0x1234);
    voice.configure(&profile);

    let mut bus = ContactEventBus::new(8, 8, MergeConfig::default());
    bus.begin_block();
    let impact = ImpactEvent::new(
        ContactId(1),
        MaterialPairId::new(1, 2),
        6.0,
        4.0,
        0.5,
        ContactPoint::new(0.4),
        64,
        1024,
    );
    bus.ingest(ContactEvent::Impact(impact));
    let surviving = bus.finalize_block();
    assert_eq!(surviving, 1);

    for e in bus.impacts() {
        voice.trigger_impact(e);
    }
    let mut out = vec![0.0f32; 1024];
    voice.render(1024, &mut out);

    assert!(all_finite(&out), "chain output must be finite");
    let p = peak(&out);
    assert!(p > 1.0e-3, "a real strike should produce audible sound");
    assert!(p < 32.0, "chain output must stay bounded");
    assert!(voice.is_active(1.0e-4), "voice should still be ringing");
}

#[test]
fn contact_event_bus_finalize_merges_orders_and_budgets() {
    // Coincident same-contact impacts within the merge window sum their
    // impulse and keep the earliest offset; a distinct later contact survives
    // separately and the finalised list is ordered by sample offset.
    let mut bus = ContactEventBus::new(8, 8, MergeConfig::default());
    bus.begin_block();
    let pair = MaterialPairId::new(3, 4);
    bus.ingest(ContactEvent::Impact(ImpactEvent::new(
        ContactId(7),
        pair,
        2.0,
        1.0,
        0.0,
        ContactPoint::new(0.5),
        10,
        512,
    )));
    bus.ingest(ContactEvent::Impact(ImpactEvent::new(
        ContactId(7),
        pair,
        3.0,
        1.0,
        0.0,
        ContactPoint::new(0.5),
        20,
        512,
    )));
    bus.ingest(ContactEvent::Impact(ImpactEvent::new(
        ContactId(9),
        pair,
        1.5,
        1.0,
        0.0,
        ContactPoint::new(0.5),
        300,
        512,
    )));
    let surviving = bus.finalize_block();
    assert_eq!(surviving, 2);
    let impacts = bus.impacts();
    assert_eq!(impacts[0].sample_offset, 10);
    assert_eq!(impacts[1].sample_offset, 300);
    assert!(fabs(impacts[0].impulse - 5.0) < 1.0e-4);
    assert!(fabs(impacts[1].impulse - 1.5) < 1.0e-4);

    // Budget eviction: a full budget drops its weakest impact only when the
    // newcomer is louder; a weaker newcomer is discarded.
    let mut small = ContactEventBus::new(2, 4, MergeConfig::default());
    small.begin_block();
    let p = MaterialPairId::new(1, 1);
    small.ingest(ContactEvent::Impact(ImpactEvent::new(
        ContactId(1),
        p,
        1.0,
        1.0,
        0.0,
        ContactPoint::new(0.5),
        10,
        512,
    )));
    small.ingest(ContactEvent::Impact(ImpactEvent::new(
        ContactId(2),
        p,
        5.0,
        1.0,
        0.0,
        ContactPoint::new(0.5),
        20,
        512,
    )));
    small.ingest(ContactEvent::Impact(ImpactEvent::new(
        ContactId(3),
        p,
        3.0,
        1.0,
        0.0,
        ContactPoint::new(0.5),
        30,
        512,
    )));
    small.ingest(ContactEvent::Impact(ImpactEvent::new(
        ContactId(4),
        p,
        0.5,
        1.0,
        0.0,
        ContactPoint::new(0.5),
        40,
        512,
    )));
    let kept = small.finalize_block();
    assert_eq!(kept, 2);
    let mut has_five = false;
    let mut has_three = false;
    for e in small.impacts() {
        if fabs(e.impulse - 5.0) < 1.0e-4 {
            has_five = true;
        }
        if fabs(e.impulse - 3.0) < 1.0e-4 {
            has_three = true;
        }
    }
    assert!(has_five && has_three, "strongest two impacts must survive");
}

#[test]
fn contact_voice_render_is_deterministic() {
    let a = render_impact_voice(0xABCD);
    let b = render_impact_voice(0xABCD);
    assert_eq!(a, b, "same seed and input must yield bit-identical output");
    assert!(peak(&a) > 1.0e-3, "the deterministic strike must be audible");
}

#[test]
fn silent_inputs_produce_silence() {
    // A voice with nothing scheduled renders digital silence.
    let mut voice = ContactVoice::new(48_000, 16, 8, 256, 1);
    let mut out = vec![0.0f32; 256];
    voice.render(256, &mut out);
    for &s in &out {
        assert_eq!(s, 0.0);
    }

    // A granular engine with a zero grain rate emits silence.
    let mut eng = GranularEngine::new(48_000, 32, 2);
    eng.set_params(GrainParams {
        grain_rate_hz: 0.0,
        ..GrainParams::default()
    });
    let mut mono = vec![0.0f32; 256];
    eng.render_mono(256, &mut mono);
    for &s in &mono {
        assert_eq!(s, 0.0);
    }

    // A sustain with no motion and no pressure parks the voice as separated.
    let mut parked = ContactVoice::new(48_000, 16, 8, 256, 7);
    parked.update_contact(&SustainEvent::new(
        ContactId(5),
        MaterialPairId::new(1, 2),
        0.0,
        0.0,
        0.5,
    ));
    assert_eq!(parked.phase(), ContactPhase::Separated);
}

#[test]
fn harder_impacts_are_louder_but_compressive() {
    let lib = MaterialLibrary::default();
    let profile = lib.profile(MaterialPairId::new(1, 2));

    let mut soft = ContactVoice::new(48_000, 24, 8, 2048, 11);
    soft.configure(&profile);
    soft.trigger_impact_raw(1.0, 1.0, 0.0, ContactPoint::new(0.5), 0);
    let mut soft_out = vec![0.0f32; 2048];
    soft.render(2048, &mut soft_out);

    let mut hard = ContactVoice::new(48_000, 24, 8, 2048, 11);
    hard.configure(&profile);
    hard.trigger_impact_raw(20.0, 1.0, 0.0, ContactPoint::new(0.5), 0);
    let mut hard_out = vec![0.0f32; 2048];
    hard.render(2048, &mut hard_out);

    let soft_peak = peak(&soft_out);
    let hard_peak = peak(&hard_out);
    assert!(soft_peak > 1.0e-4, "the soft strike must be audible");
    assert!(hard_peak > soft_peak, "a harder impact must be louder");
    assert!(
        hard_peak < soft_peak * 10.0,
        "20x impulse must grow amplitude sub-linearly (compressive)"
    );
}

#[test]
fn material_library_resolves_sound_for_every_category_pair() {
    let categories = [
        MaterialCategory::Metal,
        MaterialCategory::Wood,
        MaterialCategory::Stone,
        MaterialCategory::Glass,
        MaterialCategory::Plastic,
        MaterialCategory::Ceramic,
        MaterialCategory::Fabric,
        MaterialCategory::Liquid,
        MaterialCategory::Generic,
    ];
    assert_eq!(categories.len(), MaterialCategory::COUNT);

    let mut next_id: u16 = 100;
    for &ca in &categories {
        for &cb in &categories {
            let mut lib = MaterialLibrary::new(8);
            let id_a = MaterialId(next_id);
            let id_b = MaterialId(next_id + 1);
            next_id += 2;
            lib.register_material(id_a, ca);
            lib.register_material(id_b, cb);
            let profile = lib.profile(MaterialPairId::from_ids(id_a, id_b));
            assert!(!profile.modes.is_empty(), "every pair needs a modal table");
            for m in &profile.modes {
                assert!(m.freq_hz.is_finite() && m.freq_hz > 0.0);
                assert!(m.half_life_s.is_finite() && m.half_life_s > 0.0);
                assert!(m.gain.is_finite());
            }
            assert!(profile.fundamental_hz.is_finite() && profile.fundamental_hz > 0.0);
            assert!(profile.friction.base_gain.is_finite());
            assert!(profile.friction.centroid_hz.is_finite());
        }
    }
}

#[test]
fn granular_cloud_is_deterministic_and_rate_scales_energy() {
    let a = render_cloud(400.0, 99);
    let b = render_cloud(400.0, 99);
    assert_eq!(a, b, "same seed, params, and length must be reproducible");
    assert!(all_finite(&a), "granular output must be finite");

    let dense = energy(&render_cloud(400.0, 42));
    let sparse = energy(&render_cloud(60.0, 42));
    assert!(sparse > 0.0, "a non-zero grain rate must emit energy");
    assert!(dense > sparse, "a higher grain rate must carry more energy");
}

#[test]
fn soundscape_scatter_is_deterministic_and_bounded_in_radius() {
    let a = collect_shots(0x2024);
    let b = collect_shots(0x2024);
    assert_eq!(a, b, "same seed and sequence must replay identically");
    assert!(!a.is_empty(), "scheduler should fire one-shots");

    let listener = [5.0f32, 1.0, -2.0];
    let radius = 3.0f32;
    for (_id, pos, _gain) in &a {
        let dx = pos[0] - listener[0];
        let dz = pos[2] - listener[2];
        assert!(
            dx * dx + dz * dz <= radius * radius + 1.0e-3,
            "placement must land inside the scatter disc"
        );
        assert_eq!(pos[1], listener[1], "y is unchanged by horizontal scatter");
    }
}

#[test]
fn soundscape_gates_output_via_cap_and_spatial_filter() {
    // A rejecting spatial filter gates every placement.
    let gated_element = SoundscapeElement::new(2)
        .with_interval(0.05, 0.0)
        .with_trigger_prob(1.0)
        .with_scatter_radius(2.0)
        .with_concurrency_cap(1000);
    let gated_palette = SoundscapePalette::new().with_element(gated_element);
    let mut gated = SoundscapeScheduler::new(48_000, 4, 1);
    gated.set_max_concurrent(1000);
    let noon = SoundscapeState::outdoor_noon();
    let reject = RejectAll;
    let mut rejected_total = 0usize;
    for _ in 0..20 {
        rejected_total += gated.process(&gated_palette, noon, 4800, &reject).len();
    }
    assert_eq!(rejected_total, 0, "a rejecting filter must gate every shot");

    // The global concurrency cap bounds the active estimate.
    let capped_element = SoundscapeElement::new(3)
        .with_interval(0.02, 0.0)
        .with_trigger_prob(1.0)
        .with_scatter_radius(2.0)
        .with_concurrency_cap(1000);
    let capped_palette = SoundscapePalette::new().with_element(capped_element);
    let mut capped = SoundscapeScheduler::new(48_000, 4, 3);
    capped.set_max_concurrent(5);
    let null = NullSpatialFilter;
    for _ in 0..60 {
        capped.process(&capped_palette, noon, 4800, &null);
        assert!(
            capped.active_estimate() <= 5.0 + 1.0e-3,
            "active estimate must stay within the concurrency cap"
        );
    }

    // An outdoor-only element stays silent indoors.
    let outdoor_element = SoundscapeElement::new(4)
        .with_interval(0.05, 0.0)
        .with_trigger_prob(1.0)
        .with_environments(false, true);
    let outdoor_palette = SoundscapePalette::new().with_element(outdoor_element);
    let mut indoor_sched = SoundscapeScheduler::new(48_000, 4, 5);
    indoor_sched.set_max_concurrent(1000);
    let indoors = SoundscapeState::new(0.5, 1.0, true);
    let mut indoor_total = 0usize;
    for _ in 0..20 {
        indoor_total += indoor_sched
            .process(&outdoor_palette, indoors, 4800, &null)
            .len();
    }
    assert_eq!(indoor_total, 0, "an outdoor-only element must be silent indoors");
}
