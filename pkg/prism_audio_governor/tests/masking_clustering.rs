//! End-to-end integration coverage for the section 33 psychoacoustic
//! virtualization and source-clustering chain of `prism_audio_governor`.
//!
//! This test wires the real public surface of the `masking` and `clustering`
//! modules together the way the governor does at runtime: critical-band
//! partition -> per-voice spectra -> masking analysis -> HDR-window gating ->
//! combined virtualization verdicts, then spatial/timbral clustering ->
//! capacity merge -> per-membership fade ramps. It asserts the cross-module
//! invariants that the per-module unit tests cannot, exercising the chain the
//! way `importance`/`lod` consume it.
//!
//! # Provenance
//! Original work authored for Prism. Contains no Unreal Engine, Unity, Godot,
//! Wwise, FMOD, Steam Audio, Dolby, MPEG, Google Resonance Audio, Web Audio, or
//! Microsoft Project Acoustics source or derived code. No AI/ML is used; only
//! classical psychoacoustic DSP ideas from public standards inform the design.
//!
//! # Relationship
//! Validates section 33 (masking-aware virtualization + source clustering) of
//! the Prism audio engine design, over `prism_audio_governor::{masking,
//! clustering}`. Depends only on the crate's public API plus `bevy_math::Vec3`.

use bevy_math::Vec3;
use prism_audio_governor::clustering::assignment::{assign, merge_to_capacity, ClusterConfig};
use prism_audio_governor::clustering::cluster::{Cluster, ClusterMember};
use prism_audio_governor::clustering::timbre::Timbre;
use prism_audio_governor::clustering::fade::MembershipFades;
use prism_audio_governor::masking::critical_bands::CriticalBands;
use prism_audio_governor::masking::hdr_gate::{gate_voices, HdrWindowEdge};
use prism_audio_governor::masking::masking_model::{MaskingAnalyzer, MaskingModel, VoiceSpectrum};
use prism_audio_governor::masking::virtualization::{
    combine, force_virtualize_set, virtualized_count,
};
use prism_audio_core::math::Sample;

/// Branch-free absolute value, avoiding `Sample::abs` (std float math is
/// disallowed in these integration tests by the crate lint policy).
fn fabs(x: Sample) -> Sample {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

fn approx(a: Sample, b: Sample, tol: Sample) -> bool {
    fabs(a - b) <= tol
}

/// A strong tonal masker near 1 kHz plus a weak neighbour in an adjacent band
/// should flag the neighbour as masked, while a loud, well-separated voice must
/// survive the masking pass.
#[test]
fn masking_hides_quiet_neighbour_but_spares_separated_loud_voice() {
    let bands = CriticalBands::bark_default();
    let analyzer = MaskingAnalyzer::new(MaskingModel::default(), bands.band_count());

    // Voice 0: strong masker at 1 kHz.
    let masker = VoiceSpectrum::tonal(&bands, 1_000.0, 4.0);
    // Voice 1: weak neighbour just above the masker (adjacent critical band).
    let neighbour = VoiceSpectrum::tonal(&bands, 1_120.0, 0.02);
    // Voice 2: loud, far-separated tone at 8 kHz.
    let separated = VoiceSpectrum::tonal(&bands, 8_000.0, 4.0);

    let voices = [masker, neighbour, separated];
    let masked = analyzer.analyze(&voices, MaskingModel::default().spread_up); // margin ~0.5

    assert_eq!(masked.len(), 3, "one verdict per voice");
    assert!(!masked[0], "the dominant masker is never masked by weaker voices");
    assert!(masked[1], "the quiet neighbour hides under the 1 kHz masker");
    assert!(!masked[2], "a loud, spectrally separated voice survives masking");
}

/// `gate_voices` must virtualize everything below the HDR window's dynamic
/// floor and keep everything at or above it.
#[test]
fn hdr_gate_virtualizes_below_window_floor() {
    // Reference (loudest) at -6 dB, 24 dB window -> lower edge at -30 dB.
    let edge = HdrWindowEdge::new(-6.0, 24.0);
    assert!(approx(edge.lower_edge_db(), -30.0, 1e-4), "window floor derivation");

    let loudness_db = [-6.0, -28.0, -31.0, -60.0];
    let gated = gate_voices(&loudness_db, edge, 0.0);

    assert_eq!(gated, vec![false, false, true, true]);

    // Raising the floor pulls more quiet voices under the gate.
    let gated_raised = gate_voices(&loudness_db, edge, 4.0); // floor -> -26 dB
    assert_eq!(gated_raised, vec![false, true, true, true]);
}

/// The masking and HDR verdicts must combine into a single force-virtualize set
/// that is the elementwise union, with a matching virtualized count.
#[test]
fn combined_verdicts_union_masking_and_hdr() {
    let masked = [true, false, false, false];
    let hdr = [false, false, true, false];

    let verdicts = combine(&masked, &hdr);
    assert_eq!(verdicts.len(), 4);
    // Voices 0 (masked) and 2 (hdr-gated) virtualize; 1 and 3 stay physical.
    assert_eq!(virtualized_count(&verdicts), 2);

    let force = force_virtualize_set(&masked, &hdr);
    assert_eq!(force, vec![true, false, true, false]);

    // The force set agrees with the per-voice verdict helper.
    for (i, v) in verdicts.iter().enumerate() {
        assert_eq!(v.should_virtualize(), force[i], "verdict {i} matches union");
    }
}

/// Full chain: spectra -> masking -> HDR -> clustering of the still-physical
/// voices, respecting the cluster budget and member->cluster mapping.
#[test]
fn full_chain_clusters_surviving_voices_within_budget() {
    let bands = CriticalBands::bark_default();
    let analyzer = MaskingAnalyzer::new(MaskingModel::default(), bands.band_count());

    // Five voices: a tight pair of similar tones, a lone bright tone, plus two
    // very quiet neighbours that masking/HDR should drop before clustering.
    let spectra = [
        VoiceSpectrum::tonal(&bands, 500.0, 3.0),  // 0: pair member A
        VoiceSpectrum::tonal(&bands, 500.0, 3.0),  // 1: pair member B (co-located, same timbre)
        VoiceSpectrum::tonal(&bands, 9_000.0, 3.0), // 2: lone bright tone
        VoiceSpectrum::tonal(&bands, 540.0, 0.01), // 3: quiet neighbour -> masked
        VoiceSpectrum::tonal(&bands, 505.0, 0.01), // 4: quiet neighbour -> masked
    ];

    let masked = analyzer.analyze(&spectra, 0.5);
    // Dynamic range: the quiet neighbours sit far below the window floor.
    let loudness_db = [-6.0, -6.5, -7.0, -48.0, -50.0];
    let hdr = gate_voices(&loudness_db, HdrWindowEdge::new(-6.0, 24.0), 0.0);
    let force = force_virtualize_set(&masked, &hdr);

    assert!(force[3], "quiet neighbour 3 is virtualized before clustering");
    assert!(force[4], "quiet neighbour 4 is virtualized before clustering");

    // Build clustering members only for the voices that survive virtualization.
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),   // within spatial_radius of voice 0
        Vec3::new(40.0, 0.0, 0.0),  // far away
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(0.2, 0.0, 0.0),
    ];
    let mut members = Vec::new();
    for (i, spec) in spectra.iter().enumerate() {
        if force[i] {
            continue;
        }
        let timbre = Timbre::from_spectrum(spec, &bands);
        members.push(ClusterMember::new(i, positions[i], spec.total_energy(), timbre));
    }
    assert_eq!(members.len(), 3, "two voices virtualized, three remain physical");

    let config = ClusterConfig {
        max_clusters: 8,
        spatial_radius: 5.0,
        timbre_threshold: 0.35,
    };
    let set = assign(&members, &config);

    assert!(!set.is_empty());
    assert!(set.len() <= config.max_clusters, "cluster count honours the budget");
    assert_eq!(set.of_member.len(), members.len());
    for &c in &set.of_member {
        assert!(c < set.len(), "every member maps into an existing cluster");
    }

    // The two close, similar-timbre voices (input ids 0 and 1) share a cluster;
    // the far bright tone (id 2) sits in its own.
    let idx0 = members.iter().position(|m| m.voice == 0).expect("voice 0 present");
    let idx1 = members.iter().position(|m| m.voice == 1).expect("voice 1 present");
    let idx2 = members.iter().position(|m| m.voice == 2).expect("voice 2 present");
    assert_eq!(
        set.of_member[idx0], set.of_member[idx1],
        "the tight similar pair collapses into one representative"
    );
    assert_ne!(
        set.of_member[idx0], set.of_member[idx2],
        "the far, bright tone stays a distinct cluster"
    );
}

/// A cluster budget smaller than the natural grouping must force a merge down
/// to capacity while conserving total member coverage.
#[test]
fn capacity_merge_collapses_to_budget() {
    let bands = CriticalBands::bark_default();
    // Three spatially separated groups that would form three clusters.
    let members = [
        ClusterMember::new(0, Vec3::new(0.0, 0.0, 0.0), 2.0, Timbre::neutral()),
        ClusterMember::new(1, Vec3::new(50.0, 0.0, 0.0), 2.0, Timbre::neutral()),
        ClusterMember::new(2, Vec3::new(0.0, 50.0, 0.0), 2.0, Timbre::neutral()),
    ];
    let _ = &bands;

    let config = ClusterConfig {
        max_clusters: 8,
        spatial_radius: 2.0,
        timbre_threshold: 0.1,
    };
    let set = assign(&members, &config);
    assert_eq!(set.len(), 3, "three isolated voices seed three clusters");

    let merged = merge_to_capacity(set.clusters.clone(), 2);
    assert_eq!(merged.len(), 2, "capacity merge collapses to the budget");

    // Total membership is conserved across the merge.
    let before: usize = set.clusters.iter().map(Cluster::len).sum();
    let after: usize = merged.iter().map(Cluster::len).sum();
    assert_eq!(before, after, "no member is lost during the capacity merge");
}

/// Membership fades must ramp a newly active voice up and a dropped voice down
/// smoothly over the configured block count, not jump instantly.
#[test]
fn membership_fades_ramp_in_and_out() {
    let mut fades = MembershipFades::new(4);

    // Block 1: voice 7 becomes active -> gain starts rising from zero.
    fades.advance_block(&[7]);
    let g1 = fades.gain(7);
    assert!(g1 > 0.0 && g1 < 1.0, "newly active voice ramps up, not instant: {g1}");

    // Keep it active for several blocks; it should reach full gain.
    for _ in 0..6 {
        fades.advance_block(&[7]);
    }
    assert!(approx(fades.gain(7), 1.0, 1e-3), "sustained voice reaches unity");

    // Drop it: gain must fall back toward zero over subsequent blocks.
    fades.advance_block(&[]);
    let gd = fades.gain(7);
    assert!(gd < 1.0, "dropped voice begins fading out: {gd}");
    for _ in 0..6 {
        fades.advance_block(&[]);
    }
    assert!(approx(fades.gain(7), 0.0, 1e-3), "fully faded-out voice rests at zero");
}

/// Timbre similarity/distance must be self-consistent: a voice is maximally
/// similar to itself, and spectrally distinct voices are farther apart than
/// near-identical ones.
#[test]
fn timbre_distance_orders_spectral_similarity() {
    let bands = CriticalBands::bark_default();
    let low = Timbre::from_spectrum(&VoiceSpectrum::tonal(&bands, 300.0, 1.0), &bands);
    let low2 = Timbre::from_spectrum(&VoiceSpectrum::tonal(&bands, 320.0, 1.0), &bands);
    let high = Timbre::from_spectrum(&VoiceSpectrum::tonal(&bands, 10_000.0, 1.0), &bands);

    assert!(approx(low.distance(&low), 0.0, 1e-6), "a timbre has zero self-distance");
    assert!(approx(low.similarity(&low), 1.0, 1e-6), "a timbre is maximally self-similar");

    let near = low.distance(&low2);
    let far = low.distance(&high);
    assert!(far > near, "a bright tone is farther in timbre than a close low tone: {far} vs {near}");
    assert!(
        low.similarity(&high) < low.similarity(&low2),
        "similarity ranks the near pair above the distant pair"
    );
}
