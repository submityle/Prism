//! End-to-end integration coverage for the HRTF binaural rendering chain:
//! dataset construction, azimuth/elevation interpolation with inter-aural
//! time delay, block convolution rendering, head-tracked local-angle
//! resolution, near-field inter-aural level difference, and transaural
//! crosstalk cancellation.
//!
//! The tests build a small synthetic HRIR grid whose per-ear onset delays and
//! peak gains encode source laterality, then exercise every public stage of
//! the crate as a connected pipeline (not as isolated unit assertions).
//!
//! # Provenance
//! Original work authored for Prism. It contains no source code or derived
//! code from Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Dolby,
//! MPEG, Google Resonance, Web Audio, or Microsoft Project Acoustics, and uses
//! no artificial-intelligence or machine-learning techniques. Only the ideas
//! of publicly documented acoustic conventions (threshold-based onset delay,
//! inverse-distance gain, inter-aural level difference) are relied upon.
//!
//! # Relationship
//! Exercises the HRTF binaural chapter of
//! `docs/prism_audio_engine_design_zh.md`. Depends only on the
//! `prism_audio_hrtf` crate's public surface; direction vectors and head
//! orientations are obtained from crate helpers rather than any external math
//! crate, so the test needs no non-dev dependency.

use core::f32::consts::PI;

use prism_audio_hrtf::{
    direction_from_angles, interpolate, local_azimuth, resolve, world_to_local_direction,
    BinauralRenderer, CrosstalkCanceller, CrosstalkParams, DatasetError, HeadGeometry, HeadPose,
    HeadTracker, HrtfDataset, Measurement, NearFieldParams, DEFAULT_HEAD_RADIUS, MAX_NEIGHBORS,
};

const SAMPLE_RATE: u32 = 48_000;
const HRIR_LEN: usize = 16;
const EPS: f32 = 1.0e-4;

/// Branchless magnitude helper so the test avoids `f32::abs` (the workspace
/// lints forbid std float math in these integration tests).
fn fabs(x: f32) -> f32 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Writes a short decaying impulse into `buf` starting at `onset`.
fn write_impulse(buf: &mut [f32], onset: usize, peak: f32) {
    buf[onset] = peak;
    buf[onset + 1] = peak * 0.5;
    buf[onset + 2] = peak * 0.25;
}

/// Builds a synthetic left/right HRIR for one measurement. The near ear has a
/// shorter onset delay and a higher peak; the far (shadowed) ear is delayed by
/// `itd` samples and attenuated.
fn synth_pair(azimuth: f32, itd: usize) -> ([f32; HRIR_LEN], [f32; HRIR_LEN]) {
    let mut left = [0.0f32; HRIR_LEN];
    let mut right = [0.0f32; HRIR_LEN];
    let onset_near = 2usize;
    let onset_far = 2usize + itd;
    let peak_near = 0.8f32;
    let peak_far = 0.5f32;
    if azimuth < 0.0 {
        write_impulse(&mut left, onset_near, peak_near);
        write_impulse(&mut right, onset_far, peak_far);
    } else if azimuth > 0.0 {
        write_impulse(&mut left, onset_far, peak_far);
        write_impulse(&mut right, onset_near, peak_near);
    } else {
        write_impulse(&mut left, onset_near, 0.65);
        write_impulse(&mut right, onset_near, 0.65);
    }
    (left, right)
}

/// Assembles the five-measurement horizontal-plane dataset shared by the tests.
fn build_dataset() -> HrtfDataset {
    let azimuths: [f32; 5] = [-PI / 2.0, -PI / 4.0, 0.0, PI / 4.0, PI / 2.0];
    let itds: [usize; 5] = [4, 2, 0, 2, 4];
    let mut measurements = Vec::new();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (&azimuth, &itd) in azimuths.iter().zip(itds.iter()) {
        measurements.push(Measurement::new(azimuth, 0.0, 1.0));
        let (l, r) = synth_pair(azimuth, itd);
        left.extend_from_slice(&l);
        right.extend_from_slice(&r);
    }
    HrtfDataset::from_samples(SAMPLE_RATE, HRIR_LEN, measurements, left, right)
        .expect("synthetic dataset is well formed")
}

/// Sum of squared samples (energy proxy).
fn energy(buf: &[f32]) -> f32 {
    buf.iter().map(|&v| v * v).sum()
}

#[test]
fn dataset_exposes_measurement_grid() {
    let ds = build_dataset();
    assert_eq!(ds.len(), 5);
    assert!(!ds.is_empty());
    assert_eq!(ds.hrir_len(), HRIR_LEN);
    assert_eq!(ds.sample_rate(), SAMPLE_RATE);
    assert_eq!(ds.measurements().len(), 5);
    assert!(fabs(ds.measurement(2).expect("centre measurement").azimuth) < EPS);
    assert_eq!(ds.left_hrir(0).len(), HRIR_LEN);
    assert_eq!(ds.right_hrir(4).len(), HRIR_LEN);
    assert!(ds.measurement(99).is_none());
}

#[test]
fn interpolate_exact_hit_snaps_to_single_measurement() {
    let ds = build_dataset();
    let mut out_l = [0.0f32; HRIR_LEN];
    let mut out_r = [0.0f32; HRIR_LEN];
    let info = interpolate(&ds, -PI / 2.0, 0.0, &mut out_l, &mut out_r)
        .expect("non-empty dataset interpolates");
    assert_eq!(info.neighbors_used, 1);
    assert!(energy(&out_l) > 0.0);
    assert!(energy(&out_r) > 0.0);
}

#[test]
fn interpolate_between_measurements_blends_neighbors() {
    let ds = build_dataset();
    let mut out_l = [0.0f32; HRIR_LEN];
    let mut out_r = [0.0f32; HRIR_LEN];
    // A direction between the -45 degree and 0 degree measurements.
    let info = interpolate(&ds, -PI / 8.0, 0.0, &mut out_l, &mut out_r)
        .expect("non-empty dataset interpolates");
    assert!(info.neighbors_used >= 2);
    assert!(info.neighbors_used <= MAX_NEIGHBORS);
    assert!(energy(&out_l) > 0.0);
    assert!(energy(&out_r) > 0.0);
}

#[test]
fn interpolate_delays_follow_source_laterality() {
    let ds = build_dataset();
    let mut out_l = [0.0f32; HRIR_LEN];
    let mut out_r = [0.0f32; HRIR_LEN];

    // Far-left source: the left (near) ear should arrive no later than right.
    let left_info = interpolate(&ds, -PI / 2.0, 0.0, &mut out_l, &mut out_r)
        .expect("left interpolation");
    assert!(left_info.left_delay <= left_info.right_delay);

    // Far-right source: the right (near) ear should arrive no later than left.
    let right_info = interpolate(&ds, PI / 2.0, 0.0, &mut out_l, &mut out_r)
        .expect("right interpolation");
    assert!(right_info.right_delay <= right_info.left_delay);
}

#[test]
fn binaural_left_source_is_louder_in_left_ear() {
    let ds = build_dataset();
    let mut renderer = BinauralRenderer::new(HRIR_LEN, 64);
    renderer.set_hrir_immediate(ds.left_hrir(0), ds.right_hrir(0));

    let mut input = [0.0f32; 64];
    input[0] = 1.0;
    let mut out_l = [0.0f32; 64];
    let mut out_r = [0.0f32; 64];
    let frames = renderer.process_block(&input, &mut out_l, &mut out_r);
    assert_eq!(frames, 64);
    assert!(energy(&out_l) > energy(&out_r));
}

#[test]
fn binaural_two_instances_match_bit_for_bit() {
    let ds = build_dataset();
    let mut input = [0.0f32; 48];
    input[0] = 1.0;
    input[5] = -0.5;
    input[9] = 0.25;

    let mut a = BinauralRenderer::new(HRIR_LEN, 64);
    let mut b = BinauralRenderer::new(HRIR_LEN, 64);
    a.set_hrir_immediate(ds.left_hrir(3), ds.right_hrir(3));
    b.set_hrir_immediate(ds.left_hrir(3), ds.right_hrir(3));

    let mut a_l = vec![0.0f32; 48];
    let mut a_r = vec![0.0f32; 48];
    let mut b_l = vec![0.0f32; 48];
    let mut b_r = vec![0.0f32; 48];
    a.process_block(&input, &mut a_l, &mut a_r);
    b.process_block(&input, &mut b_l, &mut b_r);
    assert_eq!(a_l, b_l);
    assert_eq!(a_r, b_r);
}

#[test]
fn binaural_reset_restores_initial_response() {
    let ds = build_dataset();
    let mut input = [0.0f32; 48];
    input[0] = 1.0;
    input[3] = 0.4;

    let mut renderer = BinauralRenderer::new(HRIR_LEN, 64);
    renderer.set_hrir_immediate(ds.left_hrir(1), ds.right_hrir(1));

    let mut first_l = vec![0.0f32; 48];
    let mut first_r = vec![0.0f32; 48];
    renderer.process_block(&input, &mut first_l, &mut first_r);

    renderer.reset();
    let mut again_l = vec![0.0f32; 48];
    let mut again_r = vec![0.0f32; 48];
    renderer.process_block(&input, &mut again_l, &mut again_r);

    assert_eq!(first_l, again_l);
    assert_eq!(first_r, again_r);
}

#[test]
fn binaural_crossfade_completes_within_configured_length() {
    let ds = build_dataset();
    let mut renderer = BinauralRenderer::new(HRIR_LEN, 64);
    renderer.set_hrir_immediate(ds.left_hrir(0), ds.right_hrir(0));
    assert!(!renderer.is_crossfading());

    renderer.set_crossfade_len(4);
    renderer.set_hrir(ds.left_hrir(4), ds.right_hrir(4));
    assert!(renderer.is_crossfading());

    let mut input = [0.0f32; 64];
    input[0] = 1.0;
    let mut out_l = [0.0f32; 64];
    let mut out_r = [0.0f32; 64];
    renderer.process_block(&input, &mut out_l, &mut out_r);
    assert!(!renderer.is_crossfading());
}

#[test]
fn nearfield_ild_grows_as_source_approaches() {
    let dir = direction_from_angles(PI / 2.0, 0.0);
    let head = HeadGeometry::new(DEFAULT_HEAD_RADIUS);
    let params = NearFieldParams::default();

    let far = resolve(dir, 2.0, head, params);
    let near = resolve(dir, 0.25, head, params);

    // A source on the right is louder in the right ear (positive ILD).
    assert!(near.ild_db() > 0.0);
    // The parallax grows the inter-aural level difference as it gets closer.
    assert!(fabs(near.ild_db()) > fabs(far.ild_db()));
}

#[test]
fn transaural_canceller_runs_and_is_deterministic() {
    let params = CrosstalkParams::new(0.7, 8, 8_000.0, 48_000.0);
    assert!(fabs(params.contralateral_gain - 0.7) < EPS);

    let mut in_l = [0.0f32; 32];
    let mut in_r = [0.0f32; 32];
    let mut state = 1u32;
    for n in 0..32 {
        // Deterministic linear-congruential pseudo-noise in [-1, 1).
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (state >> 9) as f32 / 4_194_304.0 - 1.0;
        in_l[n] = noise;
        in_r[n] = noise * 0.5;
    }

    let mut c1 = CrosstalkCanceller::new(&params);
    assert_eq!(c1.delay_samples(), 8);
    assert!(fabs(c1.contralateral_gain() - 0.7) < EPS);

    let mut l1 = [0.0f32; 32];
    let mut r1 = [0.0f32; 32];
    let frames = c1.process_block(&in_l, &in_r, &mut l1, &mut r1);
    assert_eq!(frames, 32);

    let mut c2 = CrosstalkCanceller::new(&params);
    let mut l2 = [0.0f32; 32];
    let mut r2 = [0.0f32; 32];
    c2.process_block(&in_l, &in_r, &mut l2, &mut r2);

    assert_eq!(l1, l2);
    assert_eq!(r1, r2);

    // Resetting restores the initial state: the same input reproduces output.
    c1.reset();
    let mut l3 = [0.0f32; 32];
    let mut r3 = [0.0f32; 32];
    c1.process_block(&in_l, &in_r, &mut l3, &mut r3);
    assert_eq!(l1, l3);
    assert_eq!(r1, r3);
}

#[test]
fn head_tracker_identity_preserves_world_angles() {
    let identity = HeadPose::default().orientation;
    let tracker = HeadTracker::new(identity, 0.0, 0.0);

    let world_dir = direction_from_angles(PI / 4.0, 0.0);
    let local = tracker.local_direction(world_dir);
    assert!(fabs(local_azimuth(local) - PI / 4.0) < 1.0e-3);
    assert!(fabs(tracker.local_angles(world_dir).azimuth - PI / 4.0) < 1.0e-3);

    let passthrough = world_to_local_direction(identity, world_dir);
    assert!(fabs(passthrough.x - world_dir.x) < EPS);
    assert!(fabs(passthrough.y - world_dir.y) < EPS);
    assert!(fabs(passthrough.z - world_dir.z) < EPS);
}

#[test]
fn head_rotation_shifts_local_azimuth_monotonically() {
    let identity = HeadPose::default().orientation;
    let axis = direction_from_angles(0.0, PI / 2.0); // unit +Y (yaw axis)
    let pose = HeadPose::new(identity, axis * 20.0);

    let small = pose.predict(0.02); // ~0.4 rad yaw
    let large = pose.predict(0.1); //  ~2.0 rad yaw (prediction clamp)

    let forward = direction_from_angles(0.0, 0.0);
    let shift_small = fabs(local_azimuth(world_to_local_direction(small, forward)));
    let shift_large = fabs(local_azimuth(world_to_local_direction(large, forward)));

    assert!(shift_small > 1.0e-2);
    assert!(shift_large > shift_small + 0.5);
}

#[test]
fn dataset_rejects_malformed_inputs() {
    let good = vec![Measurement::new(0.0, 0.0, 1.0)];
    let left = vec![0.0f32; HRIR_LEN];
    let right = vec![0.0f32; HRIR_LEN];

    let bad_left = HrtfDataset::from_samples(
        SAMPLE_RATE,
        HRIR_LEN,
        good.clone(),
        vec![0.0f32; HRIR_LEN - 1],
        right.clone(),
    );
    assert!(matches!(
        bad_left,
        Err(DatasetError::LeftLengthMismatch { .. })
    ));

    let bad_right = HrtfDataset::from_samples(
        SAMPLE_RATE,
        HRIR_LEN,
        good.clone(),
        left.clone(),
        vec![0.0f32; HRIR_LEN + 1],
    );
    assert!(matches!(
        bad_right,
        Err(DatasetError::RightLengthMismatch { .. })
    ));

    let no_meas =
        HrtfDataset::from_samples(SAMPLE_RATE, HRIR_LEN, Vec::new(), Vec::new(), Vec::new());
    assert_eq!(no_meas.unwrap_err(), DatasetError::NoMeasurements);

    let empty_hrir =
        HrtfDataset::from_samples(SAMPLE_RATE, 0, good.clone(), Vec::new(), Vec::new());
    assert_eq!(empty_hrir.unwrap_err(), DatasetError::EmptyHrir);

    let zero_rate = HrtfDataset::from_samples(0, HRIR_LEN, good, left, right);
    assert_eq!(zero_rate.unwrap_err(), DatasetError::ZeroSampleRate);
}
