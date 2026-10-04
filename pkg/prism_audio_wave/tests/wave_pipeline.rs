//! End-to-end integration tests for the wave-acoustics bake-to-runtime
//! pipeline.
//!
//! These wire the whole §43 chain together exactly as a tool and an engine
//! frame loop would: a voxelised [`VoxelScene`] is baked by [`bake_field`]
//! (which internally drives the ARD [`WaveSolver`], the perceptual encoder,
//! and the quantising field builder), the resulting [`ParameterField`] is
//! serialised and read back, a runtime [`ParameterLookup`] trilinearly
//! samples it on the audio thread, a [`WaveParamSmoother`] damps the per-block
//! jumps, a [`WaveBackend`] packages the field behind the shared parameter
//! source, and finally [`route`] crossfades the wave contribution with a
//! geometric one on the shared [`SpatialParams`] bus under a governor-selected
//! [`PropagationTier`].
//!
//! The per-module unit tests cover each stage in isolation. What is only
//! observable end to end -- and therefore only covered here -- is the full
//! pipeline's emergent behaviour: that a solid wall actually bakes into a more
//! occluded field than open air, that the bake is bit-reproducible, that the
//! packed field round-trips, that the runtime lookup and backend agree, and
//! that tier routing preserves the geometric kinematics the wave field cannot
//! carry.
//!
//! # Provenance
//! Original work. Contains no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, Microsoft Project Acoustics, Dolby, or Google Resonance Audio source
//! or derived code, and no AI/ML. The wave-acoustics bake-and-lookup idea is
//! drawn only as a concept from public descriptions of precomputed-acoustics
//! pipelines; the scenes, assertions, and the implementation under test are
//! independent.
//!
//! # Relationship
//! Covers §43 (wave acoustics and hybrid propagation) and its §32 handshake as
//! a black-box public-API contract over `prism_audio_wave`
//! (`solver` -> `encoding` -> `field` -> `bake` -> `lookup`/`backend` ->
//! `router`/`hybrid`), together with the shared `prism_audio_spatial` bus and
//! the `prism_audio_governor` propagation tier.

use bevy_math::{ops, Vec3};

use prism_audio_governor::governor::lod::PropagationTier;
use prism_audio_spatial::{SpatialParams, SpreadParams};
use prism_audio_wave::field::BitDepth;
use prism_audio_wave::{
    bake_field, route, BakeConfig, MultiStateOpening, ParameterField, ParameterLookup,
    PerceptualParams, ProbeGrid, SolveConfig, SourcePlacement, VoxelScene, WaveBackend,
    WaveParamSmoother, WaveParameterSource,
};
use prism_audio_wave::grid::Aabb;

/// Voxel edge length in metres for the test scenes.
const CELL: f32 = 0.5;
/// Cell counts: a long box so a mid-span wall clearly separates source and
/// listener regions.
const NX: u32 = 13;
const NY: u32 = 4;
const NZ: u32 = 4;
/// The `x` cell index of the dividing wall.
const WALL_X: u32 = 6;

fn approx(a: f32, b: f32, eps: f32) -> bool {
    ops::abs(a - b) <= eps
}

/// A compact solve config: a short tail is enough to capture the direct
/// arrival and early energy across a few metres, and keeps the bake fast.
fn solve_config() -> SolveConfig {
    SolveConfig {
        duration_s: 0.08,
        ..SolveConfig::default()
    }
}

fn bake_config() -> BakeConfig {
    BakeConfig {
        solve: solve_config(),
        bit_depth: BitDepth::Twelve,
        ..BakeConfig::default()
    }
}

/// An all-air box scene.
fn open_scene() -> VoxelScene {
    VoxelScene::new(Vec3::ZERO, CELL, NX, NY, NZ)
}

/// The same box with a full rigid wall sealing the `x = WALL_X` plane.
fn walled_scene() -> VoxelScene {
    let mut scene = open_scene();
    for iy in 0..NY {
        for iz in 0..NZ {
            scene.set_solid(WALL_X, iy, iz, true);
        }
    }
    scene
}

/// A probe grid spanning the scene, fine enough along `x` that several probes
/// sit behind the wall.
fn grid() -> ProbeGrid {
    let max = Vec3::new(NX as f32 * CELL, NY as f32 * CELL, NZ as f32 * CELL);
    ProbeGrid::new(Aabb::new(Vec3::ZERO, max), 6, 2, 2)
}

/// A source in the air near the front (`x` low) face.
fn source() -> SourcePlacement {
    SourcePlacement::new(Vec3::new(0.25, 1.0, 1.0))
}

/// A listener position well behind the wall.
fn listener_behind_wall() -> Vec3 {
    Vec3::new(5.0, 1.0, 1.0)
}

#[test]
fn a_solid_wall_bakes_a_more_occluded_field_than_open_air() {
    let open = bake_field(&open_scene(), &grid(), &source(), &bake_config());
    let walled = bake_field(&walled_scene(), &grid(), &source(), &bake_config());

    let pos = listener_behind_wall();
    let open_params = ParameterLookup::new(&open).sample(pos);
    let walled_params = ParameterLookup::new(&walled).sample(pos);

    // The wall must attenuate the direct path behind it relative to open air.
    assert!(
        walled_params.direct_gain < open_params.direct_gain,
        "wall should reduce direct gain: walled {} vs open {}",
        walled_params.direct_gain,
        open_params.direct_gain
    );
    // A sealed region receives essentially no direct energy.
    assert!(
        walled_params.direct_gain < 0.2,
        "behind a full wall the direct gain should be near zero, got {}",
        walled_params.direct_gain
    );
    // The open path carries real reverberant energy, so the field encodes a
    // meaningful wet send there (a fully sealed region, by contrast, receives
    // no energy at all -- neither direct nor reverberant).
    assert!(
        open_params.wet_gain > 0.1,
        "the open path should carry reverberant energy, got {}",
        open_params.wet_gain
    );
    // Every decoded scalar stays finite and in a sane band.
    for p in [open_params, walled_params] {
        assert!(
            p.direct_gain.is_finite() && (-1.0e-3..=1.0 + 1.0e-3).contains(&p.direct_gain),
            "direct_gain out of band: {}",
            p.direct_gain
        );
        assert!(
            p.wet_gain.is_finite() && (-1.0e-3..=1.0 + 1.0e-3).contains(&p.wet_gain),
            "wet_gain out of band: {}",
            p.wet_gain
        );
        assert!(p.direct_cutoff_hz.is_finite() && p.direct_cutoff_hz > 0.0);
    }
}

#[test]
fn the_bake_is_bit_reproducible() {
    let first = bake_field(&walled_scene(), &grid(), &source(), &bake_config());
    let second = bake_field(&walled_scene(), &grid(), &source(), &bake_config());
    assert_eq!(
        first.to_packed_bytes(),
        second.to_packed_bytes(),
        "two identical bakes must produce identical packed fields"
    );
}

#[test]
fn a_packed_field_round_trips_through_bytes() {
    let field = bake_field(&walled_scene(), &grid(), &source(), &bake_config());
    let bytes = field.to_packed_bytes();
    let restored = ParameterField::from_packed_bytes(&bytes).expect("packed field must decode");

    assert_eq!(field.probe_count(), restored.probe_count());
    assert_eq!(field.bit_depth(), restored.bit_depth());
    for i in 0..field.probe_count() {
        let a = field.decode(i);
        let b = restored.decode(i);
        assert!(approx(a.direct_gain, b.direct_gain, 1.0e-6));
        assert!(approx(a.direct_cutoff_hz, b.direct_cutoff_hz, 1.0e-1));
        assert!(approx(a.wet_gain, b.wet_gain, 1.0e-6));
        assert!(approx(a.azimuth, b.azimuth, 1.0e-3));
        assert!(approx(a.elevation, b.elevation, 1.0e-3));
    }
}

#[test]
fn a_fully_rigid_scene_bakes_occluded_everywhere() {
    let mut rigid = open_scene();
    for ix in 0..NX {
        for iy in 0..NY {
            for iz in 0..NZ {
                rigid.set_solid(ix, iy, iz, true);
            }
        }
    }
    let field = bake_field(&rigid, &grid(), &source(), &bake_config());
    for i in 0..field.probe_count() {
        let p = field.decode(i);
        assert!(
            p.direct_gain < 1.0e-3,
            "a rigid scene must bake fully occluded, probe {i} had gain {}",
            p.direct_gain
        );
    }
}

#[test]
fn the_runtime_smoother_converges_to_the_lookup_target() {
    let field = bake_field(&open_scene(), &grid(), &source(), &bake_config());
    let lookup = ParameterLookup::new(&field);
    let target = lookup.sample(listener_behind_wall());

    // Start far from the target so the ramp is observable.
    let mut smoother = WaveParamSmoother::new(0.5);
    smoother.reset(PerceptualParams::OCCLUDED);

    let first = smoother.process(target);
    // One pole: the first step lands strictly between start and target.
    assert!(
        first.direct_gain >= PerceptualParams::OCCLUDED.direct_gain - 1.0e-6
            && first.direct_gain <= target.direct_gain + 1.0e-6,
        "first smoothed step {} must sit between start and target {}",
        first.direct_gain,
        target.direct_gain
    );

    // After many blocks of the same target the smoother converges to it.
    let mut last = first;
    for _ in 0..200 {
        last = smoother.process(target);
    }
    assert!(approx(last.direct_gain, target.direct_gain, 1.0e-3));
    assert!(approx(last.wet_gain, target.wet_gain, 1.0e-3));
    assert!(approx(last.azimuth, target.azimuth, 1.0e-3));
}

#[test]
fn the_backend_sample_matches_a_bare_lookup() {
    let field = bake_field(&walled_scene(), &grid(), &source(), &bake_config());
    let reference = ParameterLookup::new(&field).sample(listener_behind_wall());

    let backend = WaveBackend::new(field);
    let via_backend = backend.sample_perceptual(listener_behind_wall());
    let via_trait = WaveParameterSource::sample_spatial(&backend, listener_behind_wall());

    assert!(approx(via_backend.direct_gain, reference.direct_gain, 1.0e-6));
    assert!(approx(via_backend.wet_gain, reference.wet_gain, 1.0e-6));
    // The trait projects the same perceptual sample onto the spatial bus.
    assert!(approx(via_trait.direct_gain, reference.direct_gain, 1.0e-6));
    assert!(approx(via_trait.wet_gain, reference.wet_gain, 1.0e-6));
    assert!(approx(via_trait.pitch_ratio, 1.0, 1.0e-6));
}

/// A fully dynamic geometric contribution: unity direct path, Doppler, and a
/// finite image width.
fn geometric_contribution() -> SpatialParams {
    SpatialParams {
        direct_gain: 1.0,
        pitch_ratio: 1.5,
        azimuth: 0.3,
        elevation: -0.1,
        direct_cutoff_hz: 20_000.0,
        wet_gain: 0.05,
        spread: SpreadParams {
            spread: 0.8,
            focus: 0.2,
            half_width: 0.7,
        },
    }
}

/// A baked wave contribution: occluded direct path, no Doppler, point image.
fn wave_contribution() -> SpatialParams {
    SpatialParams {
        direct_gain: 0.2,
        pitch_ratio: 1.0,
        azimuth: -0.4,
        elevation: 0.2,
        direct_cutoff_hz: 500.0,
        wet_gain: 0.7,
        spread: SpreadParams::POINT,
    }
}

#[test]
fn tier_routing_crossfades_but_keeps_geometric_kinematics() {
    let geo = geometric_contribution();
    let wave = wave_contribution();

    // Geometric tier ignores the baked field entirely.
    let g = route(PropagationTier::Geometric, geo, wave);
    assert!(approx(g.direct_gain, geo.direct_gain, 1.0e-6));
    assert!(approx(g.direct_cutoff_hz, geo.direct_cutoff_hz, 1.0e-3));
    assert!(approx(g.wet_gain, geo.wet_gain, 1.0e-6));

    // Wave tier takes the wave propagation but preserves Doppler and width,
    // which a static baked field cannot carry.
    let w = route(PropagationTier::Wave, geo, wave);
    assert!(approx(w.direct_gain, wave.direct_gain, 1.0e-6));
    assert!(approx(w.wet_gain, wave.wet_gain, 1.0e-6));
    assert!(w.direct_cutoff_hz <= wave.direct_cutoff_hz + 1.0e-3);
    assert!(approx(w.pitch_ratio, geo.pitch_ratio, 1.0e-6));
    assert!(approx(w.spread.half_width, geo.spread.half_width, 1.0e-6));

    // Hybrid tier lands between the two for the crossfaded quantities while
    // still keeping the geometric kinematics.
    let h = route(PropagationTier::Hybrid, geo, wave);
    let lo = wave.direct_gain.min(geo.direct_gain);
    let hi = wave.direct_gain.max(geo.direct_gain);
    assert!(h.direct_gain >= lo - 1.0e-6 && h.direct_gain <= hi + 1.0e-6);
    assert!(h.wet_gain > geo.wet_gain && h.wet_gain < wave.wet_gain);
    assert!(approx(h.pitch_ratio, geo.pitch_ratio, 1.0e-6));
    // The low-pass tightening rule never loosens the corner below either input.
    assert!(h.direct_cutoff_hz <= geo.direct_cutoff_hz + 1.0e-3);
}

#[test]
fn a_multi_state_opening_blends_monotonically_with_openness() {
    let door = MultiStateOpening::shut_open();
    assert_eq!(door.state_count(), 2);

    let shut = door.evaluate(0.0);
    let mid = door.evaluate(0.5);
    let open = door.evaluate(1.0);

    // Direct gain rises monotonically as the door opens.
    assert!(shut.direct_gain <= mid.direct_gain);
    assert!(mid.direct_gain <= open.direct_gain);
    assert!(shut.direct_gain < open.direct_gain);
    // The wet send falls as the door opens.
    assert!(shut.wet_gain >= mid.wet_gain);
    assert!(mid.wet_gain >= open.wet_gain);
    // Endpoints clamp exactly to the keyframes.
    assert!(approx(shut.direct_gain, PerceptualParams::OCCLUDED.direct_gain, 1.0e-6));
    assert!(approx(open.direct_gain, PerceptualParams::OPEN.direct_gain, 1.0e-6));
    // Clamping below and above the range is stable.
    assert!(approx(
        door.evaluate(-1.0).direct_gain,
        PerceptualParams::OCCLUDED.direct_gain,
        1.0e-6
    ));
    assert!(approx(
        door.evaluate(2.0).direct_gain,
        PerceptualParams::OPEN.direct_gain,
        1.0e-6
    ));
}
