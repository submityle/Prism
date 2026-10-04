//! Batched multi-query validation for the `GPU` geometric-acoustics backend
//! (design section 31).
//!
//! The crate's reason for existing is resolving *one uploaded scene against
//! many `(listener, emitter)` queries at once*: the direct and reflection
//! kernels dispatch one invocation per query (and per triangle), and
//! [`GpuGeometryBackend::resolve`] returns a result vector parallel to the
//! input batch. The in-module unit tests only ever drive a single-query batch,
//! so the parallel-decode and per-query isolation of the batch path are
//! unverified. These integration tests close that gap by driving batches of
//! several distinct queries and asserting three independent properties on a
//! real device:
//!
//! 1. **Isolation / alignment** — resolving a whole batch yields, slot for
//!    slot, exactly what resolving each query on its own does. A query cannot
//!    leak into its neighbour's result and the output vector stays aligned to
//!    the input.
//! 2. **Host-twin agreement** — the device batch matches the `CPU` twin batch
//!    ([`GpuGeometryBackend::resolve_cpu`]) within a tight tolerance across
//!    every slot.
//! 3. **Reference-backend agreement** — on clear-line scenes (where the `CPU`
//!    diffraction stage is empty) the device batch tracks the authoritative
//!    [`GeometricBackend`] arrival-for-arrival across every slot.
//!
//! Each test acquires a device through [`GpuContext::try_headless`] and skips
//! gracefully when no adapter is available, so headless hosts stay green while
//! hosts with a usable adapter exercise the real kernels.
//!
//! # Provenance
//!
//! Original work; standard `wgpu` compute already implemented in the crate and
//! the classic image-source / ray-march acoustics in [`prism_audio_geometry`];
//! no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, Dolby, or Google
//! Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//!
//! Exercises the public surface of [`prism_audio_geometry_gpu`]
//! ([`GpuContext`], [`GpuScene`], [`GpuGeometryBackend`], [`ResolvedQuery`])
//! and cross-checks it against the `CPU` [`GeometricBackend`] from
//! [`prism_audio_geometry`], using the spatial
//! [`Listener`]/[`Emitter`]/[`PropagationPath`] types from
//! [`prism_audio_spatial`].

use bevy_math::{Quat, Vec3};
use prism_audio_geometry::{AcousticScene, GeometricBackend, GeometricConfig, MaterialTable};
use prism_audio_geometry_gpu::{GpuContext, GpuGeometryBackend, GpuScene, ResolvedQuery};
use prism_audio_spatial::geometry::{Emitter, Listener};
use prism_audio_spatial::propagation::{
    AcousticMaterial, PathKind, PropagationBackend, PropagationPath, MAX_PROPAGATION_PATHS,
};

/// Absolute tolerance for comparing the device against an independent `f32`
/// arithmetic path (the `CPU` reference backend), matching the in-module
/// backend tests.
const TOL: f32 = 1.0e-4;

/// Tighter tolerance for the device-versus-host-twin comparison, where the only
/// divergence is the shader's fused multiply-add versus the scalar twin.
const TWIN_TOL: f32 = 1.0e-5;

/// A large, highly reflective, acoustically opaque floor on the `y = 0` plane.
///
/// Every query in these batches keeps both endpoints above the floor, so the
/// direct line is always clear and the floor contributes a single first-order
/// specular reflection — the clear-line regime in which the `CPU` diffraction
/// stage is empty and all three backends agree.
fn floor_scene() -> AcousticScene {
    let vertices = Vec::from([
        Vec3::new(-40.0, 0.0, -40.0),
        Vec3::new(40.0, 0.0, -40.0),
        Vec3::new(40.0, 0.0, 40.0),
        Vec3::new(-40.0, 0.0, 40.0),
    ]);
    let indices = Vec::from([[0, 1, 2], [0, 2, 3]]);
    let table = MaterialTable::uniform_scalar(AcousticMaterial::new(0.0, 0.9));
    AcousticScene::new(vertices, indices, table).expect("floor scene builds")
}

/// A listener at `position`, facing forward with no velocity.
fn listener_at(position: Vec3) -> Listener {
    Listener::new(position, Quat::IDENTITY, Vec3::ZERO)
}

/// A stationary point emitter at `position`.
fn emitter_at(position: Vec3) -> Emitter {
    Emitter::point(position, Vec3::ZERO)
}

/// A spread of six distinct queries over the floor scene: different distances,
/// heights, and lateral offsets so no two slots share a direct delay or
/// reflection geometry. Each endpoint stays above the floor.
fn batch() -> Vec<(Listener, Emitter)> {
    Vec::from([
        (listener_at(Vec3::new(-4.0, 2.0, 0.0)), emitter_at(Vec3::new(4.0, 2.0, 0.0))),
        (listener_at(Vec3::new(-6.0, 1.0, 3.0)), emitter_at(Vec3::new(5.0, 3.0, -2.0))),
        (listener_at(Vec3::new(-1.0, 5.0, -8.0)), emitter_at(Vec3::new(2.0, 1.5, 9.0))),
        (listener_at(Vec3::new(-10.0, 0.5, 1.0)), emitter_at(Vec3::new(10.0, 4.0, 1.0))),
        (listener_at(Vec3::new(0.0, 3.0, 0.0)), emitter_at(Vec3::new(0.0, 3.0, 7.0))),
        (listener_at(Vec3::new(7.0, 2.5, -5.0)), emitter_at(Vec3::new(-7.0, 2.5, 5.0))),
    ])
}

/// Asserts two `ResolvedQuery` values agree within `tol`: identical path kinds
/// and counts, with each arrival's gain, delay, and direction matched to the
/// tolerance and the direct occlusion factor matched to the tolerance.
fn assert_resolved_close(got: &ResolvedQuery, want: &ResolvedQuery, tol: f32) {
    assert_eq!(got.paths.len(), want.paths.len(), "path count");
    assert!(
        (got.direct.direct_factor() - want.direct.direct_factor()).abs() < tol,
        "direct factor"
    );
    for (g, w) in got.paths.iter().zip(want.paths.iter()) {
        assert_eq!(g.kind, w.kind, "path kind");
        assert!((g.gain - w.gain).abs() < tol, "gain");
        assert!((g.delay_seconds - w.delay_seconds).abs() < tol, "delay");
        assert!(g.direction.dot(w.direction) > 1.0 - tol, "direction");
    }
}

/// Resolving a batch produces, slot for slot, exactly what resolving each query
/// alone produces: the device decode stays aligned to the input and no query
/// contaminates another.
#[test]
fn batch_result_is_isolated_and_aligned() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scene = floor_scene();
    let config = GeometricConfig::new(48_000);
    let gpu_scene = GpuScene::upload(&ctx, &scene);
    let backend = GpuGeometryBackend::new(&ctx);
    let queries = batch();

    let batched = backend.resolve(&ctx, &gpu_scene, &config, &queries);
    assert_eq!(batched.len(), queries.len());

    for (index, query) in queries.iter().enumerate() {
        let single = backend.resolve(&ctx, &gpu_scene, &config, core::slice::from_ref(query));
        assert_eq!(single.len(), 1);
        // Same shader, same inputs: the batched slot must be bit-identical to
        // the standalone resolve of the same query.
        assert_eq!(
            batched[index], single[0],
            "batched slot {index} diverged from its standalone resolve"
        );
    }
}

/// The device batch matches the host twin batch slot for slot. Reordering the
/// queries reorders the results identically, confirming the twin honours the
/// same per-slot mapping as the device.
#[test]
fn batch_matches_host_twin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scene = floor_scene();
    let config = GeometricConfig::new(48_000);
    let gpu_scene = GpuScene::upload(&ctx, &scene);
    let backend = GpuGeometryBackend::new(&ctx);
    let queries = batch();

    let device = backend.resolve(&ctx, &gpu_scene, &config, &queries);
    let twin = backend.resolve_cpu(&gpu_scene, &config, &queries);
    assert_eq!(device.len(), twin.len());
    for (index, (got, want)) in device.iter().zip(twin.iter()).enumerate() {
        assert!(
            !got.paths.is_empty(),
            "slot {index} should carry at least the direct arrival"
        );
        assert_resolved_close(got, want, TWIN_TOL);
    }

    // Reversing the batch must reverse the result vector exactly.
    let mut reversed = queries.clone();
    reversed.reverse();
    let device_reversed = backend.resolve(&ctx, &gpu_scene, &config, &reversed);
    for (index, got) in device_reversed.iter().enumerate() {
        let mirror = &device[device.len() - 1 - index];
        assert_resolved_close(got, mirror, TWIN_TOL);
    }
}

/// On the clear-line floor scene the device batch tracks the authoritative
/// `CPU` [`GeometricBackend`] arrival-for-arrival across every slot, proving the
/// drop-in device-sibling claim holds at batch scale, not just for one query.
#[test]
fn batch_tracks_reference_backend() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scene = floor_scene();
    let config = GeometricConfig::new(48_000);
    let gpu_scene = GpuScene::upload(&ctx, &scene);
    let backend = GpuGeometryBackend::new(&ctx);
    let queries = batch();

    let device = backend.resolve(&ctx, &gpu_scene, &config, &queries);
    let reference = GeometricBackend::new(scene, config);

    for (index, (listener, emitter)) in queries.iter().enumerate() {
        let mut paths = [PropagationPath::SILENT; MAX_PROPAGATION_PATHS];
        let summary = reference.query(listener, emitter, &mut paths);
        assert_eq!(
            device[index].paths.len(),
            summary.path_count,
            "slot {index} path count"
        );
        for (got, want) in device[index].paths.iter().zip(paths.iter()) {
            assert_eq!(got.kind, want.kind, "slot {index} kind");
            assert!((got.gain - want.gain).abs() < TOL, "slot {index} gain");
            assert!(
                (got.delay_seconds - want.delay_seconds).abs() < TOL,
                "slot {index} delay"
            );
            assert!(
                got.direction.dot(want.direction) > 1.0 - TOL,
                "slot {index} direction"
            );
        }
        // The clear-line direct arrival is first and audible.
        assert_eq!(device[index].paths[0].kind, PathKind::Direct);
    }
}

/// An empty batch resolves to an empty result on both the device and the host
/// twin, with no device work dispatched.
#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let scene = floor_scene();
    let config = GeometricConfig::new(48_000);
    let gpu_scene = GpuScene::upload(&ctx, &scene);
    let backend = GpuGeometryBackend::new(&ctx);

    assert!(backend.resolve(&ctx, &gpu_scene, &config, &[]).is_empty());
    assert!(backend.resolve_cpu(&gpu_scene, &config, &[]).is_empty());
}
