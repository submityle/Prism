//! Real-device parity for the signed-tetrahedron-volume twin:
//! [`GpuTetrahedronVolume`](prism_volumetric_gpu::tetrahedron_volume::GpuTetrahedronVolume)
//! must reproduce the `CPU` golden
//! [`tetrahedron_volume`](prism_render_architecture::particle::tetrahedron_volume)
//! across a positively-oriented unit tetrahedron (strictly positive volume), a
//! vertex-swapped tetrahedron (strictly negative volume), a coplanar quad
//! (degenerate, zero volume), a sliver / needle tetrahedron (tiny but still
//! clearly non-degenerate volume), and a randomized batch of clearly-conditioned
//! tetrahedra compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each tetrahedron is a fixed, non-reorderable sequence of multiplies, adds and
//! one divide, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field, while the integer orientation code
//! is compared exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the degeneracy crack: the
//! positively- and negatively-oriented fixtures have determinants of magnitude
//! `1` (far above the compare epsilon), the coplanar quad has an exactly-zero
//! determinant from integer coordinates, and the sliver tetrahedron keeps a
//! determinant well above the epsilon while being geometrically thin. This keeps
//! `CPU` and `GPU` on the same side of the orientation epsilon regardless of a
//! few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`tetrahedron_volume`](prism_render_architecture::particle::tetrahedron_volume);
//! no third-party engine source or derived code.

use prism_render_architecture::particle::tetrahedron_volume::{
    orient3d, orient3d_det, signed_volume,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::tetrahedron_volume::{
    GpuTetrahedronVolume, TetrahedronVolumeQuery, TetrahedronVolumeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// Builds a clearly-conditioned tetrahedron by rejection sampling: four vertices
/// are drawn from well-spread positions and accepted only when the `CPU`
/// reference determinant clears a safe non-degenerate magnitude, so every random
/// fixture stays far from the coplanar crack and `CPU` and `GPU` share the
/// orientation branch.
fn solid_query(state: &mut u64) -> TetrahedronVolumeQuery {
    loop {
        let a = rand_vec(state, 5.0);
        let b = rand_vec(state, 5.0);
        let c = rand_vec(state, 5.0);
        let d = rand_vec(state, 5.0);
        // The determinant must be comfortably above the compare epsilon.
        if orient3d_det(a, b, c, d).abs() < 1.0 {
            continue;
        }
        return TetrahedronVolumeQuery::new(a, b, c, d);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the orientation
/// determinant, the signed volume, the absolute volume and the orientation code
/// must all agree within bound (the orientation code exactly).
fn pin(idx: usize, query: &TetrahedronVolumeQuery, got: &TetrahedronVolumeResult) {
    let want_det = orient3d_det(query.a, query.b, query.c, query.d);
    let want_signed = signed_volume(query.a, query.b, query.c, query.d);
    let want_abs = want_signed.abs();
    let want_orient = orient3d(query.a, query.b, query.c, query.d);

    assert!(
        close(got.det, want_det),
        "query {idx} det: gpu {} vs cpu {}",
        got.det,
        want_det
    );
    assert!(
        close(got.signed_volume, want_signed),
        "query {idx} signed_volume: gpu {} vs cpu {}",
        got.signed_volume,
        want_signed
    );
    assert!(
        close(got.abs_volume, want_abs),
        "query {idx} abs_volume: gpu {} vs cpu {}",
        got.abs_volume,
        want_abs
    );
    assert_eq!(
        got.orientation, want_orient,
        "query {idx} orientation: gpu {:?} vs cpu {:?}",
        got.orientation, want_orient
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuTetrahedronVolume, queries: &[TetrahedronVolumeQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// The canonical unit tetrahedron (positively oriented, volume `1/6`).
fn unit_tetra() -> TetrahedronVolumeQuery {
    TetrahedronVolumeQuery::new(
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    )
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetrahedronVolume::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn positive_unit_tetrahedron_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetrahedronVolume::new(&ctx);
    // The canonical unit tetrahedron: determinant 1, signed volume +1/6.
    check(&ctx, &gpu, &[unit_tetra()]);
}

#[test]
fn vertex_swap_negates_volume_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetrahedronVolume::new(&ctx);
    // Swapping two vertices of the unit tetrahedron negates the signed volume,
    // flipping the orientation to Negative while the absolute volume is 1/6.
    let query = TetrahedronVolumeQuery::new(
        Vec3::ZERO,
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn coplanar_quad_is_degenerate_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetrahedronVolume::new(&ctx);
    // Four points on the z = 0 plane: the determinant is exactly zero on both
    // devices, so the Coplanar branch fires deterministically.
    let query = TetrahedronVolumeQuery::new(
        Vec3::ZERO,
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(2.0, 3.0, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn sliver_tetrahedron_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetrahedronVolume::new(&ctx);
    // A thin needle tetrahedron: the apex sits only a little off the base plane,
    // so the volume is small but the determinant (6 units) stays well clear of
    // the compare epsilon, keeping both devices on the Positive branch.
    let query = TetrahedronVolumeQuery::new(
        Vec3::ZERO,
        Vec3::new(10.0, 0.0, 0.0),
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(3.0, 3.0, 1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetrahedronVolume::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random solid
    // tetrahedra, dispatched together so the per-thread indexing and the
    // contiguous storage layout are both exercised, then pinned element-for-
    // element.
    let mut queries = vec![
        unit_tetra(),
        TetrahedronVolumeQuery::new(
            Vec3::ZERO,
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(3.0, 3.0, 1.0),
        ),
    ];
    for _ in 0..48 {
        queries.push(solid_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_solid_tetrahedra_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetrahedronVolume::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned solid tetrahedra (several
    // workgroups' worth) pins every reported field across many random
    // geometries.
    let queries: Vec<TetrahedronVolumeQuery> = (0..200).map(|_| solid_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
