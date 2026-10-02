//! Real-device parity for the rough-diffuse twin:
//! [`GpuOrenNayar`](prism_volumetric_gpu::oren_nayar::GpuOrenNayar) must
//! reproduce the `CPU` golden
//! [`oren_nayar`](prism_render_architecture::particle::oren_nayar) across both
//! closures it exposes — the trig-free Oren-Nayar reflected radiance and the
//! Burley / Disney diffuse reflectance.
//!
//! The fixtures cover the shapes the golden unit tests call out: a head-on
//! geometry where Oren-Nayar collapses to Lambert at `sigma = 0`, a moderate
//! off-axis geometry, a grazing retro-reflection where roughness lifts the
//! Burley lobe, a back-lit query (light below the horizon) and a back-facing
//! query (view below the horizon) that both gate to zero, and a randomized
//! batch compared element-for-element. Every direction is an exactly normalized
//! vector and every `sigma` / `roughness` is an interior value, so no fixture
//! samples a `transcendental` and none sits on a branch tie.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each closure is a fixed, non-reorderable sequence of multiplies, adds,
//! divides and (for Burley) one `sqrt`, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every
//! `f32` lane of both reflectances.
//!
//! # Conditioning
//!
//! Every random fixture is kept well away from the two branch cracks: the
//! clamped cosines `NoL` and `NoV` are both held comfortably positive so the
//! back-facing gate is never on a tie, and the Oren-Nayar azimuthal term `s` is
//! held comfortably away from zero so `CPU` and `GPU` take the same `s / t`
//! denominator branch regardless of a few units in the last place of slack. The
//! back-lit and back-facing fixtures deliberately cross the gate so both devices
//! return zero.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::oren_nayar`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::oren_nayar::{golden, GpuOrenNayar, OrenNayarQuery, OrenNayarResult};
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

/// Returns whether two vectors agree lane for lane within the bound.
fn close_vec(a: Vec3, b: Vec3) -> bool {
    close(a.x, b.x) && close(a.y, b.y) && close(a.z, b.z)
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

/// A pseudo-random, exactly normalized direction, rejecting a vector too short
/// to normalize cleanly. Uses only `normalize_or_zero` (one `sqrt`), so no
/// transcendental method is called.
fn rand_dir(state: &mut u64) -> Vec3 {
    loop {
        let v = Vec3::new(signed(state, 1.0), signed(state, 1.0), signed(state, 1.0));
        if v.length_squared() > 0.2 {
            return v.normalize_or_zero();
        }
    }
}

/// Builds a clearly-conditioned query by rejection sampling: both clamped
/// cosines are held above `0.2` so the back-facing gate is never on a tie, and
/// the Oren-Nayar azimuthal term `s` is held at least `0.05` from zero so both
/// devices take the same `s / t` denominator branch. `sigma` and `roughness`
/// are interior values.
fn rand_query(state: &mut u64) -> OrenNayarQuery {
    loop {
        let normal = rand_dir(state);
        let light = rand_dir(state);
        let view = rand_dir(state);
        let n_dot_l = normal.dot(light).max(0.0);
        let n_dot_v = normal.dot(view).max(0.0);
        if n_dot_l < 0.2 || n_dot_v < 0.2 {
            continue;
        }
        let s = light.dot(view) - n_dot_l * n_dot_v;
        if s.abs() < 0.05 {
            continue;
        }
        let albedo = Vec3::new(
            lcg(state) * 0.8 + 0.1,
            lcg(state) * 0.8 + 0.1,
            lcg(state) * 0.8 + 0.1,
        );
        let sigma = lcg(state) * 1.1 + 0.2;
        let roughness = lcg(state) * 0.75 + 0.15;
        return OrenNayarQuery::new(normal, light, view, albedo, sigma, roughness);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: both reflectances
/// must agree lane for lane within the bound.
fn pin(idx: usize, query: &OrenNayarQuery, got: &OrenNayarResult) {
    let want = golden(query);
    assert!(
        close_vec(got.oren, want.oren),
        "query {idx} oren_nayar: gpu {:?} vs cpu {:?}",
        got.oren,
        want.oren
    );
    assert!(
        close_vec(got.burley, want.burley),
        "query {idx} burley_diffuse: gpu {:?} vs cpu {:?}",
        got.burley,
        want.burley
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuOrenNayar, queries: &[OrenNayarQuery]) {
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

/// Builds an exactly normalized direction from raw components.
fn unit(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z).normalize_or_zero()
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrenNayar::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn head_on_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrenNayar::new(&ctx);
    // Normal, light and view all along +z: NoL = NoV = 1, so Oren-Nayar at
    // sigma = 0 is exactly Lambert and Burley reduces to albedo / pi.
    let query = OrenNayarQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(0.0, 0.0, 1.0),
        unit(0.0, 0.0, 1.0),
        Vec3::new(0.6, 0.7, 0.8),
        0.0,
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn off_axis_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrenNayar::new(&ctx);
    // A moderate off-axis geometry well clear of the horizon; both lobes are
    // non-trivial and both cosines stay comfortably positive.
    let query = OrenNayarQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(0.3, 0.1, 0.95),
        unit(-0.2, 0.25, 0.95),
        Vec3::new(0.8, 0.5, 0.2),
        0.6,
        0.4,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn grazing_retro_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrenNayar::new(&ctx);
    // Light and view nearly aligned near the silhouette (large positive s), where
    // Oren-Nayar roughness brightens the grazing retro-reflection and Burley lifts
    // FD90 above one. Cosines stay above 0.2 so the gate is never on a tie.
    let dir = unit(0.9, 0.0, 0.44);
    let query = OrenNayarQuery::new(
        unit(0.0, 0.0, 1.0),
        dir,
        dir,
        Vec3::new(0.9, 0.9, 0.9),
        1.2,
        0.85,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn back_lit_gates_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrenNayar::new(&ctx);
    // Light below the horizon (NoL <= 0): both closures gate to zero on both
    // devices.
    let query = OrenNayarQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(0.0, 0.0, -1.0),
        unit(0.0, 0.0, 1.0),
        Vec3::new(0.7, 0.7, 0.7),
        0.5,
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn back_facing_gates_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrenNayar::new(&ctx);
    // View below the horizon (NoV <= 0): both closures gate to zero on both
    // devices.
    let query = OrenNayarQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(0.0, 0.0, 1.0),
        unit(0.0, 0.0, -1.0),
        Vec3::new(0.7, 0.7, 0.7),
        0.5,
        0.5,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn random_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrenNayar::new(&ctx);
    let mut state = 0x5eed_1234_abcd_0001_u64;
    let queries: Vec<OrenNayarQuery> = (0..64).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuOrenNayar::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins both reflectances across
    // many random geometries.
    let queries: Vec<OrenNayarQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
