//! Real-device parity for the decal-projection twin:
//! [`GpuDecal`](prism_volumetric_gpu::decal::GpuDecal) must reproduce the `CPU`
//! golden
//! [`projected_sample`](prism_render_architecture::particle::decal::projected_sample)
//! across a center point squarely facing the projector (`UV` `(0.5, 0.5)`,
//! half-depth fade), an off-center interior point (`UV` away from center), a
//! point outside the box (clipped to a miss), a back-facing interior point (hit
//! with zero fade), a collapsed angle band (hard step at `cos_full`), a
//! collapsed depth band (hard step at `fade_end`), and a randomized batch
//! compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore pins the discrete `hit` flag with an
//! exact `==` and allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the `UV`
//! and `fade` fields.
//!
//! # Conditioning
//!
//! Every randomized fixture is kept well away from each branch crack: the local
//! coordinates stay inside the unit cube by a comfortable margin (so both
//! devices agree on the clip), the alignment stays comfortably positive and
//! below one (clear of the back-face branch and the full-opacity clamp), the
//! half-extents stay well above the compare epsilon, and the fade bands stay
//! comfortably non-degenerate. The degenerate-band fixtures use exact
//! `cos_full == cos_threshold` and `fade_end == fade_start` with the probe placed
//! clear of the step so both devices fold the identical hard step.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::decal`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::decal::{DecalFadeParams, DecalProjector};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::decal::{golden, DecalProjection, DecalQuery, GpuDecal};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// The default fade bands used across the fixtures: a full depth span `[-1, 1]`
/// and an angle band fading from `cos_threshold = 0` to `cos_full = 1`.
fn default_fade() -> DecalFadeParams {
    DecalFadeParams {
        depth_fade_start: -1.0,
        depth_fade_end: 1.0,
        cos_threshold: 0.0,
        cos_full: 1.0,
    }
}

/// An axis-aligned projector centered at the origin with half-extents
/// `(1, 2, 3)`, built through the reference constructor so the basis is
/// orthonormal.
fn axis_aligned() -> DecalProjector {
    DecalProjector::from_forward_up(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 2.0, 3.0),
    )
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

/// Builds a clearly-conditioned query by rejection sampling: a random
/// orthonormal projector, a receiver point placed inside the box by a comfortable
/// margin, and a surface normal whose alignment with the projector stays clear of
/// both the back-face branch and the full-opacity clamp. All arithmetic is finite
/// and avoids `f32` transcendental methods.
fn rand_query(state: &mut u64) -> DecalQuery {
    loop {
        let forward = rand_vec(state, 1.0);
        let up_reference = rand_vec(state, 1.0);
        if forward.length_squared() < 0.25 || up_reference.length_squared() < 0.25 {
            continue;
        }
        // Half-extents comfortably above the compare epsilon.
        let half_extents = Vec3::new(
            lcg(state) * 2.0 + 0.5,
            lcg(state) * 2.0 + 0.5,
            lcg(state) * 2.0 + 0.5,
        );
        let projector =
            DecalProjector::from_forward_up(rand_vec(state, 3.0), forward, up_reference, half_extents);
        // A valid orthonormal basis is guaranteed by the constructor, but guard
        // against a (degenerate) collapse before using it to place the point.
        if projector.right.length_squared() < 0.5 || projector.up.length_squared() < 0.5 {
            continue;
        }

        // Place the receiver point inside the box, each local coordinate kept in
        // [-0.8, 0.8] so both devices agree on the clip with margin to spare.
        let lx = signed(state, 0.8);
        let ly = signed(state, 0.8);
        let lz = signed(state, 0.8);
        let world = projector
            .center
            .add(projector.right.scale(lx * half_extents.x))
            .add(projector.up.scale(ly * half_extents.y))
            .add(projector.forward.scale(lz * half_extents.z));

        // Build a surface normal whose alignment -(normal . forward) lands in
        // [0.25, 0.9]: start from the facing direction (-forward) and tilt it by a
        // perpendicular jitter, then re-check the realized alignment.
        let jitter = rand_vec(state, 0.5);
        let normal_raw = projector.forward.scale(-1.0).add(jitter);
        if normal_raw.length_squared() < 0.25 {
            continue;
        }
        let normal = normal_raw.normalize_or_zero();
        let alignment = -normal.dot(projector.forward);
        if !(0.25..=0.9).contains(&alignment) {
            continue;
        }

        return DecalQuery::new(projector, default_fade(), world, normal);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the `hit` tag
/// matches exactly and, when both hit, the `UV` and combined `fade` match within
/// bound.
fn pin(idx: usize, query: &DecalQuery, got: &DecalProjection) {
    let want = golden(query);
    assert_eq!(
        got.hit, want.hit,
        "query {idx} hit: gpu {} vs cpu {}",
        got.hit, want.hit
    );
    if want.hit {
        assert!(
            close(got.uv[0], want.uv[0]),
            "query {idx} uv.u: gpu {} vs cpu {}",
            got.uv[0],
            want.uv[0]
        );
        assert!(
            close(got.uv[1], want.uv[1]),
            "query {idx} uv.v: gpu {} vs cpu {}",
            got.uv[1],
            want.uv[1]
        );
        assert!(
            close(got.fade, want.fade),
            "query {idx} fade: gpu {} vs cpu {}",
            got.fade,
            want.fade
        );
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuDecal, queries: &[DecalQuery]) {
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

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDecal::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn center_facing_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDecal::new(&ctx);
    // The box center, facing the projector: UV (0.5, 0.5); local z = 0 gives a
    // half depth fade and the squarely-facing normal gives a full angle fade, so
    // the combined fade is 0.5. Integer geometry is exact on both devices.
    let query = DecalQuery::new(
        axis_aligned(),
        default_fade(),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn off_center_interior_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDecal::new(&ctx);
    // Half-way along +right and +up (local (0.5, 0.5, 0)) maps to UV (0.75, 0.75)
    // with a half depth fade and full angle fade -> combined 0.5.
    let query = DecalQuery::new(
        axis_aligned(),
        default_fade(),
        Vec3::new(0.5, 1.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn outside_box_is_clipped_to_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDecal::new(&ctx);
    // Well past the +right face (local x = 5): the projection clips the point and
    // both devices report a miss.
    let query = DecalQuery::new(
        axis_aligned(),
        default_fade(),
        Vec3::new(5.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn back_facing_interior_is_hit_with_zero_fade() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDecal::new(&ctx);
    // Interior point but the normal faces away from the projector (alignment -1):
    // the angle fade is zero, so the combined fade is zero while the point still
    // hits the box.
    let query = DecalQuery::new(
        axis_aligned(),
        default_fade(),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_angle_band_is_hard_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDecal::new(&ctx);
    // cos_full == cos_threshold collapses the angle band to a hard step at
    // cos_full = 0.5. The squarely-facing normal gives alignment 1.0, comfortably
    // above the step, so the angle fade is 1.0 and the combined fade equals the
    // half depth fade, 0.5.
    let params = DecalFadeParams {
        depth_fade_start: -1.0,
        depth_fade_end: 1.0,
        cos_threshold: 0.5,
        cos_full: 0.5,
    };
    let query = DecalQuery::new(
        axis_aligned(),
        params,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn degenerate_depth_band_is_hard_step() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDecal::new(&ctx);
    // fade_end == fade_start collapses the depth band to a hard step at
    // fade_end = 0.4. The center's local z = 0 is comfortably below the step, so
    // the depth fade is 1.0 and the combined fade equals the full angle fade, 1.0.
    let params = DecalFadeParams {
        depth_fade_start: 0.4,
        depth_fade_end: 0.4,
        cos_threshold: 0.0,
        cos_full: 1.0,
    };
    let query = DecalQuery::new(
        axis_aligned(),
        params,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDecal::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries, dispatched
    // together so the per-thread indexing and the contiguous storage layout are
    // both exercised, then pinned element-for-element.
    let mut queries = vec![
        DecalQuery::new(
            axis_aligned(),
            default_fade(),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ),
        DecalQuery::new(
            axis_aligned(),
            default_fade(),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ),
    ];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDecal::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins the projection across many
    // random projector geometries and receiver points.
    let queries: Vec<DecalQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
