//! Real-device parity for the swept-sphere continuous-collision twin:
//! [`GpuSphereSweep`](prism_volumetric_gpu::sphere_sweep::GpuSphereSweep) must
//! reproduce the `CPU` golden
//! [`sphere_sweep`](prism_render_architecture::particle::sphere_sweep) across
//! swept-sphere-versus-sphere head-on hits, an initial overlap that reports a
//! `toi` of `0` with a separation normal, a grazing miss whose discriminant is
//! clearly negative, swept-sphere-versus-plane approaches from either side with
//! a non-zero plane offset, and swept-sphere-versus-point exact hits and
//! recessions, plus randomized well-conditioned batches of all three ops
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
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on the `f32` fields while pinning the `Miss` / `Hit`
//! classification exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the degenerate regions and the
//! branch boundaries: approaching pairs are built so the first-touch `toi` lands
//! squarely inside `[0.2, 0.8]` with an impact parameter below `0.3` of the
//! summed radius (so the discriminant is clearly positive and the two roots are
//! well separated), the initial overlap starts far inside the summed radius, the
//! grazing miss clears by more than the radius, and the plane sweeps start more
//! than one unit beyond the contact band with a clearly non-parallel approach
//! speed. No fixture uses an `f32` transcendental; the random vectors come from
//! an integer `LCG` and the only non-rational step is a `sqrt`, matching the
//! reference. This keeps `CPU` and `GPU` on the same side of every branch
//! regardless of a few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`sphere_sweep`](prism_render_architecture::particle::sphere_sweep); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::sphere_sweep::{
    sweep_sphere_vs_plane, sweep_sphere_vs_point, sweep_sphere_vs_sphere, v_add, v_cross, v_dot,
    v_normalize_or, v_scale, SweepHit,
};
use prism_volumetric_gpu::sphere_sweep::{GpuSphereSweep, SphereSweepOp, SphereSweepQuery};
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

/// Asserts two vectors agree channel-for-channel within the parity bound.
fn close_vec(label: &str, idx: usize, got: [f32; 3], want: [f32; 3]) {
    assert!(
        close(got[0], want[0]) && close(got[1], want[1]) && close(got[2], want[2]),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got[0],
        got[1],
        got[2],
        want[0],
        want[1],
        want[2]
    );
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
fn rand_vec(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// A unit vector drawn from `state`, retried until it is comfortably non-zero so
/// the normalization is well conditioned.
fn rand_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let v = rand_vec(state, 1.0);
        if v_dot(v, v) > 0.2 {
            return v_normalize_or(v, [1.0, 0.0, 0.0]);
        }
    }
}

/// Returns a unit vector perpendicular to `axis`, built by crossing `axis` with
/// whichever cardinal axis is least parallel to it.
fn perp_unit(axis: [f32; 3]) -> [f32; 3] {
    let helper = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    v_normalize_or(v_cross(axis, helper), [0.0, 0.0, 1.0])
}

/// Builds a clearly-conditioned swept-sphere-versus-sphere hit. The pair is
/// reduced to a moving point against the stationary target `B`: the relative
/// path passes the target centre with a perpendicular impact parameter below
/// `0.3` of the summed radius and a positive relative speed, placed so the first
/// (entering) root lands at `toi_target` inside `[0.2, 0.8]`. Target `B` is
/// given a random velocity and `A`'s velocity carries it, so the relative motion
/// is unchanged while the absolute velocities (which the contact point depends
/// on) are non-trivial.
fn sphere_hit_query(state: &mut u64) -> SphereSweepQuery {
    let u = rand_unit(state);
    let perp = perp_unit(u);
    let radius_a = 0.5 + lcg(state) * 1.5;
    let radius_b = 0.5 + lcg(state) * 1.5;
    let r = radius_a + radius_b;
    let p = signed(state, r * 0.3);
    let s_hit = -((r * r - p * p).max(0.0)).sqrt();
    let toi_target = 0.2 + lcg(state) * 0.6;
    let speed = r + 1.0 + lcg(state) * 2.0;
    let s0 = s_hit - speed * toi_target;

    let center_b = rand_vec(state, 4.0);
    let center_a = v_add(center_b, v_add(v_scale(perp, p), v_scale(u, s0)));
    let vel_b = rand_vec(state, 3.0);
    let vel_a = v_add(v_scale(u, speed), vel_b);

    SphereSweepQuery::sphere(center_a, radius_a, vel_a, center_b, radius_b, vel_b)
}

/// Builds a clearly-conditioned swept-sphere-versus-point hit, the zero-radius
/// stationary degenerate case of [`sphere_hit_query`].
fn point_hit_query(state: &mut u64) -> SphereSweepQuery {
    let u = rand_unit(state);
    let perp = perp_unit(u);
    let radius = 0.5 + lcg(state) * 1.5;
    let p = signed(state, radius * 0.3);
    let s_hit = -((radius * radius - p * p).max(0.0)).sqrt();
    let toi_target = 0.2 + lcg(state) * 0.6;
    let speed = radius + 1.0 + lcg(state) * 2.0;
    let s0 = s_hit - speed * toi_target;

    let point = rand_vec(state, 4.0);
    let center = v_add(point, v_add(v_scale(perp, p), v_scale(u, s0)));
    let vel = v_scale(u, speed);

    SphereSweepQuery::point(center, radius, vel, point)
}

/// Builds a clearly-conditioned swept-sphere-versus-plane hit. The centre starts
/// one radius plus a clear gap off the positive side of the plane and approaches
/// with a plainly non-parallel speed, placed so the first touch lands at
/// `toi_target` inside `[0.2, 0.8]`. A random tangential component exercises the
/// side-independence of the signed-distance solve.
fn plane_hit_query(state: &mut u64) -> SphereSweepQuery {
    let n = rand_unit(state);
    let perp = perp_unit(n);
    let radius = 0.5 + lcg(state) * 1.5;
    let gap = 1.0 + lcg(state) * 2.0;
    let s0 = radius + gap;
    let plane_d = signed(state, 3.0);
    let tang_off = signed(state, 3.0);
    let center = v_add(v_scale(n, s0 + plane_d), v_scale(perp, tang_off));

    let toi_target = 0.2 + lcg(state) * 0.6;
    let sn = -gap / toi_target;
    let tang_speed = signed(state, 2.0);
    let vel = v_add(v_scale(n, sn), v_scale(perp, tang_speed));

    SphereSweepQuery::plane(center, radius, vel, n, plane_d)
}

/// Evaluates the `CPU` golden for `query`, forwarding to the routine its `op`
/// selects.
fn reference(query: &SphereSweepQuery) -> SweepHit {
    match query.op {
        SphereSweepOp::Sphere => sweep_sphere_vs_sphere(
            query.center_a,
            query.radius_a,
            query.vel_a,
            query.geom_b,
            query.radius_b,
            query.vel_b,
        ),
        SphereSweepOp::Plane => sweep_sphere_vs_plane(
            query.center_a,
            query.radius_a,
            query.vel_a,
            query.geom_b,
            query.plane_d,
        ),
        SphereSweepOp::Point => {
            sweep_sphere_vs_point(query.center_a, query.radius_a, query.vel_a, query.geom_b)
        }
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the `Miss` / `Hit`
/// classification must match exactly and, on a hit, the `toi`, outward normal and
/// contact point must agree within the parity bound.
fn pin(idx: usize, query: &SphereSweepQuery, got: &SweepHit) {
    let want = reference(query);
    match (*got, want) {
        (
            SweepHit::Hit {
                toi: gt,
                normal: gn,
                point: gp,
            },
            SweepHit::Hit {
                toi: wt,
                normal: wn,
                point: wp,
            },
        ) => {
            assert!(close(gt, wt), "query {idx} toi: gpu {gt} vs cpu {wt}");
            close_vec("normal", idx, gn, wn);
            close_vec("point", idx, gp, wp);
        }
        (SweepHit::Miss, SweepHit::Miss) => {}
        (g, w) => panic!("query {idx}: classification mismatch gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the reference.
fn check(ctx: &GpuContext, gpu: &GpuSphereSweep, queries: &[SphereSweepQuery]) {
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
    let gpu = GpuSphereSweep::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn sphere_head_on_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    // Two unit spheres closing head-on along x: first touch at toi = 0.4 with an
    // outward normal of -x, well clear of the overlap and discriminant branches.
    let query = SphereSweepQuery::sphere(
        [-5.0, 0.0, 0.0],
        1.0,
        [10.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        1.0,
        [-10.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn sphere_initial_overlap_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    // Centres only 0.5 apart with a summed radius of 2 start deep inside the
    // overlap band: an immediate contact at toi = 0 with a separation normal.
    let query = SphereSweepQuery::sphere(
        [0.0, 0.0, 0.0],
        1.0,
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        1.0,
        [0.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn sphere_grazing_miss_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    // A vertical offset of 3 against a summed radius of 2 keeps the discriminant
    // clearly negative, so the sweep misses.
    let query = SphereSweepQuery::sphere(
        [-5.0, 3.0, 0.0],
        1.0,
        [10.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        1.0,
        [-10.0, 0.0, 0.0],
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn plane_approach_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    // A unit sphere five units below the plane y = 0 moving up at 10: first touch
    // at toi = 0.4 with an outward normal of -y (the side the centre lies on).
    let query = SphereSweepQuery::plane(
        [0.0, -5.0, 0.0],
        1.0,
        [0.0, 10.0, 0.0],
        [0.0, 1.0, 0.0],
        0.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn plane_offset_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    // A unit sphere approaching the offset plane x = 2 from x = 10 moving at -10:
    // first touch at toi = 0.7 with an outward normal of +x.
    let query = SphereSweepQuery::plane(
        [10.0, 0.0, 0.0],
        1.0,
        [-10.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        2.0,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn point_exact_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    // A unit sphere five units left of the origin moving right at 10: it touches
    // the stationary point at toi = 0.4 with an outward normal of -x.
    let query = SphereSweepQuery::point([-5.0, 0.0, 0.0], 1.0, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn point_receding_miss_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    // The same sphere moving away from the stationary point never meets it.
    let query = SphereSweepQuery::point([-5.0, 0.0, 0.0], 1.0, [-10.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing the deterministic fixtures with many random hits of all
    // three ops, dispatched together so the per-thread indexing and contiguous
    // storage layout are both exercised, then pinned element-for-element.
    let mut queries = vec![
        SphereSweepQuery::sphere(
            [-5.0, 0.0, 0.0],
            1.0,
            [10.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            1.0,
            [-10.0, 0.0, 0.0],
        ),
        SphereSweepQuery::sphere(
            [0.0, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            1.0,
            [0.0, 0.0, 0.0],
        ),
        SphereSweepQuery::plane(
            [0.0, -5.0, 0.0],
            1.0,
            [0.0, 10.0, 0.0],
            [0.0, 1.0, 0.0],
            0.0,
        ),
        SphereSweepQuery::point([-5.0, 0.0, 0.0], 1.0, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ];
    for _ in 0..30 {
        queries.push(sphere_hit_query(&mut state));
        queries.push(plane_hit_query(&mut state));
        queries.push(point_hit_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_sphere_hits_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clearly-conditioned swept-sphere hits (several
    // workgroups' worth) pins the time-of-impact, outward normal and contact
    // point across many random pair geometries.
    let queries: Vec<SphereSweepQuery> = (0..200).map(|_| sphere_hit_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_plane_hits_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereSweep::new(&ctx);
    let mut state = 0x00c0_ffee_1337_d00d_u64;
    // A multi-workgroup sweep of plane approaches across random normals, offsets
    // and tangential drift pins the signed-distance solve.
    let queries: Vec<SphereSweepQuery> = (0..200).map(|_| plane_hit_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
