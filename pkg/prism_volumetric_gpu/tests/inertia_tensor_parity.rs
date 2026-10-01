//! Real-device parity for the rigid mass-property twin:
//! [`GpuInertiaTensor`](prism_volumetric_gpu::inertia_tensor::GpuInertiaTensor)
//! must reproduce the `CPU` golden
//! [`inertia_tensor`](prism_render_architecture::particle::inertia_tensor)
//! across the point-cloud reduction (total mass, centre of mass and the inertia
//! tensor about that `COM`) and the per-body algebra (the raw parallel-axis
//! translate, the `inertia_about` re-reference and the two-system merge).
//!
//! The reduction is exercised on a single point (its own `COM`, so the tensor
//! is zero), a mass-symmetric cloud (`COM` at the origin), and a randomized
//! sweep of well-conditioned clouds compared field-for-field. The per-body
//! algebra is exercised on an off-centre reference point (the parallel-axis
//! theorem), on merge commutativity and associativity, on the zero-mass merge
//! guard that collapses to
//! [`MassProperties::EMPTY`](prism_render_architecture::particle::inertia_tensor::MassProperties::EMPTY),
//! and on a randomized batch. Every empty input is checked to short-circuit
//! before any dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! The per-body algebra is a fixed, non-reorderable sequence of multiplies and
//! adds, so `CPU` and `GPU` evaluate the same closed form; they are not
//! bit-exact only because a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate. Those fields are compared with `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` (`REL_FLOOR` `1e-6`).
//!
//! The point-cloud reduction has one extra source of slack: `WGSL` has no native
//! `f32` atomic, so each accumulator is summed through a `bitcast`
//! compare-and-swap loop whose cross-thread order is hardware-defined and
//! run-to-run non-deterministic, whereas the reference sums left to right. The
//! two totals therefore differ by the rounding of a reordered sum. The reduced
//! quantities use a slightly wider absolute floor (`REDUCE_EPS` `2e-3`) with the
//! same relative bound, kept well-conditioned (modest counts, positive masses,
//! moderate coordinates) so the reordered-sum rounding stays far inside it while
//! a genuinely wrong port still fails.
//!
//! Provenance: twinned from this repository's
//! [`inertia_tensor`](prism_render_architecture::particle::inertia_tensor); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::inertia_tensor::{Mat3, MassProperties, Vec3};
use prism_volumetric_gpu::inertia_tensor::{BodyOpQuery, BodyOpResult, GpuInertiaTensor};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound for the closed-form per-body algebra. A `GPU` may fuse
/// a multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Absolute parity bound for the atomic point-cloud reduction. The float
/// compare-and-swap accumulation sums the points in a hardware-defined,
/// run-to-run order, so the reduced totals differ from the reference's
/// left-to-right sum by the rounding of a reordered addition; `2e-3` absorbs
/// that reordering on the well-conditioned fixtures without hiding a real
/// defect.
const REDUCE_EPS: f32 = 2.0e-3;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within `abs_eps` absolutely or [`REL`]
/// relatively (with the [`REL_FLOOR`] denominator floor).
fn approx(a: f32, b: f32, abs_eps: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= abs_eps || rel <= REL
}

/// Asserts two scalars agree within `abs_eps`, reporting both sides on failure.
fn pin_scalar(label: &str, idx: usize, got: f32, want: f32, abs_eps: f32) {
    assert!(
        approx(got, want, abs_eps),
        "query {idx} {label}: gpu {got} vs cpu {want}"
    );
}

/// Asserts two vectors agree channel-for-channel within `abs_eps`.
fn pin_vec(label: &str, idx: usize, got: Vec3, want: Vec3, abs_eps: f32) {
    assert!(
        approx(got.x, want.x, abs_eps)
            && approx(got.y, want.y, abs_eps)
            && approx(got.z, want.z, abs_eps),
        "query {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.x,
        got.y,
        got.z,
        want.x,
        want.y,
        want.z
    );
}

/// Asserts two symmetric matrices agree entry-for-entry within `abs_eps`.
fn pin_mat(label: &str, idx: usize, got: Mat3, want: Mat3, abs_eps: f32) {
    assert!(
        approx(got.xx, want.xx, abs_eps)
            && approx(got.yy, want.yy, abs_eps)
            && approx(got.zz, want.zz, abs_eps)
            && approx(got.xy, want.xy, abs_eps)
            && approx(got.xz, want.xz, abs_eps)
            && approx(got.yz, want.yz, abs_eps),
        "query {idx} {label}: gpu [{} {} {} {} {} {}] vs cpu [{} {} {} {} {} {}]",
        got.xx,
        got.yy,
        got.zz,
        got.xy,
        got.xz,
        got.yz,
        want.xx,
        want.yy,
        want.zz,
        want.xy,
        want.xz,
        want.yz
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

/// A pseudo-random position with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// A pseudo-random mass in `[0.25, 2.75)`, kept comfortably above
/// [`MASS_EPS`](prism_render_architecture::particle::inertia_tensor::MASS_EPS)
/// so every cloud is non-degenerate and both devices share the same guard
/// branch.
fn rand_mass(state: &mut u64) -> f32 {
    0.25 + lcg(state) * 2.5
}

/// A well-conditioned random cloud of `count` positive-mass points with moderate
/// coordinates, so the reordered-sum rounding stays far inside [`REDUCE_EPS`].
fn rand_cloud(state: &mut u64, count: usize) -> Vec<(Vec3, f32)> {
    (0..count)
        .map(|_| (rand_vec(state, 4.0), rand_mass(state)))
        .collect()
}

/// Pins the `GPU` reduction against the `CPU` golden for `points`: total mass,
/// centre of mass and the inertia tensor about that `COM` must all agree within
/// [`REDUCE_EPS`], and the `None` guard must fire on both for a sub-threshold
/// total mass.
fn pin_reduction(idx: usize, ctx: &GpuContext, gpu: &GpuInertiaTensor, points: &[(Vec3, f32)]) {
    let want_total = MassProperties::total_mass(points);
    let got_total = gpu.total_mass(ctx, points);
    pin_scalar("total_mass", idx, got_total, want_total, REDUCE_EPS);

    match (
        MassProperties::center_of_mass(points),
        gpu.center_of_mass(ctx, points),
    ) {
        (Some(want), Some(got)) => pin_vec("center_of_mass", idx, got, want, REDUCE_EPS),
        (None, None) => {}
        (want, got) => panic!("query {idx} center_of_mass guard mismatch: cpu {want:?} gpu {got:?}"),
    }

    match (MassProperties::of(points), gpu.mass_properties(ctx, points)) {
        (Some(want), Some(got)) => {
            pin_scalar("of.total_mass", idx, got.total_mass, want.total_mass, REDUCE_EPS);
            pin_vec("of.com", idx, got.com, want.com, REDUCE_EPS);
            pin_mat("of.inertia", idx, got.inertia, want.inertia, REDUCE_EPS);
        }
        (None, None) => {}
        (want, got) => panic!("query {idx} of guard mismatch: cpu {want:?} gpu {got:?}"),
    }
}

/// Pins one `GPU` [`BodyOpResult`] against the `CPU` golden for `query`: the raw
/// parallel-axis translate, the `inertia_about` re-reference and the merged
/// system must all agree within [`EPS`] / [`REL`].
fn pin_body(idx: usize, query: &BodyOpQuery, got: &BodyOpResult) {
    let want_translated = query.a.inertia.translate(query.a.total_mass, query.about);
    let want_about = query.a.inertia_about(query.about);
    let want_merged = MassProperties::merge(query.a, query.b);

    pin_mat("translated", idx, got.translated, want_translated, EPS);
    pin_mat("inertia_about", idx, got.inertia_about, want_about, EPS);
    pin_scalar(
        "merged.total_mass",
        idx,
        got.merged.total_mass,
        want_merged.total_mass,
        EPS,
    );
    pin_vec("merged.com", idx, got.merged.com, want_merged.com, EPS);
    pin_mat("merged.inertia", idx, got.merged.inertia, want_merged.inertia, EPS);
}

/// Dispatches `queries` on the `GPU` and pins every per-body result against the
/// reference, asserting the returned count matches the input count.
fn check_body(ctx: &GpuContext, gpu: &GpuInertiaTensor, queries: &[BodyOpQuery]) {
    let got = gpu.body_ops(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin_body(idx, query, result);
    }
}

/// Runs a single two-system merge on the `GPU` and returns the merged system
/// (the reference point is irrelevant to the merge, so the origin is used).
fn gpu_merge(ctx: &GpuContext, gpu: &GpuInertiaTensor, a: MassProperties, b: MassProperties) -> MassProperties {
    let got = gpu.body_ops(ctx, &[BodyOpQuery::new(a, b, Vec3::ZERO)]);
    got[0].merged
}

#[test]
fn empty_reduction_short_circuits_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    // An empty cloud short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized): no mass, no COM, no properties.
    assert!(approx(gpu.total_mass(&ctx, &[]), 0.0, EPS));
    assert!(gpu.center_of_mass(&ctx, &[]).is_none());
    assert!(gpu.mass_properties(&ctx, &[]).is_none());
}

#[test]
fn empty_body_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    assert!(
        gpu.body_ops(&ctx, &[]).is_empty(),
        "an empty batch produces no results"
    );
}

#[test]
fn single_point_cloud_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    // A single point is its own COM, so r = 0 everywhere and the inertia tensor
    // about the COM is exactly zero; the COM is the point and the mass is its
    // mass.
    let points = [(Vec3::new(5.0, -7.0, 2.0), 3.0)];
    pin_reduction(0, &ctx, &gpu, &points);
    let got = gpu.mass_properties(&ctx, &points).expect("single point has mass");
    pin_mat("single.inertia", 0, got.inertia, Mat3::ZERO, REDUCE_EPS);
}

#[test]
fn symmetric_cloud_centres_on_origin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    // Equal masses placed symmetrically about the origin: the COM is the origin
    // and the tensor is diagonal. The fixture pins the GPU against the reference
    // (which carries the exact expected values).
    let points = [
        (Vec3::new(-3.0, 0.0, 0.0), 1.0),
        (Vec3::new(3.0, 0.0, 0.0), 1.0),
        (Vec3::new(0.0, -2.0, 0.0), 1.0),
        (Vec3::new(0.0, 2.0, 0.0), 1.0),
        (Vec3::new(0.0, 0.0, -4.0), 1.0),
        (Vec3::new(0.0, 0.0, 4.0), 1.0),
    ];
    pin_reduction(0, &ctx, &gpu, &points);
    let got = gpu.center_of_mass(&ctx, &points).expect("symmetric cloud has mass");
    pin_vec("symmetric.com", 0, got, Vec3::ZERO, REDUCE_EPS);
}

#[test]
fn random_clouds_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    let mut state = 0x51a9_f00d_dead_beef_u64;
    // A sweep of well-conditioned clouds across one and several workgroups'
    // worth of points, pinning every reduced field element-for-element.
    for (idx, &count) in [1usize, 2, 3, 7, 16, 33, 64].iter().enumerate() {
        let points = rand_cloud(&mut state, count);
        pin_reduction(idx, &ctx, &gpu, &points);
    }
}

#[test]
fn parallel_axis_off_centre_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    // Build a cloud, then re-reference its tensor about an off-centre point via
    // both the raw parallel-axis translate and the inertia_about re-reference;
    // the GPU must reproduce both closed forms of the Huygens-Steiner shift.
    let cloud = [
        (Vec3::new(1.0, 2.0, -1.0), 1.5),
        (Vec3::new(-2.0, 0.5, 3.0), 2.0),
        (Vec3::new(0.0, -3.0, 1.0), 0.75),
    ];
    let a = MassProperties::of(&cloud).expect("cloud has mass");
    let about = Vec3::new(2.5, -1.5, 4.0);
    let query = BodyOpQuery::new(a, MassProperties::EMPTY, about);
    check_body(&ctx, &gpu, &[query]);
}

#[test]
fn merge_commutes_and_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    let a = MassProperties::of(&[
        (Vec3::new(1.0, 0.0, 0.0), 2.0),
        (Vec3::new(-1.0, 1.0, 0.0), 1.0),
    ])
    .expect("system a has mass");
    let b = MassProperties::of(&[
        (Vec3::new(3.0, -2.0, 1.0), 1.5),
        (Vec3::new(2.0, 2.0, -1.0), 0.5),
    ])
    .expect("system b has mass");

    // GPU merge in both operand orders must each match the reference and each
    // other: IEEE addition is commutative, so the two orders coincide.
    check_body(&ctx, &gpu, &[BodyOpQuery::new(a, b, Vec3::ZERO)]);
    check_body(&ctx, &gpu, &[BodyOpQuery::new(b, a, Vec3::ZERO)]);
    let ab = gpu_merge(&ctx, &gpu, a, b);
    let ba = gpu_merge(&ctx, &gpu, b, a);
    pin_scalar("commute.total_mass", 0, ab.total_mass, ba.total_mass, EPS);
    pin_vec("commute.com", 0, ab.com, ba.com, EPS);
    pin_mat("commute.inertia", 0, ab.inertia, ba.inertia, EPS);
}

#[test]
fn merge_associates_within_tolerance() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    let a = MassProperties::of(&[(Vec3::new(0.0, 0.0, 0.0), 1.0), (Vec3::new(2.0, 0.0, 0.0), 1.0)])
        .expect("system a has mass");
    let b = MassProperties::of(&[(Vec3::new(0.0, 3.0, 0.0), 2.0), (Vec3::new(1.0, 1.0, 1.0), 0.5)])
        .expect("system b has mass");
    let c = MassProperties::of(&[(Vec3::new(-2.0, -1.0, 4.0), 1.5), (Vec3::new(3.0, 3.0, 3.0), 1.0)])
        .expect("system c has mass");

    // Each individual GPU merge matches the reference's closed form tightly.
    let gpu_ab = gpu_merge(&ctx, &gpu, a, b);
    let gpu_bc = gpu_merge(&ctx, &gpu, b, c);
    let cpu_ab = MassProperties::merge(a, b);
    let cpu_bc = MassProperties::merge(b, c);
    pin_mat("ab.inertia", 0, gpu_ab.inertia, cpu_ab.inertia, EPS);
    pin_mat("bc.inertia", 0, gpu_bc.inertia, cpu_bc.inertia, EPS);

    // Left- and right-associated GPU merges agree up to float rounding, matching
    // the reference's documented "associative up to float rounding" contract.
    let left = gpu_merge(&ctx, &gpu, gpu_ab, c);
    let right = gpu_merge(&ctx, &gpu, a, gpu_bc);
    pin_scalar("assoc.total_mass", 0, left.total_mass, right.total_mass, EPS);
    pin_vec("assoc.com", 0, left.com, right.com, EPS);
    pin_mat("assoc.inertia", 0, left.inertia, right.inertia, REDUCE_EPS);

    // Both groupings also match the reference's own associated results.
    let cpu_left = MassProperties::merge(cpu_ab, c);
    let cpu_right = MassProperties::merge(a, cpu_bc);
    pin_mat("assoc.left.inertia", 0, left.inertia, cpu_left.inertia, EPS);
    pin_mat("assoc.right.inertia", 0, right.inertia, cpu_right.inertia, EPS);
}

#[test]
fn zero_mass_merge_collapses_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    // Two zero-mass systems trip the MASS_EPS guard and merge to EMPTY: zero
    // mass, origin COM, zero tensor on both devices.
    let empty = MassProperties::EMPTY;
    let query = BodyOpQuery::new(empty, empty, Vec3::new(1.0, 2.0, 3.0));
    let got = gpu.body_ops(&ctx, &[query]);
    pin_scalar("empty.total_mass", 0, got[0].merged.total_mass, 0.0, EPS);
    pin_vec("empty.com", 0, got[0].merged.com, Vec3::ZERO, EPS);
    pin_mat("empty.inertia", 0, got[0].merged.inertia, Mat3::ZERO, EPS);
}

#[test]
fn random_body_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuInertiaTensor::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A batch mixing many random bodies, dispatched together so the per-thread
    // indexing and the contiguous storage layout are both exercised, then pinned
    // element-for-element. Each system is a well-conditioned small cloud.
    let mut queries = Vec::new();
    for _ in 0..64 {
        let a = MassProperties::of(&rand_cloud(&mut state, 3)).expect("system a has mass");
        let b = MassProperties::of(&rand_cloud(&mut state, 2)).expect("system b has mass");
        queries.push(BodyOpQuery::new(a, b, rand_vec(&mut state, 3.0)));
    }
    check_body(&ctx, &gpu, &queries);
}
