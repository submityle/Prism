//! Real-device parity for the sphere-versus-capsule contact twin:
//! [`GpuSphereCapsuleContact`](prism_volumetric_gpu::sphere_capsule_contact::GpuSphereCapsuleContact)
//! must reproduce the `CPU` golden `sphere_capsule` of
//! `prism_physics_core::collide::primitives`, which projects the sphere centre
//! onto the capsule core segment, tests the penetration against the sum of
//! radii, and emits the contact normal, the two witness points, the penetration
//! depth and a validity flag.
//!
//! The oracle here is an independent re-implementation of that closed form — the
//! segment projection (`t = clamp(dot(center - seg0, ab) / dot(ab, ab), 0, 1)`
//! with a degenerate zero-length segment collapsing to `seg0`), the signed
//! penetration `sum - distance`, the separation rejection, the guarded contact
//! normal (`delta / distance`, falling back to `(1, 0, 0)` on the axis) and the
//! two witness points — written in pure `f32` without `glam`, so the test never
//! imports `prism_physics_core` or `prism_render_architecture`.
//!
//! The fixtures cover a clean separation (`valid = 0`, all-zero output), a sphere
//! touching the middle of the segment (`t` interior), a sphere past an endpoint
//! (`t` clamps to `0` or `1`), a degenerate zero-length segment (sphere-versus-
//! sphere), a sphere centre on the axis (`distance <= 1e-6`, normal falls back to
//! `(1, 0, 0)`), and a batch of at least two distinct queries that catches any
//! `std430` stride aliasing. A reject-sampled sweep over random pairs follows,
//! plus an empty batch the host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every continuous channel threads through a dot product, two guarded divides,
//! a square root and several multiply-adds, so `CPU` and `GPU` evaluate the same
//! closed form but need not be bit-exact (a device may fuse a multiply-add the
//! scalar reference leaves separate). The continuous comparison is
//! `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly, and a separated pair has every continuous
//! channel equal to zero.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collide::primitives`；无第三方
//! 引擎源码或衍生代码。

use prism_volumetric_gpu::sphere_capsule_contact::{
    GpuSphereCapsuleContact, SphereCapsuleContactQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Contact tolerance: pairs separated by more than this are rejected, matching
/// the golden `CONTACT_TOLERANCE`.
const CONTACT_TOLERANCE: f32 = 1.0e-4;
/// Geometric epsilon below which a length is treated as degenerate, matching the
/// golden `GEOMETRIC_EPS`.
const GEOMETRIC_EPS: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// The resolved oracle answer: the contact normal, the two witness points, the
/// penetration depth and the validity flag.
struct Expected {
    normal: [f32; 3],
    point_a: [f32; 3],
    point_b: [f32; 3],
    penetration: f32,
    valid: u32,
}

/// The independent oracle for one query, mirroring the on-device kernel's branch
/// structure exactly, including the all-zero echo on a clean separation. Written
/// in pure `f32` with no `glam` dependency.
fn oracle(q: &SphereCapsuleContactQuery) -> Expected {
    let center = [q.scx, q.scy, q.scz];
    let seg0 = [q.s0x, q.s0y, q.s0z];
    let seg1 = [q.s1x, q.s1y, q.s1z];

    // Closest point on the capsule core segment to the sphere centre.
    let ab = [seg1[0] - seg0[0], seg1[1] - seg0[1], seg1[2] - seg0[2]];
    let len_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let on_axis = if len_sq < GEOMETRIC_EPS {
        // Degenerate zero-length segment collapses to seg0 (sphere-sphere).
        seg0
    } else {
        let cs = [
            center[0] - seg0[0],
            center[1] - seg0[1],
            center[2] - seg0[2],
        ];
        let raw_t = (cs[0] * ab[0] + cs[1] * ab[1] + cs[2] * ab[2]) / len_sq;
        let t = raw_t.clamp(0.0, 1.0);
        [
            seg0[0] + ab[0] * t,
            seg0[1] + ab[1] * t,
            seg0[2] + ab[2] * t,
        ]
    };

    let delta = [
        on_axis[0] - center[0],
        on_axis[1] - center[1],
        on_axis[2] - center[2],
    ];
    let distance = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    let sum = q.sphere_radius + q.capsule_radius;
    let penetration = sum - distance;

    // Clean separation beyond tolerance: no contact, all-zero output.
    if penetration < -CONTACT_TOLERANCE {
        return Expected {
            normal: [0.0; 3],
            point_a: [0.0; 3],
            point_b: [0.0; 3],
            penetration: 0.0,
            valid: 0,
        };
    }

    // Contact normal: delta / distance, falling back to (1, 0, 0) on the axis.
    let normal = if distance > GEOMETRIC_EPS {
        [
            delta[0] / distance,
            delta[1] / distance,
            delta[2] / distance,
        ]
    } else {
        [1.0, 0.0, 0.0]
    };
    let point_a = [
        center[0] + normal[0] * q.sphere_radius,
        center[1] + normal[1] * q.sphere_radius,
        center[2] + normal[2] * q.sphere_radius,
    ];
    let point_b = [
        on_axis[0] - normal[0] * q.capsule_radius,
        on_axis[1] - normal[1] * q.capsule_radius,
        on_axis[2] - normal[2] * q.capsule_radius,
    ];

    Expected {
        normal,
        point_a,
        point_b,
        penetration,
        valid: 1,
    }
}

/// Dispatches one query and asserts every resolved channel plus the validity
/// flag.
fn assert_parity(ctx: &GpuContext, gpu: &GpuSphereCapsuleContact, q: SphereCapsuleContactQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    assert_result(&got[0], &q);
}

/// Asserts parity for a whole batch, so the shared dispatch exercises the
/// `std430` stride.
fn assert_batch(
    ctx: &GpuContext,
    gpu: &GpuSphereCapsuleContact,
    queries: &[SphereCapsuleContactQuery],
) {
    let results = gpu.evaluate(ctx, queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        assert_result(r, q);
    }
}

/// Compares one `GPU` result against the oracle for the same query.
fn assert_result(
    r: &prism_volumetric_gpu::sphere_capsule_contact::SphereCapsuleContactResult,
    q: &SphereCapsuleContactQuery,
) {
    let e = oracle(q);
    assert_eq!(r.valid, e.valid, "valid flag mismatch: query={q:?}");
    let got = [
        r.nx,
        r.ny,
        r.nz,
        r.pax,
        r.pay,
        r.paz,
        r.pbx,
        r.pby,
        r.pbz,
        r.penetration,
    ];
    let want = [
        e.normal[0],
        e.normal[1],
        e.normal[2],
        e.point_a[0],
        e.point_a[1],
        e.point_a[2],
        e.point_b[0],
        e.point_b[1],
        e.point_b[2],
        e.penetration,
    ];
    let labels = [
        "nx",
        "ny",
        "nz",
        "pax",
        "pay",
        "paz",
        "pbx",
        "pby",
        "pbz",
        "penetration",
    ];
    for ((g, w), label) in got.iter().zip(want.iter()).zip(labels.iter()) {
        assert!(
            close(*g, *w),
            "{label} mismatch: gpu={g} cpu={w} query={q:?}"
        );
    }
}

#[test]
fn separated_pair_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereCapsuleContact::new(&ctx);
    // A sphere far above the capsule axis: distance (5) exceeds the sum of radii
    // (1) by well over the tolerance, so the pair is cleanly separated and every
    // continuous channel is zero (valid = 0).
    let q = SphereCapsuleContactQuery::new(
        [0.0, 5.0, 0.0],
        0.5,
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
    );
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].valid, 0, "separated pair must be invalid");
    assert_eq!(got[0].penetration, 0.0, "separated pair zeroes penetration");
}

#[test]
fn sphere_touches_mid_segment() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereCapsuleContact::new(&ctx);
    // A sphere centred above the segment midpoint (t = 0.5): distance 0.8 is just
    // inside the sum of radii (1.0), so the pair is in contact. The golden normal
    // points from the sphere toward the capsule axis (delta = on_axis - center),
    // so for a sphere above the axis it points straight down along -y.
    let q = SphereCapsuleContactQuery::new(
        [0.0, 0.8, 0.0],
        0.5,
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
    );
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].valid, 1, "touching pair must be valid");
    assert!(
        got[0].ny < 0.0,
        "normal must point down along -y (sphere -> axis): {:?}",
        got[0]
    );
}

#[test]
fn sphere_past_endpoint_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereCapsuleContact::new(&ctx);
    // The projection parameter runs past the far endpoint, so t clamps to 1 and
    // the closest axis point is seg1; the sum of radii (1.3) still exceeds the
    // distance, so the pair is in contact.
    let q = SphereCapsuleContactQuery::new(
        [2.0, 0.5, 0.0],
        0.8,
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
    );
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].valid, 1, "clamped-endpoint contact must be valid");
}

#[test]
fn degenerate_segment_is_sphere_sphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereCapsuleContact::new(&ctx);
    // A zero-length core segment (seg0 == seg1): len_sq < 1e-6, so the capsule
    // collapses to a sphere at seg0 and the test degenerates to sphere-sphere.
    let q =
        SphereCapsuleContactQuery::new([0.0, 0.8, 0.0], 0.5, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5);
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].valid, 1, "degenerate-segment contact must be valid");
}

#[test]
fn centre_on_axis_uses_fallback_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereCapsuleContact::new(&ctx);
    // The sphere centre sits exactly on the segment (distance = 0 <= 1e-6): there
    // is no defined contact direction, so the normal falls back to (1, 0, 0).
    let q = SphereCapsuleContactQuery::new(
        [0.0, 0.0, 0.0],
        0.5,
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        0.5,
    );
    assert_parity(&ctx, &gpu, q);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got[0].valid, 1, "on-axis contact must be valid");
    assert!(
        close(got[0].nx, 1.0) && close(got[0].ny, 0.0) && close(got[0].nz, 0.0),
        "on-axis normal must fall back to (1, 0, 0): {:?}",
        got[0]
    );
}

#[test]
fn batch_stride_reads_non_aliased_slots() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereCapsuleContact::new(&ctx);
    // A batch of several distinct queries exercises the std430 query/result
    // stride: every slot must read and write its own non-aliased data.
    let queries = [
        SphereCapsuleContactQuery::new(
            [0.0, 5.0, 0.0],
            0.5,
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
        ),
        SphereCapsuleContactQuery::new(
            [0.0, 0.8, 0.0],
            0.5,
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
        ),
        SphereCapsuleContactQuery::new(
            [2.0, 0.5, 0.0],
            0.8,
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
        ),
        SphereCapsuleContactQuery::new([0.0, 0.8, 0.0], 0.5, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5),
        SphereCapsuleContactQuery::new(
            [0.0, 0.0, 0.0],
            0.5,
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.5,
        ),
    ];
    assert_batch(&ctx, &gpu, &queries);
}

/// A small deterministic linear-congruential generator so the sweep needs no
/// external randomness. Constants are the Numerical Recipes values.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A `[0, 1)` fraction built from the top bits, keeping the fixture pure
    /// integer host-side with no transcendental call.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A `[lo, hi)` fraction.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereCapsuleContact::new(&ctx);
    let mut rng = Lcg::new(0x51_A3_7E_09);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let q = SphereCapsuleContactQuery::new(
            [
                rng.next_range(-2.0, 2.0),
                rng.next_range(-2.0, 2.0),
                rng.next_range(-2.0, 2.0),
            ],
            rng.next_range(0.2, 1.0),
            [
                rng.next_range(-2.0, 2.0),
                rng.next_range(-2.0, 2.0),
                rng.next_range(-2.0, 2.0),
            ],
            [
                rng.next_range(-2.0, 2.0),
                rng.next_range(-2.0, 2.0),
                rng.next_range(-2.0, 2.0),
            ],
            rng.next_range(0.2, 1.0),
        );

        // Reject-sample the geometric knees so CPU and GPU take the same branch:
        // keep the segment well away from zero length, the sphere centre well off
        // the axis, and the penetration well away from the -CONTACT_TOLERANCE
        // contact threshold.
        let ab = [q.s1x - q.s0x, q.s1y - q.s0y, q.s1z - q.s0z];
        let len_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        if len_sq < 1.0e-2 {
            continue;
        }
        let cs = [q.scx - q.s0x, q.scy - q.s0y, q.scz - q.s0z];
        let t = ((cs[0] * ab[0] + cs[1] * ab[1] + cs[2] * ab[2]) / len_sq).clamp(0.0, 1.0);
        let on_axis = [q.s0x + ab[0] * t, q.s0y + ab[1] * t, q.s0z + ab[2] * t];
        let delta = [on_axis[0] - q.scx, on_axis[1] - q.scy, on_axis[2] - q.scz];
        let distance = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if distance < 1.0e-2 {
            continue;
        }
        let penetration = q.sphere_radius + q.capsule_radius - distance;
        if (penetration + CONTACT_TOLERANCE).abs() < 1.0e-2 {
            continue;
        }
        queries.push(q);
    }

    assert_batch(&ctx, &gpu, &queries);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphereCapsuleContact::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
