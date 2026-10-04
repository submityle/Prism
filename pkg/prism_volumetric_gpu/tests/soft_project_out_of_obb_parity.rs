//! Real-device parity for the oriented-bounding-box ejection twin:
//! [`GpuSoftProjectOutOfObb`](prism_volumetric_gpu::soft_project_out_of_obb::GpuSoftProjectOutOfObb)
//! must reproduce the `CPU` golden `project_out_of_obb` of
//! `prism_physics_core::soft::collision::body`. For each particle the reference
//! pushes an interior point out to the nearest face of a rigid oriented box,
//! measured in the box's own local frame; a point with no interior to leave
//! (an all non-positive half extent) or one already on or outside a slab is a
//! no-op.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the all non-positive guard, the inverse-rotation transform into the box
//! frame, the per-slab outside test, the least-penetration axis selection with
//! the `x` then `y` then `z` tie-break and the signed-face snap, and the
//! rotate-and-recenter back to world space — written out directly so the test
//! never imports `prism_render_architecture` or `prism_physics_core`. The
//! quaternion rotation uses the same pure-arithmetic Rodrigues form the kernel
//! does, which equals `q * v` for a unit quaternion, so both sides agree while
//! the oracle stays independent; every fixture and sweep sample feeds a
//! normalized quaternion. The oracle mirrors the reference branch for branch,
//! so a passing comparison is evidence the ported kernel took the same
//! degenerate / outside / interior branch, not merely that the shader compiled.
//!
//! The fixtures cover each branch the kernel must honor: an all non-positive
//! half extent with no interior, a point already on or outside a slab, an
//! interior point whose least-penetration face is the `x`, `y` or `z` face in
//! turn, a batch of two or more elements that validates the `std430` stride,
//! and an empty batch the host short-circuits with no dispatch. A sweep over
//! random normalized orientations and positive half extents follows, rejecting
//! marginal configurations (where a tiny perturbation would flip the inside /
//! outside branch or the chosen least-penetration face) so the branch choice
//! agrees on both sides despite any last-bit difference.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every continuous path threads through multiplies, adds, cross products and
//! dot products, so `CPU` and `GPU` evaluate the same closed form but need not
//! be bit-exact (a `GPU` may contract a multiply-add). Each continuous output
//! is compared with `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR =
//! 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body::project_out_of_obb`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_project_out_of_obb::SoftProjectOutOfObbResult;
use prism_volumetric_gpu::soft_project_out_of_obb::{
    GpuSoftProjectOutOfObb, SoftProjectOutOfObbQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale_ref = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale_ref <= REL_EPS
}

/// Dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Vector difference `a - b`.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Vector sum `a + b`.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales a vector.
fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Per-component absolute value.
fn vabs(a: [f32; 3]) -> [f32; 3] {
    [a[0].abs(), a[1].abs(), a[2].abs()]
}

/// Rotates `v` by the quaternion `rot` (stored `[x, y, z, w]`) with the
/// pure-arithmetic Rodrigues form, which equals `rot * v` for a unit
/// quaternion — the same form the kernel evaluates.
fn quat_rotate(rot: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let axis = [rot[0], rot[1], rot[2]];
    let t = add(cross(axis, v), scale(v, rot[3]));
    add(v, scale(cross(axis, t), 2.0))
}

/// Conjugate (inverse rotation for a unit quaternion).
fn quat_conj(rot: [f32; 4]) -> [f32; 4] {
    [-rot[0], -rot[1], -rot[2], rot[3]]
}

/// Independent host re-implementation of `project_out_of_obb` for a single
/// particle. Returns the resolved (or echoed) position and the discrete
/// validity flag, replicated branch for branch.
fn oracle(q: &SoftProjectOutOfObbQuery) -> ([f32; 3], u32) {
    let pos = q.pos;
    let center = q.center;
    let rot = q.orientation;
    let he = q.half_extents;

    // No positive extent means there is no interior to project out of.
    if he[0] <= 0.0 && he[1] <= 0.0 && he[2] <= 0.0 {
        return (pos, 0);
    }

    // World -> local via the inverse (conjugate) rotation.
    let local = quat_rotate(quat_conj(rot), sub(pos, center));
    let al = vabs(local);
    // On or outside any slab => already outside the solid.
    if al[0] >= he[0] || al[1] >= he[1] || al[2] >= he[2] {
        return (pos, 0);
    }

    // Interior: push to the face of least penetration.
    let pen = sub(he, al);
    let mut local_out = local;
    if pen[0] <= pen[1] && pen[0] <= pen[2] {
        local_out[0] = if local[0] >= 0.0 { he[0] } else { -he[0] };
    } else if pen[1] <= pen[2] {
        local_out[1] = if local[1] >= 0.0 { he[1] } else { -he[1] };
    } else {
        local_out[2] = if local[2] >= 0.0 { he[2] } else { -he[2] };
    }

    (add(center, quat_rotate(rot, local_out)), 1)
}

/// Asserts the device result matches the host oracle for one query: the
/// `valid` flag exactly and every continuous output within tolerance. A no-op
/// echoes the input position, so the comparison also holds there.
fn assert_parity(gpu: &SoftProjectOutOfObbResult, q: &SoftProjectOutOfObbQuery, label: &str) {
    let (out, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid mismatch query={q:?}");
    for axis in 0..3 {
        assert!(
            close(gpu.out[axis], out[axis]),
            "{label}: out[{axis}] mismatch gpu={} cpu={} query={q:?}",
            gpu.out[axis],
            out[axis]
        );
    }
}

/// Normalizes a quaternion, matching the unit-quaternion assumption the
/// Rodrigues rotation relies on.
fn normalize_quat(raw: [f32; 4]) -> [f32; 4] {
    let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]).sqrt();
    [raw[0] / len, raw[1] / len, raw[2] / len, raw[3] / len]
}

/// A general non-axis-aligned unit orientation reused across fixtures.
fn tilted_quat() -> [f32; 4] {
    normalize_quat([0.2, -0.35, 0.1, 0.9])
}

/// Builds a world-space query whose particle lies at a chosen local point
/// inside (or relative to) the box, by rotating the local offset back out.
fn query_from_local(
    local: [f32; 3],
    center: [f32; 3],
    rot: [f32; 4],
    he: [f32; 3],
) -> SoftProjectOutOfObbQuery {
    let pos = add(center, quat_rotate(rot, local));
    SoftProjectOutOfObbQuery::new(pos, center, rot, he)
}

#[test]
fn all_non_positive_he_echoes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfObb::new(&ctx);
    // No interior: echo the position, valid = 0.
    let q = SoftProjectOutOfObbQuery::new(
        [0.3, -0.2, 0.1],
        [0.0, 0.0, 0.0],
        tilted_quat(),
        [0.0, -1.0, -0.5],
    );
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].valid, 0, "all non-positive half extents are inert");
    assert_parity(&r[0], &q, "all_non_positive_he_echoes");
}

#[test]
fn outside_slab_echoes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfObb::new(&ctx);
    let center = [0.5, -0.3, 0.2];
    let rot = tilted_quat();
    let he = [0.4, 0.6, 0.5];
    // Local x well beyond the slab (|local.x| >= he.x) => already outside.
    let q = query_from_local([0.9, 0.1, -0.2], center, rot, he);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 0, "point outside a slab is a no-op");
    assert_parity(&r[0], &q, "outside_slab_echoes");
}

#[test]
fn interior_pushes_x() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfObb::new(&ctx);
    let center = [0.1, 0.2, -0.1];
    let rot = tilted_quat();
    let he = [0.5, 0.6, 0.7];
    // Closest to the +x face (least penetration on x by a clear margin).
    let local = [0.45, 0.1, -0.15];
    let q = query_from_local(local, center, rot, he);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 1, "interior point is projected");
    assert_parity(&r[0], &q, "interior_pushes_x");
    // Verify the snapped local x reaches the +x face.
    let back = quat_rotate(quat_conj(rot), sub(r[0].out, center));
    assert!(
        close(back[0], he[0]),
        "projected local x should reach +x face: {back:?}"
    );
}

#[test]
fn interior_pushes_y() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfObb::new(&ctx);
    let center = [-0.2, 0.0, 0.3];
    let rot = tilted_quat();
    let he = [0.6, 0.5, 0.7];
    // Closest to the -y face.
    let local = [0.1, -0.45, 0.2];
    let q = query_from_local(local, center, rot, he);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 1, "interior point is projected");
    assert_parity(&r[0], &q, "interior_pushes_y");
    let back = quat_rotate(quat_conj(rot), sub(r[0].out, center));
    assert!(
        close(back[1], -he[1]),
        "projected local y should reach -y face: {back:?}"
    );
}

#[test]
fn interior_pushes_z() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfObb::new(&ctx);
    let center = [0.4, -0.1, -0.2];
    let rot = tilted_quat();
    let he = [0.7, 0.6, 0.5];
    // Closest to the +z face.
    let local = [0.15, -0.1, 0.45];
    let q = query_from_local(local, center, rot, he);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 1, "interior point is projected");
    assert_parity(&r[0], &q, "interior_pushes_z");
    let back = quat_rotate(quat_conj(rot), sub(r[0].out, center));
    assert!(
        close(back[2], he[2]),
        "projected local z should reach +z face: {back:?}"
    );
}

#[test]
fn batch_of_two_or_more_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfObb::new(&ctx);
    let center = [0.1, 0.2, -0.1];
    let rot = tilted_quat();
    let he = [0.5, 0.6, 0.7];
    // Mixed batch exercises the std430 stride across all branches at once.
    let queries = [
        SoftProjectOutOfObbQuery::new(
            [0.3, -0.2, 0.1],
            [0.0, 0.0, 0.0],
            tilted_quat(),
            [0.0, -1.0, -0.5],
        ),
        query_from_local([0.9, 0.1, -0.2], center, rot, he),
        query_from_local([0.45, 0.1, -0.15], center, rot, he),
        query_from_local([0.1, -0.45, 0.2], center, rot, he),
        query_from_local([0.15, -0.1, 0.45], center, rot, he),
    ];
    let r = gpu.evaluate(&ctx, &queries);
    assert_eq!(r.len(), queries.len(), "one result per query");
    for (res, q) in r.iter().zip(queries.iter()) {
        assert_parity(res, q, "batch_of_two_or_more_validates_stride");
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfObb::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
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

/// Rejects marginal configurations where a last-bit difference could flip a
/// branch: the point must be comfortably interior on every slab (so the inside
/// / outside test cannot flip) and the three penetration depths must be
/// pairwise separated (so the chosen least-penetration face cannot flip).
fn well_conditioned(local: [f32; 3], he: [f32; 3]) -> bool {
    let al = vabs(local);
    // Interior margin on every slab.
    for axis in 0..3 {
        if he[axis] - al[axis] < 2.0e-2 {
            return false;
        }
        if al[axis] < 1.0e-2 {
            // Keep the signed-face choice unambiguous (local far from the center
            // plane of the slab) so sign(0) ambiguity never arises.
            return false;
        }
    }
    // Pairwise-separated penetration depths so the tie-break never flips.
    let pen = [he[0] - al[0], he[1] - al[1], he[2] - al[2]];
    (pen[0] - pen[1]).abs() >= 1.0e-2
        && (pen[0] - pen[2]).abs() >= 1.0e-2
        && (pen[1] - pen[2]).abs() >= 1.0e-2
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftProjectOutOfObb::new(&ctx);
    let mut rng = Lcg::new(0x51_A3_C7_0D);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        // Random normalized orientation (reject near-zero norm).
        let raw = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let norm2 = raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3];
        if norm2 < 1.0e-2 {
            continue;
        }
        let rot = normalize_quat(raw);
        let center = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        let he = [
            rng.next_range(0.3, 1.2),
            rng.next_range(0.3, 1.2),
            rng.next_range(0.3, 1.2),
        ];
        // Place the particle at a well-conditioned interior local point.
        let local = [
            rng.next_range(-0.95, 0.95) * he[0],
            rng.next_range(-0.95, 0.95) * he[1],
            rng.next_range(-0.95, 0.95) * he[2],
        ];
        if !well_conditioned(local, he) {
            continue;
        }
        queries.push(query_from_local(local, center, rot, he));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    let mut projected = 0u32;
    for (q, r) in queries.iter().zip(results.iter()) {
        let (_, valid) = oracle(q);
        if valid == 1 {
            projected += 1;
        }
        assert_parity(r, q, "random_sweep_matches_oracle");
    }
    // The sweep is constructed to project every interior point; guard against a
    // degenerate sweep that would weakly test parity.
    assert!(projected > 0, "sweep produced no projections");
}
