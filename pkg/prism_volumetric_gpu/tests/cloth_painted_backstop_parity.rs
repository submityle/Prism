//! Real-device parity for the cloth painted-backstop twin:
//! [`GpuClothPaintedBackstop`](prism_volumetric_gpu::cloth_painted_backstop::GpuClothPaintedBackstop)
//! must reproduce the `CPU` golden `apply_painted_backstop` of
//! `prism_render_architecture::cloth::painted`, composed with the
//! `PaintedConstraint::clamped` backstop clamp and `Vec3::normalize_or_zero`.
//! For each vertex the reference sinks a cushion sphere one radius behind the
//! skinned anchor along its outward normal and, when the vertex is caught
//! inside that sphere, projects it radially back onto the surface; every other
//! case is a no-op.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the `NaN -> 0` / `max(.., 0)` backstop clamp, the `normalize_or_zero` guard
//! on the anchor normal, the cushion-centre placement, the `dist_sq >= radius^2`
//! outside test and the radial / normal-fallback projection — written out
//! directly so the test never imports `prism_render_architecture` or
//! `prism_physics_core`. It mirrors the reference branch for branch, so a
//! passing comparison is evidence the ported kernel took the same pinned /
//! disabled / degenerate-normal / outside-the-sphere branch, not merely that
//! the shader compiled.
//!
//! The fixtures cover each branch the kernel must honor: a vertex inside the
//! cushion projected to the surface, a vertex outside left untouched, a pinned
//! vertex skipped, a non-positive and a `NaN` backstop skipped, a zero anchor
//! normal skipped, a vertex exactly at the cushion centre taking the
//! normal-direction fallback, a vertex exactly on the surface left untouched, a
//! batch of two or more elements that validates the `std430` stride, and an
//! empty batch the host short-circuits with no dispatch. A sweep over random
//! configurations follows, rejecting marginal geometry (where a tiny
//! perturbation would flip the `valid` flag or shift the projected position
//! across the `dist_sq == radius^2` knee) so the branch choice agrees on both
//! sides despite any last-bit difference.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every path threads through multiplies, adds, a guarded division and a
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). Each continuous position
//! component is compared with `abs_diff <= 1e-4 || rel_diff <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::cloth::painted::apply_painted_backstop`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cloth_painted_backstop::{
    ClothPaintedBackstopQuery, GpuClothPaintedBackstop,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;
/// Squared-length floor matching the reference cloth `EPS_LEN_SQ`, used to
/// reject a degenerate anchor normal and to detect a vertex at the centre.
const EPS_LEN_SQ: f32 = 1.0e-12;

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

/// Independent host re-implementation of `apply_painted_backstop` for a single
/// vertex, composed with `PaintedConstraint::clamped` (`NaN -> 0`, otherwise
/// `max(.., 0)`) and `Vec3::normalize_or_zero`. Returns the resolved position
/// and the discrete validity flag, replicated branch for branch.
fn oracle(q: &ClothPaintedBackstopQuery) -> ([f32; 3], u32) {
    let particle = q.particle_pos;
    let anchor = q.anchor_pos;
    let anchor_normal = q.anchor_normal;

    // PaintedConstraint::clamped().backstop: a NaN collapses to 0, otherwise
    // the authored value is floored at 0.
    let bs = if q.backstop.is_nan() {
        0.0
    } else {
        q.backstop.max(0.0)
    };

    if q.pinned != 0 {
        return (particle, 0);
    }
    if bs <= 0.0 {
        return (particle, 0);
    }

    // Vec3::normalize_or_zero guard: a (near-)zero normal disables the pass.
    let nlen_sq = dot(anchor_normal, anchor_normal);
    if nlen_sq <= EPS_LEN_SQ {
        return (particle, 0);
    }
    let inv = 1.0 / nlen_sq.sqrt();
    let normal = scale(anchor_normal, inv);

    let center = sub(anchor, scale(normal, bs));
    let rel = sub(particle, center);
    let dist_sq = dot(rel, rel);
    let radius = bs;
    if dist_sq >= radius * radius {
        return (particle, 0);
    }

    if dist_sq > EPS_LEN_SQ {
        let dist = dist_sq.sqrt();
        (add(center, scale(rel, radius / dist)), 1)
    } else {
        // Degenerate: the vertex sits at the cushion centre; push it out along
        // the anchor normal to the nearest surface point.
        (add(center, scale(normal, radius)), 1)
    }
}

/// Asserts the device result matches the host oracle for one query: the
/// `valid` flag exactly and, when valid, every position component within
/// tolerance. A no-op echoes the input position bit-for-bit, so the position
/// comparison also holds there.
fn assert_parity(
    gpu: &prism_volumetric_gpu::cloth_painted_backstop::ClothPaintedBackstopResult,
    q: &ClothPaintedBackstopQuery,
    label: &str,
) {
    let (pos, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid mismatch query={q:?}");
    for axis in 0..3 {
        assert!(
            close(gpu.position[axis], pos[axis]),
            "{label}: position[{axis}] mismatch gpu={} cpu={} query={q:?}",
            gpu.position[axis],
            pos[axis]
        );
    }
}

#[test]
fn inside_projects_to_surface() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPaintedBackstop::new(&ctx);
    // Anchor at the origin, outward normal +y, backstop 1 => cushion centre at
    // (0, -1, 0). The vertex at (0, -0.5, 0) is inside and projects to the
    // surface point (0, 0, 0).
    let q =
        ClothPaintedBackstopQuery::new([0.0, -0.5, 0.0], 0, [0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0]);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].valid, 1, "vertex inside the cushion is projected");
    assert_parity(&r[0], &q, "inside_projects_to_surface");
}

#[test]
fn outside_cushion_is_untouched() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPaintedBackstop::new(&ctx);
    // Same cushion, but a vertex far outside the sphere is left untouched.
    let q =
        ClothPaintedBackstopQuery::new([5.0, 5.0, 5.0], 0, [0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0]);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 0, "vertex outside the cushion is a no-op");
    assert_parity(&r[0], &q, "outside_cushion_is_untouched");
}

#[test]
fn pinned_is_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPaintedBackstop::new(&ctx);
    // Would be inside the cushion, but the pinned flag disables the pass.
    let q =
        ClothPaintedBackstopQuery::new([0.0, -0.5, 0.0], 1, [0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0]);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 0, "pinned vertex is skipped");
    assert_parity(&r[0], &q, "pinned_is_skipped");
}

#[test]
fn non_positive_backstop_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPaintedBackstop::new(&ctx);
    // A zero, a negative and a NaN backstop all clamp to <= 0 and skip the pass.
    let queries = [
        ClothPaintedBackstopQuery::new([0.0, -0.5, 0.0], 0, [0.0, 0.0, 0.0], 0.0, [0.0, 1.0, 0.0]),
        ClothPaintedBackstopQuery::new([0.0, -0.5, 0.0], 0, [0.0, 0.0, 0.0], -2.0, [0.0, 1.0, 0.0]),
        ClothPaintedBackstopQuery::new(
            [0.0, -0.5, 0.0],
            0,
            [0.0, 0.0, 0.0],
            f32::NAN,
            [0.0, 1.0, 0.0],
        ),
    ];
    let r = gpu.evaluate(&ctx, &queries);
    for (res, q) in r.iter().zip(queries.iter()) {
        assert_eq!(res.valid, 0, "non-positive backstop is skipped: {q:?}");
        assert_parity(res, q, "non_positive_backstop_skipped");
    }
}

#[test]
fn zero_normal_skipped() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPaintedBackstop::new(&ctx);
    // A zero-length anchor normal cannot be normalized, so the pass is skipped.
    let q =
        ClothPaintedBackstopQuery::new([0.0, -0.5, 0.0], 0, [0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0]);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 0, "degenerate normal is skipped");
    assert_parity(&r[0], &q, "zero_normal_skipped");
}

#[test]
fn vertex_at_center_pushes_along_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPaintedBackstop::new(&ctx);
    // Anchor origin, normal +y, backstop 1 => centre (0, -1, 0). A vertex
    // exactly at the centre takes the normal-direction fallback, landing on the
    // anchor (0, 0, 0).
    let q =
        ClothPaintedBackstopQuery::new([0.0, -1.0, 0.0], 0, [0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0]);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 1, "vertex at the centre is pushed out");
    assert_parity(&r[0], &q, "vertex_at_center_pushes_along_normal");
}

#[test]
fn on_boundary_not_pushed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPaintedBackstop::new(&ctx);
    // Centre (0, -1, 0), radius 1. The vertex (1, -1, 0) is exactly on the
    // surface (dist_sq == radius^2), so the >= test leaves it untouched.
    let q =
        ClothPaintedBackstopQuery::new([1.0, -1.0, 0.0], 0, [0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0]);
    let r = gpu.evaluate(&ctx, &[q]);
    assert_eq!(r[0].valid, 0, "vertex on the surface is a no-op");
    assert_parity(&r[0], &q, "on_boundary_not_pushed");
}

#[test]
fn batch_of_two_or_more_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPaintedBackstop::new(&ctx);
    // Mixed batch exercises the std430 stride across several branches at once.
    let queries = [
        ClothPaintedBackstopQuery::new([0.0, -0.5, 0.0], 0, [0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0]),
        ClothPaintedBackstopQuery::new([5.0, 5.0, 5.0], 0, [0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0]),
        ClothPaintedBackstopQuery::new([0.2, 0.1, -0.3], 1, [1.0, 0.0, 0.0], 0.8, [1.0, 0.0, 0.0]),
        ClothPaintedBackstopQuery::new(
            [-0.4, 0.6, 0.2],
            0,
            [-0.5, 0.5, 0.0],
            0.9,
            [0.3, 0.8, -0.2],
        ),
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
    let gpu = GpuClothPaintedBackstop::new(&ctx);
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

/// Perturbs a query slightly for conditioning checks: nudges each geometric
/// coordinate, the normal and the backstop by a small matching amount.
fn perturb(q: &ClothPaintedBackstopQuery, delta: f32) -> ClothPaintedBackstopQuery {
    ClothPaintedBackstopQuery::new(
        [
            q.particle_pos[0] + delta,
            q.particle_pos[1] - delta,
            q.particle_pos[2] + delta,
        ],
        q.pinned,
        [
            q.anchor_pos[0] - delta,
            q.anchor_pos[1] + delta,
            q.anchor_pos[2] - delta,
        ],
        q.backstop + delta,
        [
            q.anchor_normal[0] + delta,
            q.anchor_normal[1] - delta,
            q.anchor_normal[2] + delta,
        ],
    )
}

/// Rejects marginal configurations where a tiny perturbation flips the `valid`
/// flag or shifts the projected position across the `dist_sq == radius^2` knee,
/// so the branch choice agrees on both sides despite any last-bit difference
/// between host and device.
fn well_conditioned(q: &ClothPaintedBackstopQuery) -> bool {
    let (base_pos, base_valid) = oracle(q);
    for &delta in &[1.0e-3_f32, -1.0e-3] {
        let (pos, valid) = oracle(&perturb(q, delta));
        if valid != base_valid {
            return false;
        }
        if base_valid == 1 {
            for axis in 0..3 {
                if (pos[axis] - base_pos[axis]).abs() > 2.0e-2 {
                    return false;
                }
            }
        }
    }
    true
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuClothPaintedBackstop::new(&ctx);
    let mut rng = Lcg::new(0x51_7A_C3_9D);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let particle = [
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
            rng.next_range(-3.0, 3.0),
        ];
        let anchor = [
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        ];
        // A non-degenerate normal so nlen_sq stays well above EPS_LEN_SQ; the
        // zero-normal corner is covered by a named fixture.
        let normal = [
            rng.next_range(-1.5, 1.5),
            rng.next_range(-1.5, 1.5),
            rng.next_range(-1.5, 1.5),
        ];
        if dot(normal, normal) < 0.25 {
            continue;
        }
        let backstop = rng.next_range(0.05, 2.0);
        // Inject an occasional pinned vertex so the sweep covers that no-op too.
        let pinned = u32::from(rng.next_unit() < 0.1);
        let q = ClothPaintedBackstopQuery::new(particle, pinned, anchor, backstop, normal);
        if !well_conditioned(&q) {
            continue;
        }
        queries.push(q);
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    let mut valid_count = 0u32;
    let mut invalid_count = 0u32;
    for (q, r) in queries.iter().zip(results.iter()) {
        let (_, valid) = oracle(q);
        if valid == 1 {
            valid_count += 1;
        } else {
            invalid_count += 1;
        }
        assert_parity(r, q, "random_sweep_matches_oracle");
    }
    // Guard against a degenerate sweep that would weakly test parity: the sweep
    // must exercise both the projection and the no-op branches.
    assert!(valid_count > 0, "sweep produced no projections");
    assert!(invalid_count > 0, "sweep produced no no-ops");
}
