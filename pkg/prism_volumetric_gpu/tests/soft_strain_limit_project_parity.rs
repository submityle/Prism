//! Real-device parity for the strain-limit twin:
//! [`GpuSoftStrainLimitProject`](prism_volumetric_gpu::soft_strain_limit_project::GpuSoftStrainLimitProject)
//! must reproduce the `CPU` golden `project_strain_limit` of
//! `prism_physics_core::soft::constraint::strain_limit`, the hard, mass-weighted
//! biphasic length clamp a stretch edge is projected through after the
//! compliant distance sweeps: it removes any length outside
//! `[rest_length * min_scale, rest_length * max_scale]` along the edge
//! direction, split between the two endpoints in proportion to their inverse
//! masses so a pinned particle never moves.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the two ordered degeneracy guards (non-positive inverse-mass sum, coincident
//! endpoints), the biphasic signed length error and the mass-weighted
//! correction — written out directly so the test never imports
//! `prism_physics_core` or `prism_render_architecture`. It mirrors the
//! reference branch for branch.
//!
//! The fixtures cover the regimes the kernel must honor: both endpoints pinned
//! (degenerate), coincident endpoints (degenerate), a length inside the band
//! (no move, valid), pure overstretch, pure over-compression, `min_scale == 0`
//! disabling the compression clamp, plus a multi-element mixed batch that
//! validates the `std430` array stride end to end. A sweep over random edges
//! kept away from every branch and clamp knee follows, plus an empty batch the
//! host short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The correction threads through a subtraction, a `sqrt` and guarded
//! divisions, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact. The continuous comparison is `abs_diff <= 1e-4 || rel_diff <=
//! 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//! The sweep keeps every length comfortably away from the band edges and the
//! coincidence floor so parity never sits on a branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::constraint::strain_limit::project_strain_limit`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_strain_limit_project::{
    GpuSoftStrainLimitProject, SoftStrainLimitProjectQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Coincidence floor on the edge length, matching the golden `EPSILON`
/// (`f32::EPSILON`).
const EPSILON: f32 = 1.1920929e-7;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Returns `true` when two vectors agree component-wise within tolerance.
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Independent host re-implementation of the golden `project_strain_limit`,
/// returning the projected endpoints and the `valid` flag without importing the
/// golden crate.
fn oracle(q: &SoftStrainLimitProjectQuery) -> ([f32; 3], [f32; 3], u32) {
    let pa = q.pa;
    let pb = q.pb;
    let w_sum = q.wa + q.wb;
    // Both endpoints pinned: nothing to project.
    if w_sum <= 0.0 {
        return (pa, pb, 0);
    }
    let delta = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
    let len = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    // Coincident endpoints: direction is undefined, skip.
    if len < EPSILON {
        return (pa, pb, 0);
    }
    let max_len = q.rest_length * q.max_scale;
    let min_len = q.rest_length * q.min_scale;
    // Signed length error outside the band; 0 inside (no move, still valid).
    let err = if len > max_len {
        len - max_len
    } else if q.min_scale > 0.0 && len < min_len {
        len - min_len
    } else {
        0.0
    };
    let direction = [delta[0] / len, delta[1] / len, delta[2] / len];
    let correction = [direction[0] * err, direction[1] * err, direction[2] * err];
    let fa = q.wa / w_sum;
    let fb = q.wb / w_sum;
    let new_pa = [
        pa[0] - correction[0] * fa,
        pa[1] - correction[1] * fa,
        pa[2] - correction[2] * fa,
    ];
    let new_pb = [
        pb[0] + correction[0] * fb,
        pb[1] + correction[1] * fb,
        pb[2] + correction[2] * fb,
    ];
    (new_pa, new_pb, 1)
}

/// Dispatches one query and asserts its parity against the host oracle.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuSoftStrainLimitProject,
    q: SoftStrainLimitProjectQuery,
) {
    let results = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let (new_pa, new_pb, valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close3(r.new_pa, new_pa),
        "new_pa mismatch: gpu={:?} cpu={new_pa:?} query={q:?}",
        r.new_pa
    );
    assert!(
        close3(r.new_pb, new_pb),
        "new_pb mismatch: gpu={:?} cpu={new_pb:?} query={q:?}",
        r.new_pb
    );
}

#[test]
fn both_pinned_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftStrainLimitProject::new(&ctx);
    // Zero inverse-mass sum: both particles pinned, output equals input.
    assert_parity(
        &ctx,
        &gpu,
        SoftStrainLimitProjectQuery::new([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], 0.0, 0.0, 1.0, 1.1, 0.9),
    );
}

#[test]
fn coincident_endpoints_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftStrainLimitProject::new(&ctx);
    // Endpoints coincide: length below EPSILON, direction undefined → invalid.
    assert_parity(
        &ctx,
        &gpu,
        SoftStrainLimitProjectQuery::new([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0, 1.0, 1.0, 1.1, 0.9),
    );
}

#[test]
fn inside_band_is_no_op() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftStrainLimitProject::new(&ctx);
    // Length 2.0 sits inside [rest*0.9, rest*1.1] = [1.8, 2.2]: no move, valid.
    let q =
        SoftStrainLimitProjectQuery::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], 1.0, 1.0, 2.0, 1.1, 0.9);
    assert_parity(&ctx, &gpu, q);
    // Explicitly confirm the no-op: output equals input exactly.
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.new_pa, q.pa, "inside-band endpoint A must not move");
    assert_eq!(r.new_pb, q.pb, "inside-band endpoint B must not move");
}

#[test]
fn pure_overstretch_splits_by_inverse_mass() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftStrainLimitProject::new(&ctx);
    // Length 4.0 exceeds max_len = 1.0 * 2.0 = 2.0; asymmetric masses split the
    // correction unevenly (A heavier → moves less).
    assert_parity(
        &ctx,
        &gpu,
        SoftStrainLimitProjectQuery::new([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.5, 1.5, 1.0, 2.0, 0.5),
    );
}

#[test]
fn pure_overcompression_pushes_apart() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftStrainLimitProject::new(&ctx);
    // Length 1.0 is below min_len = 2.0 * 0.8 = 1.6 with min_scale > 0: the
    // negative error pushes the endpoints apart. Use a 3-D diagonal edge.
    assert_parity(
        &ctx,
        &gpu,
        SoftStrainLimitProjectQuery::new(
            [1.0, 1.0, 1.0],
            [1.0 + 0.5773503, 1.0 + 0.5773503, 1.0 + 0.5773503],
            1.0,
            1.0,
            2.0,
            1.2,
            0.8,
        ),
    );
}

#[test]
fn min_scale_zero_disables_compression() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftStrainLimitProject::new(&ctx);
    // Length 0.5 is far below rest 2.0, but min_scale == 0 disables the
    // compression clamp: inside the (one-sided) band → no move, valid.
    let q =
        SoftStrainLimitProjectQuery::new([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 1.0, 1.0, 2.0, 1.1, 0.0);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, std::slice::from_ref(&q))[0];
    assert_eq!(r.valid, 1);
    assert_eq!(r.new_pa, q.pa, "compression disabled: A must not move");
    assert_eq!(r.new_pb, q.pb, "compression disabled: B must not move");
}

#[test]
fn multi_element_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftStrainLimitProject::new(&ctx);
    // A mixed batch (pinned, coincident, inside-band, overstretch,
    // over-compression) exercises the std430 array stride: every slot must
    // decode at the right byte offset.
    let queries = [
        SoftStrainLimitProjectQuery::new([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], 0.0, 0.0, 1.0, 1.1, 0.9),
        SoftStrainLimitProjectQuery::new([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], 1.0, 1.0, 1.0, 1.1, 0.9),
        SoftStrainLimitProjectQuery::new([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], 1.0, 1.0, 2.0, 1.1, 0.9),
        SoftStrainLimitProjectQuery::new([0.0, 0.0, 0.0], [0.0, 5.0, 0.0], 0.5, 1.5, 1.0, 2.0, 0.5),
        SoftStrainLimitProjectQuery::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 1.0, 2.0, 1.2, 0.8),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (new_pa, new_pb, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close3(r.new_pa, new_pa),
            "batch new_pa mismatch: gpu={:?} cpu={new_pa:?} query={q:?}",
            r.new_pa
        );
        assert!(
            close3(r.new_pb, new_pb),
            "batch new_pb mismatch: gpu={:?} cpu={new_pb:?} query={q:?}",
            r.new_pb
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftStrainLimitProject::new(&ctx);
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

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftStrainLimitProject::new(&ctx);
    let mut rng = Lcg::new(0x51_7A_2C_93);
    let mut queries = Vec::with_capacity(512);
    // Margin keeping every target length clear of the band edges so the host
    // and device never disagree about which branch the error takes.
    let margin = 0.1_f32;
    for i in 0..512 {
        // Positive inverse masses: w_sum is always strictly positive.
        let wa = rng.next_range(0.1, 2.0);
        let wb = rng.next_range(0.1, 2.0);
        let rest_length = rng.next_range(0.5, 3.0);
        let max_scale = rng.next_range(1.1, 1.5);
        let min_scale = rng.next_range(0.5, 0.9);
        let max_len = rest_length * max_scale;
        let min_len = rest_length * min_scale;

        // A well-conditioned unit direction: components bounded away from zero.
        let sx = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
        let sy = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
        let sz = if rng.next_u32() & 1 == 0 { 1.0 } else { -1.0 };
        let dx = sx * rng.next_range(0.3, 1.0);
        let dy = sy * rng.next_range(0.3, 1.0);
        let dz = sz * rng.next_range(0.3, 1.0);
        let inv_norm = 1.0 / (dx * dx + dy * dy + dz * dz).sqrt();
        let ux = dx * inv_norm;
        let uy = dy * inv_norm;
        let uz = dz * inv_norm;

        // Rotate through the three regimes, each with a safe margin from the
        // band edges (and from the coincidence floor, since min_len >= 0.25).
        let target_len = match i % 3 {
            0 => max_len + margin + rng.next_range(0.0, 1.0),
            1 => {
                // Inside band: the band is at least rest*(1.1-0.9)=0.2*0.5=0.1
                // wide, so clamp the usable interior to a non-empty range.
                let lo = min_len + margin;
                let hi = (max_len - margin).max(lo + 1.0e-3);
                rng.next_range(lo, hi)
            }
            _ => (min_len - margin - rng.next_range(0.0, 0.1)).max(0.05),
        };

        // Random base endpoint, with the other endpoint offset by target_len
        // along the unit direction.
        let pb = [
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ];
        let pa = [
            pb[0] + ux * target_len,
            pb[1] + uy * target_len,
            pb[2] + uz * target_len,
        ];
        queries.push(SoftStrainLimitProjectQuery::new(
            pa,
            pb,
            wa,
            wb,
            rest_length,
            max_scale,
            min_scale,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (new_pa, new_pb, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close3(r.new_pa, new_pa),
            "sweep new_pa mismatch: gpu={:?} cpu={new_pa:?} query={q:?}",
            r.new_pa
        );
        assert!(
            close3(r.new_pb, new_pb),
            "sweep new_pb mismatch: gpu={:?} cpu={new_pb:?} query={q:?}",
            r.new_pb
        );
    }
}
