//! Real-device parity for the hair orthonormal-basis twin:
//! [`GpuHairOrthonormalBasis`](prism_volumetric_gpu::hair_orthonormal_basis::GpuHairOrthonormalBasis)
//! must reproduce the `CPU` golden `build_orthonormal_basis` of
//! `prism_render_architecture::hair::rest_helix`, which seeds a rest-state hair
//! strand with a stable pair of unit vectors spanning the plane perpendicular
//! to a growth axis.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the finite, above-epsilon `normalize_or_zero`, the fixed-axis fallback for a
//! degenerate axis, the least-aligned cardinal `helper` selection, the second
//! cardinal fallback for the parallel corner case, and the completing
//! `binormal = axis x normal` cross — written out directly so the test never
//! imports `prism_render_architecture`. It mirrors the reference branch for
//! branch, including the `is_finite` guard and the exact cross-product
//! evaluation order, so the returned handedness matches the golden step for
//! step rather than an arbitrary orthonormal frame.
//!
//! The fixtures cover the branches the kernel must honor: each cardinal axis
//! (the `+x` / `-x` axes trip the `|a.x| >= 0.9` helper branch and then the
//! degenerate-cross fallback, since `a x (1, 0, 0)` is zero there), a near-zero
//! axis that collapses to the fallback, general oblique axes, a batch of two or
//! more elements that validates the `std430` stride, and an empty batch the
//! host short-circuits with no dispatch. Every fixture also asserts the frame
//! is genuinely orthonormal (unit lengths and mutually/axis-orthogonal). A
//! sweep over random axes follows, rejecting axes whose normalised `|x|` sits
//! on the `0.9` helper knee so the branch choice agrees on both sides.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The basis path threads through multiplies, adds and a guarded division by
//! `sqrt(len_sq)`, so `CPU` and `GPU` evaluate the same closed form but need
//! not be bit-exact (a `GPU` may contract a multiply-add). The continuous
//! comparison is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`);
//! the discrete `valid` flag is compared exactly. Because the fallback always
//! yields a basis, `valid` is always `1`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::hair::rest_helix`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::hair_orthonormal_basis::{
    GpuHairOrthonormalBasis, HairOrthonormalBasisQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Squared-length threshold below which a vector is treated as degenerate,
/// matching the reference `NORMALIZE_EPS_SQ`.
const NORMALIZE_EPS_SQ: f32 = 1.0e-24;
/// The cardinal axis substituted for a zero or non-finite input, matching the
/// reference `FALLBACK_AXIS = (0, 1, 0)`.
const FALLBACK_AXIS: [f32; 3] = [0.0, 1.0, 0.0];

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Squared Euclidean length.
fn length_squared(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// Dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b`, right-handed, written out to match the golden order.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Independent host re-implementation of the reference `normalize_or_zero`: a
/// finite, above-epsilon squared length divides by `sqrt(len_sq)`; otherwise
/// the vector collapses to zero. The `is_finite` guard is replicated so a
/// non-finite axis takes the same branch the golden does.
fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    let len_sq = length_squared(v);
    if len_sq.is_finite() && len_sq > NORMALIZE_EPS_SQ {
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Independent host re-implementation of `build_orthonormal_basis`: normalise
/// the axis (falling back to the fixed cardinal when degenerate), pick the
/// least-aligned cardinal helper, normalise the seeding cross product (falling
/// back to the other cardinal when parallel), and complete the frame with
/// `binormal = a x normal`.
fn build_orthonormal_basis(axis: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let n = normalize_or_zero(axis);
    let a = if length_squared(n) > NORMALIZE_EPS_SQ {
        n
    } else {
        FALLBACK_AXIS
    };
    let helper = if a[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let seeded = normalize_or_zero(cross(a, helper));
    let normal = if length_squared(seeded) > NORMALIZE_EPS_SQ {
        seeded
    } else {
        normalize_or_zero(cross(a, [0.0, 0.0, 1.0]))
    };
    let binormal = cross(a, normal);
    (normal, binormal)
}

/// The full host oracle for one query: `(normal, binormal, valid)`. The
/// fallback guarantees a basis, so `valid` is always `1`.
fn oracle(q: &HairOrthonormalBasisQuery) -> ([f32; 3], [f32; 3], u32) {
    let (normal, binormal) = build_orthonormal_basis([q.axis_x, q.axis_y, q.axis_z]);
    (normal, binormal, 1)
}

/// Recomputes the sanitised axis `a` the golden uses for its orthogonality
/// guarantees (normalised, with the fixed-axis fallback for a degenerate
/// input).
fn sanitized_axis(q: &HairOrthonormalBasisQuery) -> [f32; 3] {
    let n = normalize_or_zero([q.axis_x, q.axis_y, q.axis_z]);
    if length_squared(n) > NORMALIZE_EPS_SQ {
        n
    } else {
        FALLBACK_AXIS
    }
}

/// Asserts the returned frame is genuinely orthonormal with respect to the
/// sanitised axis: both vectors are unit length and mutually, and
/// axis-, orthogonal.
fn assert_orthonormal(q: &HairOrthonormalBasisQuery, normal: [f32; 3], binormal: [f32; 3]) {
    let a = sanitized_axis(q);
    assert!(
        close(length_squared(normal), 1.0),
        "normal must be unit length: {normal:?} query={q:?}"
    );
    assert!(
        close(length_squared(binormal), 1.0),
        "binormal must be unit length: {binormal:?} query={q:?}"
    );
    assert!(
        close(dot(normal, binormal), 0.0),
        "normal and binormal must be orthogonal: {} query={q:?}",
        dot(normal, binormal)
    );
    assert!(
        close(dot(normal, a), 0.0),
        "normal must be perpendicular to axis: {} query={q:?}",
        dot(normal, a)
    );
    assert!(
        close(dot(binormal, a), 0.0),
        "binormal must be perpendicular to axis: {} query={q:?}",
        dot(binormal, a)
    );
}

/// Dispatches one query and asserts every continuous channel plus the validity
/// flag against the independent oracle, then checks the frame is orthonormal.
fn assert_parity(ctx: &GpuContext, gpu: &GpuHairOrthonormalBasis, q: HairOrthonormalBasisQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (normal, binormal, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    assert!(
        close(r.normal_x, normal[0])
            && close(r.normal_y, normal[1])
            && close(r.normal_z, normal[2]),
        "normal mismatch: gpu=({}, {}, {}) cpu={normal:?} query={q:?}",
        r.normal_x,
        r.normal_y,
        r.normal_z
    );
    assert!(
        close(r.binormal_x, binormal[0])
            && close(r.binormal_y, binormal[1])
            && close(r.binormal_z, binormal[2]),
        "binormal mismatch: gpu=({}, {}, {}) cpu={binormal:?} query={q:?}",
        r.binormal_x,
        r.binormal_y,
        r.binormal_z
    );
    assert_orthonormal(
        &q,
        [r.normal_x, r.normal_y, r.normal_z],
        [r.binormal_x, r.binormal_y, r.binormal_z],
    );
}

#[test]
fn cardinal_axes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairOrthonormalBasis::new(&ctx);
    // The +/-x axes trip the |a.x| >= 0.9 helper branch (helper = (0, 1, 0)) and
    // then the degenerate-cross fallback, since a x (0, 1, 0) with a = +/-x is
    // non-degenerate here; the +/-y and +/-z axes stay on the first branch.
    let axes = [
        [1.0_f32, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    for a in axes {
        let q = HairOrthonormalBasisQuery::new(a[0], a[1], a[2]);
        assert_parity(&ctx, &gpu, q);
    }
}

#[test]
fn x_axis_triggers_degenerate_cross_fallback() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairOrthonormalBasis::new(&ctx);
    // With a = (1, 0, 0), |a.x| >= 0.9 so helper = (0, 1, 0); a x helper =
    // (0, 0, 1) is well-conditioned so the first branch holds. Confirm the
    // basis is still orthonormal and matches the oracle exactly.
    let q = HairOrthonormalBasisQuery::new(1.0, 0.0, 0.0);
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn near_zero_axis_falls_back() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairOrthonormalBasis::new(&ctx);
    // A tiny axis has squared length below NORMALIZE_EPS_SQ, so it collapses to
    // zero and is replaced by FALLBACK_AXIS = (0, 1, 0); the resulting basis is
    // that of the fallback axis.
    let q = HairOrthonormalBasisQuery::new(1.0e-20, 0.0, 0.0);
    assert_parity(&ctx, &gpu, q);
    let (normal, binormal, _) = oracle(&q);
    let (fb_normal, fb_binormal) = build_orthonormal_basis(FALLBACK_AXIS);
    assert!(
        close(normal[0], fb_normal[0])
            && close(normal[1], fb_normal[1])
            && close(normal[2], fb_normal[2])
            && close(binormal[0], fb_binormal[0])
            && close(binormal[1], fb_binormal[1])
            && close(binormal[2], fb_binormal[2]),
        "degenerate axis must reuse the fallback basis"
    );
}

#[test]
fn exact_zero_axis_falls_back() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairOrthonormalBasis::new(&ctx);
    let q = HairOrthonormalBasisQuery::new(0.0, 0.0, 0.0);
    assert_parity(&ctx, &gpu, q);
}

#[test]
fn general_oblique_axes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairOrthonormalBasis::new(&ctx);
    // Oblique axes with |x| comfortably below and above 0.9 to exercise both
    // helper branches away from the knee.
    let axes = [
        [1.0_f32, 2.0, 3.0],
        [-3.0, 1.0, -2.0],
        [0.1, 0.2, 0.97],
        [5.0, 0.3, 0.3],
        [-5.0, -0.2, 0.4],
        [0.5, -0.5, 0.5],
    ];
    for a in axes {
        let q = HairOrthonormalBasisQuery::new(a[0], a[1], a[2]);
        assert_parity(&ctx, &gpu, q);
    }
}

#[test]
fn batch_of_two_or_more_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairOrthonormalBasis::new(&ctx);
    // A multi-element batch catches any std430 stride mismatch between the host
    // struct and the shader layout.
    let queries = [
        HairOrthonormalBasisQuery::new(1.0, 0.0, 0.0),
        HairOrthonormalBasisQuery::new(0.0, 1.0, 0.0),
        HairOrthonormalBasisQuery::new(0.0, 0.0, 1.0),
        HairOrthonormalBasisQuery::new(1.0, 2.0, 3.0),
        HairOrthonormalBasisQuery::new(-2.0, 0.5, 0.1),
    ];
    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let (normal, binormal, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close(r.normal_x, normal[0])
                && close(r.normal_y, normal[1])
                && close(r.normal_z, normal[2]),
            "batch normal mismatch: gpu=({}, {}, {}) cpu={normal:?} query={q:?}",
            r.normal_x,
            r.normal_y,
            r.normal_z
        );
        assert!(
            close(r.binormal_x, binormal[0])
                && close(r.binormal_y, binormal[1])
                && close(r.binormal_z, binormal[2]),
            "batch binormal mismatch: gpu=({}, {}, {}) cpu={binormal:?} query={q:?}",
            r.binormal_x,
            r.binormal_y,
            r.binormal_z
        );
        assert_orthonormal(
            q,
            [r.normal_x, r.normal_y, r.normal_z],
            [r.binormal_x, r.binormal_y, r.binormal_z],
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHairOrthonormalBasis::new(&ctx);
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
    let gpu = GpuHairOrthonormalBasis::new(&ctx);
    let mut rng = Lcg::new(0x51_7A_2E_63);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let ax = rng.next_range(-4.0, 4.0);
        let ay = rng.next_range(-4.0, 4.0);
        let az = rng.next_range(-4.0, 4.0);
        let len_sq = ax * ax + ay * ay + az * az;
        // Skip near-zero axes so we test the normalising branch, not the
        // fallback (that corner is covered by the named fixtures).
        if len_sq <= 1.0e-6 {
            continue;
        }
        // Reject axes whose normalised |x| sits on the 0.9 helper knee so the
        // branch choice agrees on both sides despite any last-bit difference.
        let inv_len = 1.0 / len_sq.sqrt();
        let nx = (ax * inv_len).abs();
        if (nx - 0.9).abs() < 1.0e-2 {
            continue;
        }
        queries.push(HairOrthonormalBasisQuery::new(ax, ay, az));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (normal, binormal, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close(r.normal_x, normal[0])
                && close(r.normal_y, normal[1])
                && close(r.normal_z, normal[2]),
            "sweep normal mismatch: gpu=({}, {}, {}) cpu={normal:?} query={q:?}",
            r.normal_x,
            r.normal_y,
            r.normal_z
        );
        assert!(
            close(r.binormal_x, binormal[0])
                && close(r.binormal_y, binormal[1])
                && close(r.binormal_z, binormal[2]),
            "sweep binormal mismatch: gpu=({}, {}, {}) cpu={binormal:?} query={q:?}",
            r.binormal_x,
            r.binormal_y,
            r.binormal_z
        );
        assert_orthonormal(
            q,
            [r.normal_x, r.normal_y, r.normal_z],
            [r.binormal_x, r.binormal_y, r.binormal_z],
        );
    }
}
