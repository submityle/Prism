//! Real-device parity for the anisotropic-`GGX` evaluation twin:
//! [`GpuGgxAnisotropic`](prism_volumetric_gpu::microfacet_anisotropic::GpuGgxAnisotropic)
//! must reproduce the `CPU` golden `GgxAnisotropic::distribution`, `g1`, `g2`
//! and `reflection_pdf` of
//! `prism_render_architecture::reference_pt::microfacet_aniso`, the stateless,
//! no-`RNG` evaluation terms of the elliptical (brushed-metal) microfacet lobe.
//!
//! The oracle here is an independent re-implementation of those closed forms —
//! the elliptical `GGX` `distribution`, the Smith `lambda` feeding `g1`/`g2`,
//! and the solid-angle `reflection_pdf` — written out directly so the test
//! never imports `prism_render_architecture`. It mirrors the reference branch
//! for branch, including the below-horizon (`wo.z <= 0`) and back-facing
//! half-vector (`h.z <= 0`) guards that clear the outputs and report
//! `valid = 0`.
//!
//! The fixtures cover the lobes the kernel must honor: an isotropic lobe at
//! normal incidence and at an oblique angle, a strongly anisotropic lobe, a
//! grazing-but-valid outgoing direction, the roughness/anisotropy remap entry
//! point, and two degenerate pairs (back-facing half vector and below-horizon
//! view) that must report `valid = 0`. A sweep over random widths and
//! upper-hemisphere directions follows, plus an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every channel threads through multiplies, adds, guarded divisions and
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact (a `GPU` may contract a multiply-add). The continuous comparison
//! is `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete
//! `valid` flag is compared exactly. Fixtures and the sweep keep both `wo.z`
//! and `h.z` comfortably positive and retain a tangential component, so parity
//! never sits on the `wo.z = 0`, `h.z = 0` or `cz = 1` branch knife edges.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::microfacet_aniso`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::microfacet_anisotropic::{GgxAnisotropicQuery, GpuGgxAnisotropic};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Minimum direction squared length below which normalization is degenerate.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Dot product of two three-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalizes a three-vector, returning zero for a degenerate length.
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(v, v);
    if len_sq <= EPS_LEN_SQ {
        return [0.0, 0.0, 0.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Independent oracle for the anisotropic `GGX` normal distribution `D(h)`.
fn distribution(h: [f32; 3], ax: f32, ay: f32) -> f32 {
    if h[2] <= 0.0 {
        return 0.0;
    }
    let hx = h[0] / ax;
    let hy = h[1] / ay;
    let hz = h[2];
    let q = hx * hx + hy * hy + hz * hz;
    (1.0 / std::f32::consts::PI) / (ax * ay * q * q)
}

/// Independent oracle for the Smith `Lambda` auxiliary of a local-frame
/// direction.
fn lambda(w: [f32; 3], ax: f32, ay: f32) -> f32 {
    let cz = w[2].abs();
    if cz >= 1.0 {
        return 0.0;
    }
    let axx = ax * w[0];
    let ayy = ay * w[1];
    let numer = axx * axx + ayy * ayy;
    if numer <= 0.0 {
        return 0.0;
    }
    let ratio = numer / (cz * cz);
    0.5 * ((1.0 + ratio).sqrt() - 1.0)
}

/// Smith `G1` single-direction masking term.
fn g1(w: [f32; 3], ax: f32, ay: f32) -> f32 {
    1.0 / (1.0 + lambda(w, ax, ay))
}

/// Smith `G2` height-correlated masking-shadowing term.
fn g2(wo: [f32; 3], wi: [f32; 3], ax: f32, ay: f32) -> f32 {
    1.0 / (1.0 + lambda(wo, ax, ay) + lambda(wi, ax, ay))
}

/// Solid-angle reflection density `G1(wo) * D(h) / (4 wo.z)`.
fn reflection_pdf(wo: [f32; 3], h: [f32; 3], ax: f32, ay: f32) -> f32 {
    if wo[2] <= 0.0 {
        return 0.0;
    }
    g1(wo, ax, ay) * distribution(h, ax, ay) / (4.0 * wo[2])
}

/// The independent oracle for one query, mirroring the kernel branch for
/// branch: returns `(d, g1_wo, g2, pdf, valid)`. The widths are read directly
/// from the query, which already clamped them to `MIN_ALPHA` at construction.
fn oracle(q: &GgxAnisotropicQuery) -> (f32, f32, f32, f32, u32) {
    let ax = q.alpha_x;
    let ay = q.alpha_y;
    let h = q.h;
    let wo = q.wo;
    let wi = q.wi;
    if wo[2] <= 0.0 || h[2] <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0, 0u32);
    }
    let d = distribution(h, ax, ay);
    let g1_wo = g1(wo, ax, ay);
    let g2v = g2(wo, wi, ax, ay);
    let pdf = reflection_pdf(wo, h, ax, ay);
    (d, g1_wo, g2v, pdf, 1u32)
}

/// Dispatches one query and asserts all four continuous channels plus the
/// validity flag.
fn assert_parity(ctx: &GpuContext, gpu: &GpuGgxAnisotropic, q: GgxAnisotropicQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (d, g1_wo, g2v, pdf, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    if valid == 1 {
        assert!(close(r.d, d), "d mismatch: gpu={} cpu={d} query={q:?}", r.d);
        assert!(
            close(r.g1_wo, g1_wo),
            "g1_wo mismatch: gpu={} cpu={g1_wo} query={q:?}",
            r.g1_wo
        );
        assert!(
            close(r.g2, g2v),
            "g2 mismatch: gpu={} cpu={g2v} query={q:?}",
            r.g2
        );
        assert!(
            close(r.pdf, pdf),
            "pdf mismatch: gpu={} cpu={pdf} query={q:?}",
            r.pdf
        );
    }
}

#[test]
fn isotropic_normal_incidence() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
    // Isotropic lobe, all directions along the shading normal.
    assert_parity(
        &ctx,
        &gpu,
        GgxAnisotropicQuery::new(0.3, 0.3, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
    );
}

#[test]
fn isotropic_oblique() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
    let h = normalize3([0.2, 0.1, 0.95]);
    let wo = normalize3([0.3, 0.2, 0.9]);
    let wi = normalize3([-0.25, 0.15, 0.9]);
    assert_parity(&ctx, &gpu, GgxAnisotropicQuery::new(0.4, 0.4, h, wo, wi));
}

#[test]
fn wide_vs_narrow_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
    // Strongly anisotropic lobe: wide along x, narrow along y.
    let h = normalize3([0.35, 0.05, 0.9]);
    let wo = normalize3([0.4, -0.1, 0.85]);
    let wi = normalize3([-0.3, 0.2, 0.88]);
    assert_parity(&ctx, &gpu, GgxAnisotropicQuery::new(0.6, 0.08, h, wo, wi));
}

#[test]
fn grazing_outgoing_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
    // Outgoing direction near the horizon but still above it (wo.z > 0).
    let wo = normalize3([0.8, 0.2, 0.12]);
    let h = normalize3([0.3, 0.1, 0.9]);
    let wi = normalize3([-0.2, 0.3, 0.8]);
    assert_parity(&ctx, &gpu, GgxAnisotropicQuery::new(0.25, 0.5, h, wo, wi));
}

#[test]
fn from_roughness_anisotropy_entry_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
    let h = normalize3([0.15, 0.25, 0.95]);
    let wo = normalize3([0.3, 0.1, 0.9]);
    let wi = normalize3([-0.2, 0.2, 0.9]);
    // Build the widths through the Disney / UE roughness-anisotropy remap.
    assert_parity(
        &ctx,
        &gpu,
        GgxAnisotropicQuery::from_roughness_anisotropy(0.5, 0.8, h, wo, wi),
    );
}

#[test]
fn back_facing_half_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
    // Back-facing half vector (h.z <= 0): valid = 0, outputs cleared.
    assert_parity(
        &ctx,
        &gpu,
        GgxAnisotropicQuery::new(0.3, 0.3, [0.2, 0.1, -0.9], [0.1, 0.2, 0.9], [0.0, 0.0, 1.0]),
    );
}

#[test]
fn below_horizon_outgoing_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
    // Outgoing direction below the surface (wo.z <= 0): valid = 0.
    assert_parity(
        &ctx,
        &gpu,
        GgxAnisotropicQuery::new(0.3, 0.3, [0.1, 0.1, 0.9], [0.2, 0.3, -0.8], [0.0, 0.0, 1.0]),
    );
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
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

    /// A unit upper-hemisphere direction with a comfortably positive `z` and a
    /// retained tangential component, so parity stays away from the `wo.z = 0`,
    /// `h.z = 0` and `cz = 1` branch edges.
    fn next_upper_dir(&mut self) -> [f32; 3] {
        normalize3([
            self.next_range(-0.8, 0.8),
            self.next_range(-0.8, 0.8),
            self.next_range(0.3, 0.95),
        ])
    }
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
    let mut rng = Lcg::new(0x7E_41_C3_09);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        let h = rng.next_upper_dir();
        let wo = rng.next_upper_dir();
        let wi = rng.next_upper_dir();
        // Keep the widths away from the mirror-sharp regime so the GGX
        // denominator stays well conditioned on both sides.
        let alpha_x = rng.next_range(0.02, 1.0);
        let alpha_y = rng.next_range(0.02, 1.0);
        queries.push(GgxAnisotropicQuery::new(alpha_x, alpha_y, h, wo, wi));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (d, g1_wo, g2v, pdf, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        if valid == 1 {
            assert!(
                close(r.d, d),
                "sweep d mismatch: gpu={} cpu={d} query={q:?}",
                r.d
            );
            assert!(
                close(r.g1_wo, g1_wo),
                "sweep g1_wo mismatch: gpu={} cpu={g1_wo} query={q:?}",
                r.g1_wo
            );
            assert!(
                close(r.g2, g2v),
                "sweep g2 mismatch: gpu={} cpu={g2v} query={q:?}",
                r.g2
            );
            assert!(
                close(r.pdf, pdf),
                "sweep pdf mismatch: gpu={} cpu={pdf} query={q:?}",
                r.pdf
            );
        }
    }
}

/// Regression for the std430 query-stride bug. This exact query is element
/// index 1 of the `random_sweep_matches_oracle` sweep (seed `0x7E_41_C3_09`).
/// It is non-degenerate — `wo.z = 0.86`, `h.z = 0.79`, far from every branch
/// edge — yet the device reported `valid = 0`: the kernel's `Query` struct was
/// a 44-byte std430 record while the host uploaded a 48-byte one, so every
/// element past the first read four bytes shifted and `wo.z`/`h.z` aliased
/// negative neighbors. The query is dispatched *behind* a leading element so
/// the non-zero stride is actually exercised: a single-element batch always
/// reads slot 0 correctly and would never catch this. With the WGSL `Query`
/// padded to 48 bytes the lanes line up and element 1 reports `valid = 1` with
/// all four continuous channels matching the oracle.
#[test]
fn sweep_regression_valid_point() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxAnisotropic::new(&ctx);
    // A leading well-conditioned query occupying slot 0, then the regression
    // query in slot 1 where the array stride first matters.
    let lead = GgxAnisotropicQuery::new(
        0.3,
        0.3,
        normalize3([0.2, 0.1, 0.95]),
        normalize3([0.3, 0.2, 0.9]),
        normalize3([-0.25, 0.15, 0.9]),
    );
    let regression = GgxAnisotropicQuery::new(
        0.079_069_406,
        0.383_818_86,
        [-0.603_819_43, -0.129_855_28, 0.786_472_86],
        [0.334_864_6, -0.385_116_07, 0.859_971_76],
        [0.524_644_43, -0.480_824_38, 0.702_535_5],
    );
    let queries = [lead, regression];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), 2, "one result per query");

    let (d, g1_wo, g2v, pdf, valid) = oracle(&regression);
    assert_eq!(
        valid, 1u32,
        "the regression query must be a valid interaction"
    );
    let r = results[1];
    assert_eq!(
        r.valid, valid,
        "stride-shifted slot-1 valid mismatch: query={regression:?}"
    );
    assert!(
        close(r.d, d),
        "regression d mismatch: gpu={} cpu={d} query={regression:?}",
        r.d
    );
    assert!(
        close(r.g1_wo, g1_wo),
        "regression g1_wo mismatch: gpu={} cpu={g1_wo} query={regression:?}",
        r.g1_wo
    );
    assert!(
        close(r.g2, g2v),
        "regression g2 mismatch: gpu={} cpu={g2v} query={regression:?}",
        r.g2
    );
    assert!(
        close(r.pdf, pdf),
        "regression pdf mismatch: gpu={} cpu={pdf} query={regression:?}",
        r.pdf
    );
}
