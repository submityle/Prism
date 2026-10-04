//! Real-device parity for the Janssen stress-state twin:
//! [`GpuJanssenStressState`](prism_volumetric_gpu::janssen_stress_state::GpuJanssenStressState)
//! must reproduce the `CPU` golden `JanssenProfile` depth getters of
//! `prism_physics_core::collider::janssen_pressure`. The profile saturates the
//! vertical stress exponentially toward `sigma_inf` with characteristic depth
//! `z_c`: `sigma_v = sigma_inf * (1 - exp(-depth / z_c))`,
//! `sigma_h = K * sigma_v`, `tau_w = mu_w * sigma_h`, and the friction
//! screening fraction is `clamp(1 - sigma_v / (rho * g * depth), 0, 1)`.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`
//! or `prism_physics_core`. It mirrors the golden by evaluating the saturation
//! `exp` in `f64` (the golden bans `exp` on `f32`), while the device uses the
//! `WGSL` built-in `exp` in `f32`.
//!
//! The fixtures cover a mid-ratio point, a non-positive depth, a `NaN` and an
//! infinite depth (all all-zero), a shallow ratio-`0.05` point, a deep
//! ratio-`8` point whose screening approaches `1`, a zeroed hydrostatic
//! reference (gravity `0`) that zeroes only the screening while the stresses
//! stay non-zero, a batch of two or more elements mixing valid and degenerate
//! queries to validate the `std430` stride, and an empty batch the host
//! short-circuits with no dispatch. A sweep over random finite inputs follows;
//! it reject-samples the depth so the ratio `depth / z_c` stays in `[0.05, 8]`
//! with a margin, keeping the `f32`/`f64` `exp` gap inside tolerance.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The saturation factor is evaluated in `f32` `exp` on the device against the
//! golden's `f64` `exp`, so `CPU` and `GPU` evaluate the same closed form but
//! need not be bit-exact. Every stress scalar is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::janssen_pressure`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::janssen_stress_state::{
    GpuJanssenStressState, JanssenStressStateQuery, JanssenStressStateResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces the golden `JanssenProfile` depth
/// getters in the golden operator order, returning the four stresses. The
/// saturation `exp` is evaluated in `f64` to mirror the golden exactly.
fn oracle(q: &JanssenStressStateQuery) -> (f32, f32, f32, f32) {
    let depth_ok = q.depth.is_finite()
        && q.depth > 0.0
        && q.characteristic_depth.is_finite()
        && q.characteristic_depth > 0.0;
    let vertical = if depth_ok {
        let ratio = (q.depth / q.characteristic_depth) as f64;
        let factor = 1.0 - (-ratio).exp();
        q.saturation_vertical * factor as f32
    } else {
        0.0
    };
    let horizontal = q.k_ratio * vertical;
    let wall_shear = q.wall_friction * horizontal;
    let screening = if depth_ok {
        let hydro = q.bulk_density * q.gravity * q.depth;
        if hydro > 0.0 {
            (1.0 - vertical / hydro).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };
    (vertical, horizontal, wall_shear, screening)
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle on all four
/// stress scalars to tolerance.
fn assert_parity(gpu: &JanssenStressStateResult, q: &JanssenStressStateQuery, label: &str) {
    let (vertical, horizontal, wall_shear, screening) = oracle(q);
    assert!(
        close(gpu.vertical_stress, vertical),
        "{label}: vertical gpu={} oracle={}",
        gpu.vertical_stress,
        vertical
    );
    assert!(
        close(gpu.horizontal_stress, horizontal),
        "{label}: horizontal gpu={} oracle={}",
        gpu.horizontal_stress,
        horizontal
    );
    assert!(
        close(gpu.wall_shear_stress, wall_shear),
        "{label}: wall_shear gpu={} oracle={}",
        gpu.wall_shear_stress,
        wall_shear
    );
    assert!(
        close(gpu.screening_fraction, screening),
        "{label}: screening gpu={} oracle={}",
        gpu.screening_fraction,
        screening
    );
}

/// A representative well-conditioned profile: z_c=2, sigma_inf=1e4, K=0.5,
/// mu_w=0.4, rho=1500, g=9.81.
fn profile(depth: f32) -> JanssenStressStateQuery {
    JanssenStressStateQuery::new(2.0, 1.0e4, 0.5, 0.4, 1500.0, 9.81, depth)
}

#[test]
fn mid_ratio_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    // ratio = depth/z_c = 2 -> factor = 1 - e^-2 ~ 0.8647.
    let q = profile(4.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].vertical_stress > 0.0);
    assert_parity(&out[0], &q, "mid_ratio");
}

#[test]
fn non_positive_depth_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    let zero = profile(0.0);
    let neg = profile(-3.0);
    let out = gpu.evaluate(&ctx, &[zero, neg]);
    assert_eq!(out.len(), 2);
    for (i, (res, q)) in out.iter().zip([zero, neg].iter()).enumerate() {
        assert_eq!(res.vertical_stress, 0.0, "idx {i}: vertical");
        assert_eq!(res.horizontal_stress, 0.0, "idx {i}: horizontal");
        assert_eq!(res.wall_shear_stress, 0.0, "idx {i}: wall_shear");
        assert_eq!(res.screening_fraction, 0.0, "idx {i}: screening");
        assert_parity(res, q, "non_positive_depth");
    }
}

#[test]
fn nan_depth_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    let q = profile(f32::NAN);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].vertical_stress, 0.0);
    assert_eq!(out[0].screening_fraction, 0.0);
    assert_parity(&out[0], &q, "nan_depth");
}

#[test]
fn infinite_depth_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    let q = profile(f32::INFINITY);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].vertical_stress, 0.0);
    assert_eq!(out[0].screening_fraction, 0.0);
    assert_parity(&out[0], &q, "infinite_depth");
}

#[test]
fn shallow_ratio_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    // ratio = 0.05 -> factor ~ 0.0488, near the surface.
    let q = profile(0.1);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_parity(&out[0], &q, "shallow_ratio");
}

#[test]
fn deep_ratio_screening_approaches_one() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    // ratio = 8 -> factor ~ 0.99966, deep in the silo; screening -> ~1.
    let q = profile(16.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let (_, _, _, screening) = oracle(&q);
    assert!(
        screening > 0.9,
        "deep screening should be high: {screening}"
    );
    assert_parity(&out[0], &q, "deep_ratio");
}

#[test]
fn zero_gravity_zeroes_only_screening() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    // gravity = 0 -> hydrostatic reference 0 -> screening gated to 0, but the
    // stresses stay non-zero since sigma_inf carries them.
    let q = JanssenStressStateQuery::new(2.0, 1.0e4, 0.5, 0.4, 1500.0, 0.0, 4.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].vertical_stress > 0.0, "vertical stays positive");
    assert_eq!(out[0].screening_fraction, 0.0, "screening gated to 0");
    assert_parity(&out[0], &q, "zero_gravity");
}

#[test]
fn batch_mixes_valid_and_degenerate_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    let queries = vec![
        profile(4.0),
        profile(-1.0),
        profile(0.1),
        profile(f32::NAN),
        JanssenStressStateQuery::new(2.0, 1.0e4, 0.5, 0.4, 1500.0, 0.0, 4.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    assert!(out[0].vertical_stress > 0.0);
    assert_eq!(out[1].vertical_stress, 0.0);
    assert!(out[2].vertical_stress > 0.0);
    assert_eq!(out[3].vertical_stress, 0.0);
    assert_eq!(out[4].screening_fraction, 0.0);
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("batch[{i}]"));
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuJanssenStressState::new(&ctx);
    let mut lcg = Lcg::new(0x4A17_55E7);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let z_c = lcg.next_range(0.1, 10.0);
        let sat = lcg.next_range(1.0e2, 1.0e5);
        let k_ratio = lcg.next_range(0.3, 0.7);
        let wall_friction = lcg.next_range(0.2, 0.8);
        let bulk_density = lcg.next_range(500.0, 2500.0);
        let gravity = lcg.next_range(9.0, 10.0);
        // Reject-sample depth so ratio = depth/z_c stays in [0.05, 8] with a
        // margin, keeping the f32/f64 exp gap inside tolerance.
        let ratio = lcg.next_range(0.05 + 0.05, 8.0 - 0.05);
        let depth = ratio * z_c;
        queries.push(JanssenStressStateQuery::new(
            z_c,
            sat,
            k_ratio,
            wall_friction,
            bulk_density,
            gravity,
            depth,
        ));
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("sweep[{i}]"));
    }
}

/// A small deterministic linear-congruential generator; the fixture carries no
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
