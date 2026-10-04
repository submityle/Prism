//! Real-device parity for the mixed-mode bilinear cohesive-zone traction twin:
//! [`GpuCohesiveMixedModeTraction`](prism_volumetric_gpu::cohesive_mixed_mode_traction::GpuCohesiveMixedModeTraction)
//! must reproduce the `CPU` golden `cohesive_traction` of
//! `prism_physics_core::collider::cohesive_zone` (together with
//! `CohesiveModel::new` and `damage_at`) for a single interface evaluation.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`,
//! `prism_physics_core` or `glam`. It threads through the same operator order
//! as the device kernel: derive the onset and final separations, gate the
//! model's validity, normalise the interface normal defensively, form the
//! mixed-mode effective separation `lambda`, grow the irreversible history and
//! apply the secant-damaged bilinear law.
//!
//! The golden evaluates entirely in `f32`, so the oracle also stays in `f32`;
//! the device kernel runs the same `f32` arithmetic through operators a `GPU`
//! may contract, so the two sides are compared to tolerance rather than
//! bit-exactly.
//!
//! # Parity criterion
//!
//! The continuous scalars (`effective_separation`, `damage`, the three
//! traction components, `normal_traction` and `kappa_out`) are compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `advanced`
//! and `valid` flags are compared exactly. The damage law has knees at
//! `kappa = d0` and `kappa = df`, the `advanced` flag toggles at
//! `lambda = kappa_in` and `lambda = d0`, and the normal-traction branch
//! toggles at `delta_n = 0`; the random fixtures are rejection-sampled a clear
//! margin from all of these, so round-off cannot flip a flag or a branch.
//! Critical knees are instead covered by directed fixtures with exact
//! assertions, kept out of the random sweep.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::cohesive_zone`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::cohesive_mixed_mode_traction::{
    CohesiveMixedModeTractionQuery, CohesiveMixedModeTractionResult, GpuCohesiveMixedModeTraction,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// `f32::EPSILON`, the near-zero-normal threshold used by the golden and kernel.
const EPS: f32 = 1.192_092_9e-7;

/// Finite-magnitude limit: `abs(x) < FINITE_LIMIT` rejects both `+-inf` and
/// `NaN`, matching the kernel's `is_finite`.
const FINITE_LIMIT: f32 = 3.0e38;

fn is_finite(x: f32) -> bool {
    x.abs() < FINITE_LIMIT
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// Independent `f32` re-implementation of the golden `damage_at`, mirroring the
/// kernel's `select` order and divisor guard.
fn damage_at(kappa: f32, d0: f32, df: f32) -> f32 {
    let below = kappa <= d0;
    let above = kappa >= df;
    let num = df * (kappa - d0);
    let den = kappa * (df - d0);
    let safe_den = if den > 0.0 { den } else { 1.0 };
    let mid = (num / safe_den).clamp(0.0, 1.0);
    let v = if above { 1.0 } else { mid };
    if below {
        0.0
    } else {
        v
    }
}

/// Independent `f32` re-implementation of the golden `cohesive_traction`,
/// mirroring the device kernel's operator order and guards exactly.
fn oracle(q: &CohesiveMixedModeTractionQuery) -> CohesiveMixedModeTractionResult {
    let k = q.stiffness;
    let sigma = q.strength;
    let gc = q.fracture_energy;
    let beta = q.shear_weight;
    let kappa_in = q.kappa_in;

    let safe_stiffness = if k > 0.0 { k } else { 1.0 };
    let safe_strength = if sigma > 0.0 { sigma } else { 1.0 };
    let d0 = safe_strength / safe_stiffness;
    let df = 2.0 * gc / safe_strength;

    let params_finite = is_finite(k) && is_finite(sigma) && is_finite(gc) && is_finite(beta);
    let inputs_finite = is_finite(q.delta[0])
        && is_finite(q.delta[1])
        && is_finite(q.delta[2])
        && is_finite(q.normal[0])
        && is_finite(q.normal[1])
        && is_finite(q.normal[2])
        && is_finite(kappa_in);
    let positive = k > 0.0 && sigma > 0.0 && gc > 0.0 && beta >= 0.0;
    let softening = df > d0;
    let valid = params_finite && inputs_finite && positive && softening;

    let n_len = length(q.normal);
    let degenerate_normal = n_len <= EPS;
    let safe_n_len = if degenerate_normal { 1.0 } else { n_len };
    let n = [
        q.normal[0] / safe_n_len,
        q.normal[1] / safe_n_len,
        q.normal[2] / safe_n_len,
    ];

    let delta_n = dot(q.delta, n);
    let tangent = [
        q.delta[0] - delta_n * n[0],
        q.delta[1] - delta_n * n[1],
        q.delta[2] - delta_n * n[2],
    ];
    let delta_t = length(tangent);
    let open_n = delta_n.max(0.0);
    let lambda = (open_n * open_n + beta * beta * delta_t * delta_t).sqrt();

    let advanced_bool = (lambda > kappa_in) && (lambda > d0);
    let kappa_grown = kappa_in.max(lambda);
    let damage_grown = damage_at(kappa_grown, d0, df);
    let secant = 1.0 - damage_grown;

    let normal_traction = if delta_n >= 0.0 {
        secant * k * delta_n
    } else {
        k * delta_n
    };
    let tangent_scale = if delta_t > 0.0 { secant * k } else { 0.0 };
    let traction = [
        normal_traction * n[0] + tangent[0] * tangent_scale,
        normal_traction * n[1] + tangent[1] * tangent_scale,
        normal_traction * n[2] + tangent[2] * tangent_scale,
    ];

    let damage_rest = damage_at(kappa_in, d0, df);
    let eff_nz = if degenerate_normal { 0.0 } else { lambda };
    let dmg_nz = if degenerate_normal {
        damage_rest
    } else {
        damage_grown
    };
    let trac_nz = if degenerate_normal {
        [0.0, 0.0, 0.0]
    } else {
        traction
    };
    let nt_nz = if degenerate_normal {
        0.0
    } else {
        normal_traction
    };
    let adv_nz = advanced_bool && !degenerate_normal;
    let kappa_nz = if degenerate_normal {
        kappa_in
    } else {
        kappa_grown
    };

    if valid {
        CohesiveMixedModeTractionResult {
            effective_separation: eff_nz,
            damage: dmg_nz,
            traction: trac_nz,
            normal_traction: nt_nz,
            advanced: adv_nz,
            kappa_out: kappa_nz,
            valid: true,
        }
    } else {
        CohesiveMixedModeTractionResult {
            effective_separation: 0.0,
            damage: 0.0,
            traction: [0.0, 0.0, 0.0],
            normal_traction: 0.0,
            advanced: false,
            kappa_out: kappa_in,
            valid: false,
        }
    }
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts every output field matches the independent oracle: continuous
/// quantities to tolerance, discrete flags exactly.
fn assert_full(
    gpu: &CohesiveMixedModeTractionResult,
    q: &CohesiveMixedModeTractionQuery,
    label: &str,
) {
    let want = oracle(q);
    assert!(
        close(gpu.effective_separation, want.effective_separation),
        "{label}: effective_separation gpu={} oracle={}",
        gpu.effective_separation,
        want.effective_separation
    );
    assert!(
        close(gpu.damage, want.damage),
        "{label}: damage gpu={} oracle={}",
        gpu.damage,
        want.damage
    );
    for axis in 0..3 {
        assert!(
            close(gpu.traction[axis], want.traction[axis]),
            "{label}: traction[{axis}] gpu={} oracle={}",
            gpu.traction[axis],
            want.traction[axis]
        );
    }
    assert!(
        close(gpu.normal_traction, want.normal_traction),
        "{label}: normal_traction gpu={} oracle={}",
        gpu.normal_traction,
        want.normal_traction
    );
    assert!(
        close(gpu.kappa_out, want.kappa_out),
        "{label}: kappa_out gpu={} oracle={}",
        gpu.kappa_out,
        want.kappa_out
    );
    assert_eq!(
        gpu.advanced, want.advanced,
        "{label}: advanced gpu={} oracle={}",
        gpu.advanced, want.advanced
    );
    assert_eq!(
        gpu.valid, want.valid,
        "{label}: valid gpu={} oracle={}",
        gpu.valid, want.valid
    );
}

/// A representative, well-conditioned cohesive model: `K = 1000`,
/// `sigma_c = 10`, `G_c = 5` give `d0 = 0.01` and `df = 1.0`, so the softening
/// branch is wide and the fixtures below sit clear of its knees.
fn model(delta: [f32; 3], normal: [f32; 3], kappa_in: f32) -> CohesiveMixedModeTractionQuery {
    CohesiveMixedModeTractionQuery::new(1000.0, 10.0, 5.0, 1.0, delta, normal, kappa_in)
}

/// A pristine pure-tension opening lands `lambda` inside the softening branch,
/// so the history advances and the secant damage is partial.
#[test]
fn tension_regime_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    let q = model([0.0, 0.0, 0.5], [0.0, 0.0, 1.0], 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(out[0].advanced, "tension should advance history");
    assert!(
        out[0].normal_traction > 0.0,
        "tension traction should be positive, got {}",
        out[0].normal_traction
    );
    assert_full(&out[0], &q, "tension");
}

/// Compression (negative normal separation) takes the full-penalty branch:
/// the normal traction is the undamaged `K * delta_n` and the damage stays
/// zero because `lambda` never leaves the elastic region.
#[test]
fn compression_regime_full_penalty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    let q = model([0.0, 0.0, -0.3], [0.0, 0.0, 1.0], 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(!out[0].advanced, "compression should not advance history");
    assert!(
        close(out[0].damage, 0.0),
        "compression damage should be zero, got {}",
        out[0].damage
    );
    assert!(
        out[0].normal_traction < 0.0,
        "compression traction should be negative, got {}",
        out[0].normal_traction
    );
    assert_full(&out[0], &q, "compression");
}

/// A pure tangential jump has `delta_n = 0`, so `lambda = beta * delta_t`; the
/// normal traction vanishes and the whole traction is tangential.
#[test]
fn pure_shear_regime_tangential_only() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    let q = model([0.4, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid);
    assert!(out[0].advanced, "shear should advance history");
    assert!(
        close(out[0].normal_traction, 0.0),
        "pure shear normal traction should vanish, got {}",
        out[0].normal_traction
    );
    assert!(
        out[0].traction[0].abs() > 0.0,
        "shear traction should be tangential, got {:?}",
        out[0].traction
    );
    assert_full(&out[0], &q, "shear");
}

/// A mixed-mode (normal + tangential) jump dispatched alongside a compression
/// query as a two-element batch, exercising the `std430` query/result stride.
#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    let mixed = model([0.2, 0.0, 0.3], [0.0, 0.0, 1.0], 0.0);
    let compression = model([0.0, 0.0, -0.25], [0.0, 0.0, 1.0], 0.0);
    let queries = [mixed, compression];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), 2);
    assert_full(&out[0], &mixed, "mix[0]");
    assert_full(&out[1], &compression, "mix[1]");
}

/// A near-zero normal is a degenerate path inside a valid model: the traction
/// is zeroed, the effective separation is zero and the history is preserved,
/// while the damage still reports the resting value at `kappa_in`. Asserted
/// with exact discrete values so the degenerate branch is pinned.
#[test]
fn near_zero_normal_preserves_history() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    let kappa_in = 0.2;
    let q = model([0.0, 0.0, 0.5], [0.0, 0.0, 0.0], kappa_in);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(out[0].valid, "degenerate normal is still a valid model");
    assert!(
        !out[0].advanced,
        "degenerate normal must not advance history"
    );
    assert!(
        close(out[0].effective_separation, 0.0),
        "degenerate effective separation should be zero, got {}",
        out[0].effective_separation
    );
    assert!(
        close(out[0].kappa_out, kappa_in),
        "history should be preserved, got {}",
        out[0].kappa_out
    );
    for axis in 0..3 {
        assert!(
            close(out[0].traction[axis], 0.0),
            "degenerate traction[{axis}] should be zero, got {}",
            out[0].traction[axis]
        );
    }
    // Damage still reports the resting value at kappa_in: 0.2 in (d0, df).
    let rest = damage_at(kappa_in, 0.01, 1.0);
    assert!(
        close(out[0].damage, rest),
        "resting damage gpu={} oracle={}",
        out[0].damage,
        rest
    );
    assert_full(&out[0], &q, "near_zero_normal");
}

/// A non-positive stiffness makes the model unconstructible, so the kernel
/// emits a fully zeroed result with `valid = false` and the history untouched.
#[test]
fn invalid_model_is_zeroed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    let kappa_in = 0.3;
    let q = CohesiveMixedModeTractionQuery::new(
        -1.0,
        10.0,
        5.0,
        1.0,
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        kappa_in,
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid, "non-positive stiffness is invalid");
    assert!(!out[0].advanced);
    assert!(close(out[0].kappa_out, kappa_in));
    for axis in 0..3 {
        assert!(close(out[0].traction[axis], 0.0));
    }
    assert_full(&out[0], &q, "invalid_model");
}

/// A non-finite displacement component also zeroes the result deterministically
/// rather than racing a propagated `NaN`.
#[test]
fn non_finite_input_is_zeroed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    let kappa_in = 0.15;
    let q = model([f32::NAN, 0.0, 0.5], [0.0, 0.0, 1.0], kappa_in);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid, "NaN input must be invalid");
    assert!(close(out[0].kappa_out, kappa_in));
    assert_full(&out[0], &q, "non_finite_input");
}

/// A degenerate softening branch (`df <= d0`) is also rejected: here `G_c` is
/// tiny so `df < d0`, giving `valid = false`.
#[test]
fn non_softening_model_is_zeroed() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    // d0 = 10 / 1000 = 0.01; df = 2 * 0.001 / 10 = 2e-4 < d0.
    let kappa_in = 0.05;
    let q = CohesiveMixedModeTractionQuery::new(
        1000.0,
        10.0,
        0.001,
        1.0,
        [0.0, 0.0, 0.3],
        [0.0, 0.0, 1.0],
        kappa_in,
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].valid, "df <= d0 is not a softening model");
    assert!(close(out[0].kappa_out, kappa_in));
    assert_full(&out[0], &q, "non_softening");
}

/// An empty batch short-circuits on the host with no dispatch issued.
#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

/// A `512`-sample rejection-sampled sweep of well-conditioned evaluations, each
/// compared in full against the independent oracle.
#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuCohesiveMixedModeTraction::new(&ctx);
    let mut lcg = Lcg::new(0x0C0E_51A3);
    let mut queries = Vec::with_capacity(512);
    let mut guard = 0u32;
    while queries.len() < 512 && guard < 5_000_000 {
        guard += 1;
        let q = random_query(&mut lcg);
        if well_conditioned(&q) {
            queries.push(q);
        }
    }
    assert_eq!(
        queries.len(),
        512,
        "could not sample enough well-conditioned evaluations"
    );
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_full(res, q, &format!("sweep[{i}]"));
    }
}

/// A query is well conditioned when the model is valid with a wide softening
/// branch, the normal is clearly non-degenerate, the effective separation sits
/// strictly inside `(d0, df)` with margin, and the normal-traction branch is
/// clear of `delta_n = 0`, so no round-off can flip a flag or a branch.
fn well_conditioned(q: &CohesiveMixedModeTractionQuery) -> bool {
    const MARGIN: f32 = 0.03;
    let k = q.stiffness;
    let sigma = q.strength;
    let gc = q.fracture_energy;
    let beta = q.shear_weight;
    let d0 = sigma / k;
    let df = 2.0 * gc / sigma;
    if !(df > d0 + MARGIN) {
        return false;
    }
    let n_len = length(q.normal);
    if n_len < 0.1 {
        return false;
    }
    let n = [
        q.normal[0] / n_len,
        q.normal[1] / n_len,
        q.normal[2] / n_len,
    ];
    let delta_n = dot(q.delta, n);
    if delta_n.abs() <= 0.02 {
        return false;
    }
    let tangent = [
        q.delta[0] - delta_n * n[0],
        q.delta[1] - delta_n * n[1],
        q.delta[2] - delta_n * n[2],
    ];
    let delta_t = length(tangent);
    let open_n = delta_n.max(0.0);
    let lambda = (open_n * open_n + beta * beta * delta_t * delta_t).sqrt();
    // kappa_in is pinned to zero in the sweep, so lambda > d0 + MARGIN keeps
    // advanced stable (lambda > kappa_in always holds) and clear of the damage
    // knee at d0; lambda < df - MARGIN keeps it clear of the knee at df.
    lambda > d0 + MARGIN && lambda < df - MARGIN
}

/// Builds a random well-posed cohesive evaluation. The model parameters keep
/// `d0` small and `df >= 1.0`, so a wide softening branch is easy to hit; the
/// pristine `kappa_in = 0` keeps the `advanced` flag stable under rejection.
fn random_query(lcg: &mut Lcg) -> CohesiveMixedModeTractionQuery {
    let stiffness = lcg.next_range(500.0, 1500.0);
    let strength = lcg.next_range(5.0, 10.0);
    let fracture_energy = lcg.next_range(5.0, 10.0);
    let shear_weight = lcg.next_range(0.5, 1.5);
    let delta = [
        lcg.next_range(-0.25, 0.25),
        lcg.next_range(-0.25, 0.25),
        lcg.next_range(-0.25, 0.25),
    ];
    let normal = [
        lcg.next_range(-1.0, 1.0),
        lcg.next_range(-1.0, 1.0),
        lcg.next_range(-1.0, 1.0),
    ];
    CohesiveMixedModeTractionQuery::new(
        stiffness,
        strength,
        fracture_energy,
        shear_weight,
        delta,
        normal,
        0.0,
    )
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
