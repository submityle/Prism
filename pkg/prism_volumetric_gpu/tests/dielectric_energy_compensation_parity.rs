//! Real-device parity for the rough-dielectric energy-compensation twin:
//! [`GpuDielectricEnergy`](prism_volumetric_gpu::dielectric_energy_compensation::GpuDielectricEnergy)
//! must reproduce the `CPU` golden `single_scatter_albedo`, `smooth_albedo` and
//! `compensation_factor` of
//! `prism_render_architecture::reference_pt::dielectric_energy`, which add the
//! multiple-scattering energy a single-scatter rough dielectric drops back with
//! a Turquin-style multiplicative compensation.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the trilinear baked-albedo lookup with its shared cell-centred axis-weight
//! clamping, the inlined dielectric `Fresnel` reflectance, the smooth ceiling
//! `R + (1 - R) / eta^2` and the clamped compensation ratio — carrying its own
//! copy of the baked `ALBEDO` table, so the test never imports
//! `prism_render_architecture`.
//!
//! The fixtures cover the baked table's corner nodes, the interior where all
//! three axes interpolate, the clamped out-of-range `eta` / `alpha` / `mu`
//! regime, and both ends of the compensation clamp. A sweep over random
//! `(eta, alpha, mu)` triples follows, plus an empty batch the host
//! short-circuits with no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! # Parity criterion
//!
//! Every continuous channel threads through multiplies, adds, guarded divisions
//! and `sqrt`, so `CPU` and `GPU` evaluate the same closed form but need not be
//! bit-exact. The continuous comparison is `abs_diff <= 1e-4 || rel_diff <=
//! 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag is compared exactly.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::dielectric_energy`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::dielectric_energy_compensation::{
    DielectricEnergyQuery, GpuDielectricEnergy,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Smallest relative index covered by the baked table, matching the kernel.
const ETA_MIN: f32 = 1.1;
/// Largest relative index covered by the baked table, matching the kernel.
const ETA_MAX: f32 = 2.5;
/// Upper bound on the compensation boost, matching the kernel.
const MAX_FACTOR: f32 = 3.0;
/// Floor on the albedo denominator, matching the kernel.
const MIN_ALBEDO: f32 = 0.05;
/// Table axis extents, matching the kernel.
const ETA_SIZE: usize = 6;
const ALPHA_SIZE: usize = 8;
const MU_SIZE: usize = 8;

/// Independent copy of the golden baked single-scatter directional-albedo table,
/// flattened in eta-major, alpha-middle, mu-minor order (flat = e*64 + a*8 + m).
const DE_ALBEDO: [f32; 384] = [
    0.73804, 0.72718, 0.70504, 0.68996, 0.68273, 0.67986, 0.67868, 0.67847, 0.67819, 0.66063,
    0.66459, 0.66876, 0.67154, 0.67329, 0.67530, 0.67679, 0.63255, 0.61704, 0.62857, 0.64182,
    0.65290, 0.66077, 0.66771, 0.67289, 0.59579, 0.57771, 0.59305, 0.61321, 0.63071, 0.64638,
    0.65782, 0.66684, 0.56644, 0.54256, 0.55957, 0.58326, 0.60728, 0.62816, 0.64613, 0.66011,
    0.54392, 0.51456, 0.52964, 0.55602, 0.58240, 0.60942, 0.63210, 0.65129, 0.52451, 0.49156,
    0.50505, 0.53054, 0.55992, 0.58877, 0.61684, 0.64212, 0.50854, 0.47271, 0.48259, 0.50765,
    0.53701, 0.56815, 0.60143, 0.63192, 0.65676, 0.61244, 0.56193, 0.52658, 0.50977, 0.49948,
    0.49490, 0.49385, 0.56366, 0.52537, 0.51175, 0.50315, 0.49764, 0.49274, 0.49068, 0.49045,
    0.51299, 0.48429, 0.47761, 0.47600, 0.47959, 0.48099, 0.48336, 0.48538, 0.48170, 0.45173,
    0.44853, 0.45234, 0.45825, 0.46646, 0.47150, 0.47668, 0.45567, 0.42350, 0.42177, 0.42938,
    0.43955, 0.44983, 0.45865, 0.46663, 0.43544, 0.39949, 0.39830, 0.40775, 0.41927, 0.43207,
    0.44479, 0.45655, 0.42032, 0.37967, 0.37747, 0.38592, 0.39928, 0.41472, 0.43049, 0.44473,
    0.40553, 0.36254, 0.35795, 0.36687, 0.38155, 0.39722, 0.41595, 0.43329, 0.61221, 0.54882,
    0.48601, 0.44044, 0.41705, 0.40343, 0.39550, 0.39446, 0.49421, 0.45177, 0.42977, 0.41477,
    0.40395, 0.39652, 0.39374, 0.39068, 0.44484, 0.40854, 0.39311, 0.38873, 0.38425, 0.38276,
    0.38200, 0.38405, 0.41196, 0.37761, 0.36762, 0.36440, 0.36465, 0.36773, 0.36937, 0.37303,
    0.38917, 0.35355, 0.34401, 0.34386, 0.34599, 0.34974, 0.35515, 0.36001, 0.37090, 0.33117,
    0.32275, 0.32257, 0.32640, 0.33309, 0.34144, 0.34728, 0.35506, 0.31381, 0.30300, 0.30334,
    0.30772, 0.31619, 0.32520, 0.33350, 0.34288, 0.29831, 0.28736, 0.28733, 0.29234, 0.30100,
    0.31042, 0.31970, 0.58283, 0.50853, 0.43915, 0.39572, 0.36744, 0.35531, 0.34514, 0.34416,
    0.45867, 0.40587, 0.38242, 0.36633, 0.35174, 0.34327, 0.34277, 0.33954, 0.40501, 0.36791,
    0.34772, 0.33678, 0.33301, 0.33093, 0.32962, 0.32888, 0.37290, 0.33664, 0.32120, 0.31619,
    0.31420, 0.31193, 0.31446, 0.31496, 0.35083, 0.31323, 0.29959, 0.29401, 0.29288, 0.29462,
    0.29643, 0.29883, 0.33221, 0.29429, 0.27987, 0.27513, 0.27477, 0.27660, 0.27941, 0.28281,
    0.31869, 0.27848, 0.26265, 0.25585, 0.25778, 0.25940, 0.26442, 0.26758, 0.30774, 0.26178,
    0.24686, 0.24017, 0.24037, 0.24277, 0.24788, 0.25407, 0.55666, 0.48340, 0.41282, 0.37065,
    0.34581, 0.33026, 0.32248, 0.32003, 0.43423, 0.38121, 0.35683, 0.33696, 0.32854, 0.32124,
    0.31570, 0.31621, 0.38391, 0.34101, 0.32143, 0.31175, 0.30541, 0.30074, 0.29998, 0.30066,
    0.35112, 0.31141, 0.29669, 0.28986, 0.28259, 0.28219, 0.28211, 0.28364, 0.33069, 0.29226,
    0.27481, 0.26652, 0.26231, 0.26330, 0.26436, 0.26385, 0.31504, 0.27339, 0.25533, 0.24895,
    0.24314, 0.24430, 0.24539, 0.24630, 0.30163, 0.25786, 0.23929, 0.22966, 0.22741, 0.22502,
    0.22693, 0.22752, 0.29161, 0.24463, 0.22560, 0.21425, 0.21105, 0.20930, 0.21211, 0.21282,
    0.54112, 0.46424, 0.39925, 0.35671, 0.33521, 0.32270, 0.31595, 0.31149, 0.41664, 0.36889,
    0.33929, 0.32711, 0.31922, 0.31135, 0.30705, 0.30905, 0.36914, 0.32728, 0.30835, 0.29885,
    0.29197, 0.28901, 0.28941, 0.29049, 0.33972, 0.30283, 0.28532, 0.27335, 0.26925, 0.26807,
    0.26803, 0.26827, 0.32112, 0.28204, 0.26536, 0.25333, 0.25027, 0.24650, 0.24578, 0.24785,
    0.30642, 0.26503, 0.24605, 0.23421, 0.22988, 0.22622, 0.22448, 0.22564, 0.29376, 0.24856,
    0.22814, 0.21577, 0.21107, 0.20771, 0.20649, 0.20585, 0.28420, 0.23535, 0.21348, 0.19920,
    0.19338, 0.18936, 0.18748, 0.18784,
];

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Independent oracle for the normalized `eta` table coordinate in `[0, 1]`.
fn eta_unit(eta: f32) -> f32 {
    ((eta - ETA_MIN) / (ETA_MAX - ETA_MIN)).clamp(0.0, 1.0)
}

/// Independent oracle for the shared cell-centred axis-weight clamping of a
/// length-`n` table axis, returning `(lo, hi, frac)`.
fn axis_weights(x: f32, n: usize) -> (usize, usize, f32) {
    let t = (x * n as f32 - 0.5).clamp(0.0, (n - 1) as f32);
    let lo = t as usize;
    let hi = (lo + 1).min(n - 1);
    (lo, hi, t - lo as f32)
}

/// Independent oracle for a flattened lookup into the baked albedo table.
fn fetch_albedo(e: usize, a: usize, m: usize) -> f32 {
    DE_ALBEDO[e * 64 + a * 8 + m]
}

/// Linear interpolation between `a` and `b` by `t`.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Independent oracle for the trilinear single-scatter directional albedo.
fn single_scatter_albedo(eta: f32, alpha: f32, mu: f32) -> f32 {
    let (e_lo, e_hi, e_f) = axis_weights(eta_unit(eta), ETA_SIZE);
    let (a_lo, a_hi, a_f) = axis_weights(alpha, ALPHA_SIZE);
    let (m_lo, m_hi, m_f) = axis_weights(mu, MU_SIZE);
    let along_mu =
        |e: usize, a: usize| lerp(fetch_albedo(e, a, m_lo), fetch_albedo(e, a, m_hi), m_f);
    let along_alpha = |e: usize| lerp(along_mu(e, a_lo), along_mu(e, a_hi), a_f);
    lerp(along_alpha(e_lo), along_alpha(e_hi), e_f).clamp(0.0, 1.0)
}

/// Independent oracle for the exact unpolarized dielectric `Fresnel`
/// reflectance (`eta_i -> eta_t`).
fn fresnel_dielectric(cos_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let eta = eta_i / eta_t;
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = eta * eta * sin2_i;
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    let r_parl = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    let r_perp = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// Independent oracle for the smooth-interface ceiling.
fn smooth_albedo(eta: f32, mu: f32) -> f32 {
    let r = fresnel_dielectric(mu, 1.0, eta);
    r + (1.0 - r) / (eta * eta)
}

/// Independent oracle for the clamped multiplicative compensation factor.
fn compensation_factor(eta: f32, alpha: f32, mu: f32) -> f32 {
    let e = single_scatter_albedo(eta, alpha, mu).max(MIN_ALBEDO);
    let ceiling = smooth_albedo(eta, mu);
    (ceiling / e).clamp(1.0, MAX_FACTOR)
}

/// The independent oracle for one query: `(e_ss, e_smooth, comp, valid)`.
fn oracle(q: &DielectricEnergyQuery) -> (f32, f32, f32, u32) {
    (
        single_scatter_albedo(q.eta, q.alpha, q.mu),
        smooth_albedo(q.eta, q.mu),
        compensation_factor(q.eta, q.alpha, q.mu),
        1u32,
    )
}

/// Dispatches one query and asserts every channel plus the validity flag.
fn assert_parity(ctx: &GpuContext, gpu: &GpuDielectricEnergy, q: DielectricEnergyQuery) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (e_ss, e_smooth, comp, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    assert!(
        close(r.e_ss, e_ss),
        "e_ss mismatch: gpu={} cpu={e_ss} query={q:?}",
        r.e_ss
    );
    assert!(
        close(r.e_smooth, e_smooth),
        "e_smooth mismatch: gpu={} cpu={e_smooth} query={q:?}",
        r.e_smooth
    );
    assert!(
        close(r.comp, comp),
        "comp mismatch: gpu={} cpu={comp} query={q:?}",
        r.comp
    );
}

#[test]
fn table_corner_node_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    // eta/alpha/mu landing exactly on the first baked node (the (i+0.5)/N cell
    // centres), so the trilinear lookup reduces to a single table fetch.
    let eta = ETA_MIN + (ETA_MAX - ETA_MIN) * 0.5 / ETA_SIZE as f32;
    let alpha = 0.5 / ALPHA_SIZE as f32;
    let mu = 0.5 / MU_SIZE as f32;
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(eta, alpha, mu));
}

#[test]
fn table_far_corner_node_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    // The opposite baked corner: the last cell centre on every axis.
    let eta = ETA_MIN + (ETA_MAX - ETA_MIN) * (ETA_SIZE as f32 - 0.5) / ETA_SIZE as f32;
    let alpha = (ALPHA_SIZE as f32 - 0.5) / ALPHA_SIZE as f32;
    let mu = (MU_SIZE as f32 - 0.5) / MU_SIZE as f32;
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(eta, alpha, mu));
}

#[test]
fn axis_midpoint_interpolation_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    // A well-conditioned interior point so all three axes interpolate.
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(1.72, 0.43, 0.57));
}

#[test]
fn eta_below_range_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    // eta below ETA_MIN clamps to the nearest baked node.
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(0.7, 0.4, 0.6));
}

#[test]
fn eta_above_range_clamps() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    // eta above ETA_MAX clamps to the nearest baked node.
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(3.4, 0.4, 0.6));
}

#[test]
fn mu_endpoints_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    // mu at both ends of [0, 1], clamped to the first/last baked node.
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(1.5, 0.4, 0.0));
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(1.5, 0.4, 1.0));
}

#[test]
fn alpha_endpoints_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    // alpha at both ends of [0, 1], clamped to the first/last baked node.
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(1.5, 0.0, 0.6));
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(1.5, 1.0, 0.6));
}

#[test]
fn compensation_lower_clamp_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    // A smooth, near-normal low-roughness interface: the ceiling barely exceeds
    // the albedo, so the compensation sits against its lower clamp of 1.
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(1.2, 0.02, 0.95));
}

#[test]
fn compensation_upper_clamp_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    // A high-index grazing interface where the baked albedo hits MIN_ALBEDO and
    // the ceiling is large, driving the compensation against its upper clamp.
    assert_parity(&ctx, &gpu, DielectricEnergyQuery::new(2.5, 1.0, 0.02));
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
    let gpu = GpuDielectricEnergy::new(&ctx);
    let mut rng = Lcg::new(0x51_7C_C1_93);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        let eta = rng.next_range(1.1, 2.5);
        let alpha = rng.next_range(0.02, 1.0);
        let mu = rng.next_range(0.05, 1.0);
        queries.push(DielectricEnergyQuery::new(eta, alpha, mu));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (e_ss, e_smooth, comp, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close(r.e_ss, e_ss),
            "sweep e_ss mismatch: gpu={} cpu={e_ss} query={q:?}",
            r.e_ss
        );
        assert!(
            close(r.e_smooth, e_smooth),
            "sweep e_smooth mismatch: gpu={} cpu={e_smooth} query={q:?}",
            r.e_smooth
        );
        assert!(
            close(r.comp, comp),
            "sweep comp mismatch: gpu={} cpu={comp} query={q:?}",
            r.comp
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuDielectricEnergy::new(&ctx);
    assert!(
        gpu.evaluate(&ctx, &[]).is_empty(),
        "empty batch returns an empty vector with no dispatch"
    );
}
