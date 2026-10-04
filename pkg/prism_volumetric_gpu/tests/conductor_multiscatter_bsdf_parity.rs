//! Real-device parity for the energy-conserving rough-conductor twin:
//! [`GpuConductorMultiscatterBsdf`](prism_volumetric_gpu::conductor_multiscatter_bsdf::GpuConductorMultiscatterBsdf)
//! must reproduce the `CPU` golden `MultiscatterConductor::evaluate` and
//! `MultiscatterConductor::pdf` of
//! `prism_render_architecture::reference_pt::conductor_ms`, which wrap the
//! exact complex-index single-scatter `GGX` conductor with the Kulla-Conty
//! multiple-scattering compensation lobe tinted by the metal's average
//! `Fresnel` reflectance.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the exact unpolarized conductor `Fresnel`, the `32`-node midpoint
//! hemispherical-average `Fresnel`, the isotropic `GGX` `distribution`, the
//! Smith `lambda` feeding `g1`/`g2`, the baked `16x16` directional albedo and
//! `16`-entry average albedo tables with their bilinear/linear lookups, and the
//! Kulla-Conty lobe and tint — written out directly so the test never imports
//! `prism_render_architecture`. It mirrors the reference branch for branch,
//! including the below-horizon and degenerate half-vector guards: the
//! single-scatter term self-zeros on a degenerate half vector while the
//! compensation lobe still contributes, and `valid` is `cos_o > 0 && cos_i > 0`
//! only.
//!
//! The fixtures cover normal incidence, oblique configurations, a nearly-smooth
//! and a very-rough lobe, measured copper/aluminium metals, two below-horizon
//! pairs that must report `valid = 0`, and a `>= 3`-element mixed valid/invalid
//! batch dispatched in one call so any `std430` stride bug in the packed result
//! would surface. A `512`-query sweep over random metals, roughness and
//! hemisphere directions follows, plus an empty batch the host short-circuits
//! with no dispatch.
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
//! `valid` flag is compared exactly. The sweep builds each direction from an
//! explicit cosine about the shading normal, keeping both cosines comfortably
//! positive so parity never sits on a branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::conductor_ms`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::conductor_multiscatter_bsdf::{
    ConductorMultiscatterBsdfQuery, GpuConductorMultiscatterBsdf,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_EPS: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Minimum half-vector pre-normalization squared length, matching the kernel.
const EPS_LEN_SQ: f32 = 1.0e-12;
/// Minimum `GGX` lobe width, matching the kernel.
const MIN_ALPHA: f32 = 0.001;
/// Reciprocal of pi, matching the kernel's baked constant usage.
const INV_PI: f32 = std::f32::consts::FRAC_1_PI;
/// Number of midpoint nodes for the average-`Fresnel` quadrature.
const AVG_FRESNEL_NODES: u32 = 32;

/// The reference metal used across several named fixtures: a measured complex
/// index for a warm conductor.
const GOLD_ETA: [f32; 3] = [0.143, 0.375, 1.442];
/// Extinction coefficient paired with [`GOLD_ETA`].
const GOLD_K: [f32; 3] = [3.983, 2.386, 1.603];

/// Baked single-scatter directional albedo `E(mu, alpha)`, row-major
/// `idx = a * 16 + c`, copied verbatim from the golden `ggx_energy` table so
/// the oracle stays independent of `prism_render_architecture`.
const ALBEDO: [f32; 256] = [
    0.89053, 0.94328, 0.97589, 0.98818, 0.99206, 0.99439, 0.99607, 0.99685, 0.99744, 0.99806,
    0.99817, 0.99817, 0.99858, 0.99886, 0.99885, 0.99904, 0.93451, 0.88498, 0.89344, 0.91887,
    0.93761, 0.95195, 0.96259, 0.96975, 0.97565, 0.97898, 0.98269, 0.98372, 0.98662, 0.98678,
    0.98818, 0.98973, 0.94954, 0.88988, 0.87099, 0.87141, 0.88501, 0.89715, 0.91246, 0.92474,
    0.93381, 0.94122, 0.94994, 0.95394, 0.95931, 0.96293, 0.96593, 0.96903, 0.95385, 0.89693,
    0.86369, 0.85375, 0.85025, 0.85593, 0.86688, 0.87682, 0.88543, 0.89518, 0.90309, 0.91140,
    0.91951, 0.92397, 0.93080, 0.93376, 0.95163, 0.89433, 0.86047, 0.83623, 0.82537, 0.82326,
    0.82529, 0.83233, 0.83790, 0.84373, 0.85562, 0.86116, 0.87050, 0.87715, 0.88401, 0.88939,
    0.94932, 0.88946, 0.84891, 0.82225, 0.80592, 0.79468, 0.79105, 0.79034, 0.79559, 0.80038,
    0.80365, 0.80971, 0.81667, 0.82446, 0.82961, 0.83616, 0.94567, 0.88040, 0.83628, 0.80512,
    0.78428, 0.76896, 0.76043, 0.75496, 0.75359, 0.75547, 0.75319, 0.75845, 0.76217, 0.76699,
    0.77263, 0.77446, 0.94142, 0.87175, 0.82330, 0.79018, 0.76237, 0.74273, 0.73166, 0.72252,
    0.71253, 0.70897, 0.70770, 0.70686, 0.70679, 0.70787, 0.71245, 0.71734, 0.93671, 0.86121,
    0.80691, 0.76759, 0.73937, 0.72034, 0.69780, 0.68750, 0.67520, 0.66904, 0.66345, 0.66195,
    0.65407, 0.65477, 0.65289, 0.65404, 0.92915, 0.84826, 0.79412, 0.75430, 0.71678, 0.69452,
    0.66872, 0.65459, 0.64058, 0.62912, 0.62176, 0.61331, 0.60612, 0.60542, 0.60027, 0.59651,
    0.92425, 0.83505, 0.77650, 0.73150, 0.69712, 0.66509, 0.64125, 0.61974, 0.60482, 0.59115,
    0.57870, 0.56560, 0.56079, 0.55360, 0.54713, 0.54434, 0.91756, 0.82414, 0.75863, 0.71164,
    0.67160, 0.63993, 0.61433, 0.59389, 0.57071, 0.55353, 0.53875, 0.52792, 0.51651, 0.50474,
    0.50088, 0.49233, 0.91143, 0.81112, 0.74357, 0.68803, 0.65290, 0.61420, 0.58876, 0.55983,
    0.53882, 0.51787, 0.50314, 0.48608, 0.47485, 0.46401, 0.45315, 0.44662, 0.90468, 0.79955,
    0.73071, 0.67325, 0.62805, 0.59150, 0.55480, 0.52925, 0.50862, 0.48782, 0.46760, 0.45411,
    0.43806, 0.42644, 0.41378, 0.40239, 0.89889, 0.78596, 0.71155, 0.65008, 0.60615, 0.56783,
    0.53284, 0.50402, 0.47753, 0.45741, 0.43591, 0.41926, 0.39868, 0.38978, 0.37722, 0.36418,
    0.89346, 0.77409, 0.69544, 0.63218, 0.58563, 0.54142, 0.50747, 0.47595, 0.44834, 0.42990,
    0.40566, 0.38615, 0.36961, 0.35676, 0.34430, 0.32918,
];

/// Baked cosine-weighted average albedo `E_avg(alpha)` per alpha node, copied
/// verbatim from the golden `ggx_energy` table.
const AVG_ALBEDO: [f32; 16] = [
    0.99608, 0.97482, 0.94267, 0.90348, 0.86012, 0.81500, 0.76850, 0.72253, 0.67764, 0.63568,
    0.59390, 0.55526, 0.51834, 0.48476, 0.45266, 0.42318,
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

/// Returns `true` when two three-channel values agree channel-wise.
fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Dot product of two three-vectors.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two three-vectors.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
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

/// Exact unpolarized conductor `Fresnel` reflectance for one channel.
fn fresnel_channel(cos_theta: f32, eta: f32, k: f32) -> f32 {
    let cos_i = cos_theta.clamp(0.0, 1.0);
    let cos2 = cos_i * cos_i;
    let sin2 = 1.0 - cos2;
    let eta2 = eta * eta;
    let k2 = k * k;
    let t0 = eta2 - k2 - sin2;
    let a2b2 = (t0 * t0 + 4.0 * eta2 * k2).max(0.0).sqrt();
    let a = (0.5 * (a2b2 + t0)).max(0.0).sqrt();
    let t1 = a2b2 + cos2;
    let t2 = 2.0 * a * cos_i;
    let denom_s = t1 + t2;
    let r_s = if denom_s > 0.0 {
        (t1 - t2) / denom_s
    } else {
        1.0
    };
    let t3 = cos2 * a2b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let denom_p = t3 + t4;
    let r_p = if denom_p > 0.0 {
        r_s * (t3 - t4) / denom_p
    } else {
        r_s
    };
    (0.5 * (r_s + r_p)).clamp(0.0, 1.0)
}

/// Per-channel conductor `Fresnel` reflectance.
fn fresnel(eta: [f32; 3], k: [f32; 3], cos_theta: f32) -> [f32; 3] {
    [
        fresnel_channel(cos_theta, eta[0], k[0]),
        fresnel_channel(cos_theta, eta[1], k[1]),
        fresnel_channel(cos_theta, eta[2], k[2]),
    ]
}

/// Hemispherical cosine-weighted average `Fresnel` via `32`-node midpoint
/// quadrature.
fn average_fresnel(eta: [f32; 3], k: [f32; 3]) -> [f32; 3] {
    let mut acc = [0.0f32; 3];
    for i in 0..AVG_FRESNEL_NODES {
        let mu = (i as f32 + 0.5) / AVG_FRESNEL_NODES as f32;
        let w = 2.0 * mu / AVG_FRESNEL_NODES as f32;
        let f = fresnel(eta, k, mu);
        acc[0] += f[0] * w;
        acc[1] += f[1] * w;
        acc[2] += f[2] * w;
    }
    acc
}

/// Isotropic `GGX` normal distribution `D(h)` from the half-vector cosine.
fn ggx_distribution(cos_h: f32, alpha: f32) -> f32 {
    if cos_h <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let c2 = cos_h * cos_h;
    let denom = c2 * (a2 - 1.0) + 1.0;
    a2 * INV_PI / (denom * denom)
}

/// Smith `Lambda` auxiliary for a direction whose cosine to the normal is
/// `cos_w`.
fn ggx_lambda(cos_w: f32, alpha: f32) -> f32 {
    let c = cos_w.abs();
    if c >= 1.0 {
        return 0.0;
    }
    let c2 = c * c;
    let tan2 = (1.0 - c2) / c2;
    let a2 = alpha * alpha;
    0.5 * ((1.0 + a2 * tan2).sqrt() - 1.0)
}

/// Smith `G1` masking term.
fn ggx_g1(cos_w: f32, alpha: f32) -> f32 {
    1.0 / (1.0 + ggx_lambda(cos_w, alpha))
}

/// Smith height-correlated `G2` masking-shadowing term.
fn ggx_g2(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
    1.0 / (1.0 + ggx_lambda(cos_o, alpha) + ggx_lambda(cos_i, alpha))
}

/// `GGX` solid-angle reflection density for the one-sample `MIS` mix.
fn ggx_reflection_pdf(cos_o: f32, cos_h: f32, alpha: f32) -> f32 {
    if cos_o <= 0.0 {
        return 0.0;
    }
    ggx_g1(cos_o, alpha) * ggx_distribution(cos_h, alpha) / (4.0 * cos_o)
}

/// Fractional node position `(lo, hi, frac)` for a continuous axis coordinate.
fn axis_weights(x: f32) -> (usize, usize, f32) {
    let t = (x * 16.0 - 0.5).clamp(0.0, 15.0);
    let lo_f = t.floor();
    let lo = lo_f as usize;
    let hi = (lo + 1).min(15);
    (lo, hi, t - lo_f)
}

/// Single-scatter directional albedo `E(cos_theta, alpha)`, bilinear from the
/// baked table.
fn directional_albedo(cos_theta: f32, alpha: f32) -> f32 {
    let (a_lo, a_hi, a_frac) = axis_weights(alpha);
    let (c_lo, c_hi, c_frac) = axis_weights(cos_theta);
    let base_lo = a_lo * 16;
    let base_hi = a_hi * 16;
    let row_lo =
        ALBEDO[base_lo + c_lo] + (ALBEDO[base_lo + c_hi] - ALBEDO[base_lo + c_lo]) * c_frac;
    let row_hi =
        ALBEDO[base_hi + c_lo] + (ALBEDO[base_hi + c_hi] - ALBEDO[base_hi + c_lo]) * c_frac;
    (row_lo + (row_hi - row_lo) * a_frac).clamp(0.0, 1.0)
}

/// Cosine-weighted average albedo `E_avg(alpha)`, linear from the baked table.
fn average_albedo(alpha: f32) -> f32 {
    let (lo, hi, frac) = axis_weights(alpha);
    (AVG_ALBEDO[lo] + (AVG_ALBEDO[hi] - AVG_ALBEDO[lo]) * frac).clamp(0.0, 1.0)
}

/// Scalar Kulla-Conty multiple-scattering lobe.
fn multiscatter_lobe(cos_o: f32, cos_i: f32, alpha: f32) -> f32 {
    let e_avg = average_albedo(alpha);
    let denom = 1.0 - e_avg;
    if denom <= 1.0e-4 {
        return 0.0;
    }
    let e_o = directional_albedo(cos_o, alpha);
    let e_i = directional_albedo(cos_i, alpha);
    (1.0 - e_o) * (1.0 - e_i) / (std::f32::consts::PI * denom)
}

/// Per-channel multiple-scatter `Fresnel` tint `F_ms`.
fn ms_fresnel_channel(f: f32, e_avg: f32, one_minus: f32) -> f32 {
    let denom = 1.0 - f * one_minus;
    if denom <= 1.0e-4 {
        f
    } else {
        f * f * e_avg / denom
    }
}

/// Three-channel multiple-scatter `Fresnel` tint.
fn multiscatter_fresnel(f_avg: [f32; 3], alpha: f32) -> [f32; 3] {
    let e_avg = average_albedo(alpha);
    let one_minus = 1.0 - e_avg;
    [
        ms_fresnel_channel(f_avg[0], e_avg, one_minus),
        ms_fresnel_channel(f_avg[1], e_avg, one_minus),
        ms_fresnel_channel(f_avg[2], e_avg, one_minus),
    ]
}

/// The independent oracle for one query, mirroring the kernel branch for
/// branch: returns `(value, pdf, valid)`.
fn oracle(q: &ConductorMultiscatterBsdfQuery) -> ([f32; 3], f32, u32) {
    let zero = ([0.0, 0.0, 0.0], 0.0, 0u32);
    let normal = q.normal;
    let wo = q.wo;
    let wi = q.wi;

    let cos_o = dot3(normal, wo);
    let cos_i = dot3(normal, wi);
    if cos_o <= 0.0 || cos_i <= 0.0 {
        return zero;
    }

    let r = q.roughness.clamp(0.0, 1.0);
    let alpha = (r * r).max(MIN_ALPHA);

    // World-space half vector with a degenerate-length guard matching the
    // golden normalize_or_zero(len_sq > EPS_LEN_SQ) contract.
    let half_sum = [wo[0] + wi[0], wo[1] + wi[1], wo[2] + wi[2]];
    let hls = dot3(half_sum, half_sum);
    let half_valid = hls > EPS_LEN_SQ;
    let inv = if half_valid { 1.0 / hls.sqrt() } else { 0.0 };
    let half_vec = [half_sum[0] * inv, half_sum[1] * inv, half_sum[2] * inv];
    let cos_h = dot3(normal, half_vec);

    // Single-scatter exact conductor term; self-zeros on a degenerate or
    // back-facing half vector.
    let single = if half_valid && cos_h > 0.0 {
        let d = ggx_distribution(cos_h, alpha);
        let g2v = ggx_g2(cos_o, cos_i, alpha);
        let woh = dot3(wo, half_vec).max(0.0);
        let fres = fresnel(q.eta, q.k, woh);
        let scale = d * g2v / (4.0 * cos_o * cos_i);
        [fres[0] * scale, fres[1] * scale, fres[2] * scale]
    } else {
        [0.0, 0.0, 0.0]
    };

    // Kulla-Conty compensation lobe tinted by the average Fresnel.
    let f_avg = average_fresnel(q.eta, q.k);
    let lobe = multiscatter_lobe(cos_o, cos_i, alpha);
    let tint = multiscatter_fresnel(f_avg, alpha);
    let value = [
        single[0] + tint[0] * lobe,
        single[1] + tint[1] * lobe,
        single[2] + tint[2] * lobe,
    ];

    // One-sample MIS density mixing the GGX and cosine strategies.
    let p_ss = average_albedo(alpha).clamp(0.1, 0.9);
    let ggx = if half_valid {
        ggx_reflection_pdf(cos_o, cos_h, alpha)
    } else {
        0.0
    };
    let diffuse = dot3(normal, wi).max(0.0) * INV_PI;
    let pdf = p_ss * ggx + (1.0 - p_ss) * diffuse;
    (value, pdf, 1u32)
}

/// Dispatches one query and asserts both the value, density and validity flag.
fn assert_parity(
    ctx: &GpuContext,
    gpu: &GpuConductorMultiscatterBsdf,
    q: ConductorMultiscatterBsdfQuery,
) {
    let got = gpu.evaluate(ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    let (value, pdf, valid) = oracle(&q);
    let r = got[0];
    assert_eq!(r.valid, valid, "valid flag mismatch: query={q:?}");
    if valid == 1 {
        assert!(
            close3(r.value, value),
            "value mismatch: gpu={:?} cpu={value:?} query={q:?}",
            r.value
        );
        assert!(
            close(r.pdf, pdf),
            "pdf mismatch: gpu={} cpu={pdf} query={q:?}",
            r.pdf
        );
    }
}

/// Builds a unit direction at a given cosine about `normal`, with the azimuth
/// chosen by the two tangent-plane coefficients.
fn direction_about(normal: [f32; 3], cos: f32, azimuth_a: f32, azimuth_b: f32) -> [f32; 3] {
    let reference = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let tangent = normalize3(cross3(normal, reference));
    let bitangent = cross3(normal, tangent);
    let plane = normalize3([
        azimuth_a * tangent[0] + azimuth_b * bitangent[0],
        azimuth_a * tangent[1] + azimuth_b * bitangent[1],
        azimuth_a * tangent[2] + azimuth_b * bitangent[2],
    ]);
    let s = (1.0 - cos * cos).max(0.0).sqrt();
    normalize3([
        cos * normal[0] + s * plane[0],
        cos * normal[1] + s * plane[1],
        cos * normal[2] + s * plane[2],
    ])
}

#[test]
fn normal_incidence() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    // Both directions straight up the normal, moderate roughness.
    assert_parity(
        &ctx,
        &gpu,
        ConductorMultiscatterBsdfQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.3,
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ),
    );
}

#[test]
fn oblique_reflection() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    let normal = [0.0, 0.0, 1.0];
    let wo = normalize3([0.3, 0.1, 1.0]);
    let wi = normalize3([-0.2, 0.15, 1.0]);
    // Oblique configuration exercising the full GGX + compensation body.
    assert_parity(
        &ctx,
        &gpu,
        ConductorMultiscatterBsdfQuery::new(GOLD_ETA, GOLD_K, 0.4, wo, wi, normal),
    );
}

#[test]
fn very_rough_lobe() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    // Near-unit roughness maximises the Kulla-Conty compensation contribution.
    let normal = normalize3([0.1, 0.2, 1.0]);
    let wo = direction_about(normal, 0.6, 0.7, 0.3);
    let wi = direction_about(normal, 0.55, -0.4, 0.8);
    assert_parity(
        &ctx,
        &gpu,
        ConductorMultiscatterBsdfQuery::new(GOLD_ETA, GOLD_K, 0.95, wo, wi, normal),
    );
}

#[test]
fn nearly_smooth_lobe() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    // Very low roughness pushes alpha toward MIN_ALPHA, a sharp single-scatter
    // lobe where the compensation term nearly vanishes.
    let normal = [0.0, 0.0, 1.0];
    let wo = normalize3([0.15, 0.05, 1.0]);
    let wi = normalize3([-0.12, 0.08, 1.0]);
    assert_parity(
        &ctx,
        &gpu,
        ConductorMultiscatterBsdfQuery::new(GOLD_ETA, GOLD_K, 0.02, wo, wi, normal),
    );
}

#[test]
fn copper_metal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    // Measured copper complex index, oblique rough configuration.
    let eta = [0.2004, 0.9240, 1.1022];
    let k = [3.9129, 2.4528, 2.1421];
    let normal = normalize3([0.05, 0.1, 1.0]);
    let wo = direction_about(normal, 0.8, 0.6, 0.4);
    let wi = direction_about(normal, 0.7, -0.5, 0.5);
    assert_parity(
        &ctx,
        &gpu,
        ConductorMultiscatterBsdfQuery::new(eta, k, 0.45, wo, wi, normal),
    );
}

#[test]
fn aluminium_metal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    // Measured aluminium complex index, moderate roughness.
    let eta = [1.3456, 0.9654, 0.6170];
    let k = [7.4746, 6.3995, 5.3031];
    let normal = normalize3([0.2, -0.1, 1.0]);
    let wo = direction_about(normal, 0.75, -0.3, 0.7);
    let wi = direction_about(normal, 0.65, 0.8, -0.2);
    assert_parity(
        &ctx,
        &gpu,
        ConductorMultiscatterBsdfQuery::new(eta, k, 0.6, wo, wi, normal),
    );
}

#[test]
fn below_horizon_outgoing_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    // Outgoing direction below the surface: valid = 0, outputs cleared.
    assert_parity(
        &ctx,
        &gpu,
        ConductorMultiscatterBsdfQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.3,
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ),
    );
}

#[test]
fn below_horizon_incoming_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    // Incoming direction below the surface: valid = 0, outputs cleared.
    assert_parity(
        &ctx,
        &gpu,
        ConductorMultiscatterBsdfQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.3,
            [0.2, 0.1, 0.9],
            [0.1, 0.1, -0.9],
            [0.0, 0.0, 1.0],
        ),
    );
}

#[test]
fn multi_element_batch_regression() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    // A >=3-element batch mixing valid and invalid queries in a single
    // dispatch, so any std430 stride/offset bug in the packed Result (which a
    // sibling module hit on its valid flag) would surface here.
    let normal = [0.0, 0.0, 1.0];
    let queries = vec![
        // valid oblique
        ConductorMultiscatterBsdfQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.4,
            normalize3([0.3, 0.1, 1.0]),
            normalize3([-0.2, 0.15, 1.0]),
            normal,
        ),
        // invalid: outgoing below horizon
        ConductorMultiscatterBsdfQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.5,
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            normal,
        ),
        // valid normal incidence
        ConductorMultiscatterBsdfQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.25,
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            normal,
        ),
        // invalid: incoming below horizon
        ConductorMultiscatterBsdfQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.6,
            [0.1, 0.2, 0.95],
            [0.0, 0.0, -1.0],
            normal,
        ),
        // valid very rough
        ConductorMultiscatterBsdfQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.9,
            normalize3([0.2, -0.1, 1.0]),
            normalize3([-0.15, 0.2, 1.0]),
            normal,
        ),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (value, pdf, valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        if valid == 1 {
            assert!(
                close3(r.value, value),
                "batch value mismatch: gpu={:?} cpu={value:?} query={q:?}",
                r.value
            );
            assert!(
                close(r.pdf, pdf),
                "batch pdf mismatch: gpu={} cpu={pdf} query={q:?}",
                r.pdf
            );
        }
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
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
    let gpu = GpuConductorMultiscatterBsdf::new(&ctx);
    let mut rng = Lcg::new(0x4B_1D_9E_27);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // A shading normal with a comfortably positive z.
        let normal = normalize3([
            rng.next_range(-0.5, 0.5),
            rng.next_range(-0.5, 0.5),
            rng.next_range(0.4, 1.0),
        ]);
        // Both directions are built from an explicit cosine about the normal,
        // so both cosines stay comfortably positive and parity never sits on
        // the below-horizon knife edge.
        let wo = direction_about(
            normal,
            rng.next_range(0.25, 0.97),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        );
        let wi = direction_about(
            normal,
            rng.next_range(0.25, 0.97),
            rng.next_range(-1.0, 1.0),
            rng.next_range(-1.0, 1.0),
        );
        // Keep roughness away from the mirror-sharp regime so the GGX
        // denominator stays well conditioned on both sides.
        let roughness = rng.next_range(0.1, 1.0);
        let eta = [
            rng.next_range(0.1, 2.5),
            rng.next_range(0.1, 2.5),
            rng.next_range(0.1, 2.5),
        ];
        let k = [
            rng.next_range(1.0, 4.0),
            rng.next_range(1.0, 4.0),
            rng.next_range(1.0, 4.0),
        ];
        queries.push(ConductorMultiscatterBsdfQuery::new(
            eta, k, roughness, wo, wi, normal,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let (value, pdf, valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        if valid == 1 {
            assert!(
                close3(r.value, value),
                "sweep value mismatch: gpu={:?} cpu={value:?} query={q:?}",
                r.value
            );
            assert!(
                close(r.pdf, pdf),
                "sweep pdf mismatch: gpu={} cpu={pdf} query={q:?}",
                r.pdf
            );
        }
    }
}
