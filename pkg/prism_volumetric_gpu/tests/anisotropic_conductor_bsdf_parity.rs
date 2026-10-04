//! Real-device parity for the anisotropic rough-conductor twin:
//! [`GpuAnisoConductor`](prism_volumetric_gpu::anisotropic_conductor_bsdf::GpuAnisoConductor)
//! must reproduce the `CPU` golden `AnisoConductor::evaluate` and
//! `AnisoConductor::pdf` of
//! `prism_render_architecture::reference_pt::conductor_aniso`, which pair the
//! exact complex-index `Fresnel` reflectance with an anisotropic `GGX`
//! microfacet lobe to shade brushed metal.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! the Disney / `UE` roughness-anisotropy remap, the elliptical `GGX`
//! `distribution`, the Smith `lambda` feeding `g1`/`g2`, the Duff tangent
//! basis, the local-frame projection and the per-channel conductor `Fresnel` —
//! written out directly so the test never imports `prism_render_architecture`.
//! It mirrors the reference branch for branch, including the below-horizon and
//! degenerate half-vector guards that return a zero value and density.
//!
//! The fixtures cover the lobes the kernel must honor: an isotropic lobe at
//! normal incidence and at an oblique angle, a strongly anisotropic lobe under
//! a tilted shading normal, and two below-horizon pairs that must report
//! `valid = 0`. A sweep over random metals, roughness/anisotropy tunings and
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
//! positive and the normal `z` well away from the Duff basis pole at `-1`, so
//! parity never sits on a branch knife edge.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::reference_pt::conductor_aniso`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::anisotropic_conductor_bsdf::{AnisoConductorQuery, GpuAnisoConductor};
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

/// The reference metal used across the named fixtures: a measured complex index
/// for a warm conductor.
const GOLD_ETA: [f32; 3] = [0.143, 0.375, 1.442];
/// Extinction coefficient paired with [`GOLD_ETA`].
const GOLD_K: [f32; 3] = [3.983, 2.386, 1.603];

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

/// Independent oracle for the Disney / `UE` roughness-anisotropy remap into the
/// two `GGX` widths `(ax, ay)`.
fn alpha_from_roughness_anisotropy(roughness: f32, anisotropy: f32) -> (f32, f32) {
    let r = roughness.clamp(0.0, 1.0);
    let alpha = r * r;
    let aniso = anisotropy.clamp(0.0, 1.0);
    let aspect = (1.0 - 0.9 * aniso).max(1.0e-4).sqrt();
    let ax = (alpha / aspect).max(MIN_ALPHA);
    let ay = (alpha * aspect).max(MIN_ALPHA);
    (ax, ay)
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

/// Smith `G1` masking term.
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

/// The independent oracle for one query, mirroring the kernel branch for
/// branch: returns `(value, pdf, valid)`.
fn oracle(q: &AnisoConductorQuery) -> ([f32; 3], f32, u32) {
    let zero = ([0.0, 0.0, 0.0], 0.0, 0u32);
    let (ax, ay) = alpha_from_roughness_anisotropy(q.roughness, q.anisotropy);
    let normal = q.normal;
    let wo = q.wo;
    let wi = q.wi;

    let cos_o = dot3(normal, wo);
    let cos_i = dot3(normal, wi);
    if cos_o <= 0.0 || cos_i <= 0.0 {
        return zero;
    }

    // Branchless Duff tangent basis with an ordered-compare sign, matching the
    // kernel exactly.
    let sign_z = if normal[2] >= 0.0 { 1.0 } else { -1.0 };
    let a_duff = -1.0 / (sign_z + normal[2]);
    let b_duff = normal[0] * normal[1] * a_duff;
    let tangent = [
        1.0 + sign_z * normal[0] * normal[0] * a_duff,
        sign_z * b_duff,
        -sign_z * normal[0],
    ];
    let bitangent = [b_duff, sign_z + normal[1] * normal[1] * a_duff, -normal[1]];

    let wo_local = [dot3(wo, tangent), dot3(wo, bitangent), dot3(wo, normal)];
    let wi_local = [dot3(wi, tangent), dot3(wi, bitangent), dot3(wi, normal)];

    let half_sum = [
        wo_local[0] + wi_local[0],
        wo_local[1] + wi_local[1],
        wo_local[2] + wi_local[2],
    ];
    let half_len_sq = dot3(half_sum, half_sum);
    let half_valid = half_len_sq > EPS_LEN_SQ;
    let inv_len = if half_valid {
        1.0 / half_len_sq.sqrt()
    } else {
        0.0
    };
    let half_local = [
        half_sum[0] * inv_len,
        half_sum[1] * inv_len,
        half_sum[2] * inv_len,
    ];
    if !half_valid || half_local[2] <= 0.0 {
        return zero;
    }

    let d = distribution(half_local, ax, ay);
    let g2v = g2(wo_local, wi_local, ax, ay);
    let woh = dot3(wo_local, half_local).max(0.0);
    let fres = fresnel(q.eta, q.k, woh);
    let scale = d * g2v / (4.0 * cos_o * cos_i);
    let value = [fres[0] * scale, fres[1] * scale, fres[2] * scale];
    let pdf = reflection_pdf(wo_local, half_local, ax, ay);
    (value, pdf, 1u32)
}

/// Dispatches one query and asserts both channels and the validity flag.
fn assert_parity(ctx: &GpuContext, gpu: &GpuAnisoConductor, q: AnisoConductorQuery) {
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
fn isotropic_normal_incidence() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisoConductor::new(&ctx);
    // Isotropic lobe, both directions straight up the normal.
    assert_parity(
        &ctx,
        &gpu,
        AnisoConductorQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.3,
            0.0,
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ),
    );
}

#[test]
fn isotropic_oblique() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisoConductor::new(&ctx);
    let normal = [0.0, 0.0, 1.0];
    let wo = normalize3([0.3, 0.1, 1.0]);
    let wi = normalize3([-0.2, 0.15, 1.0]);
    // Isotropic lobe at an oblique configuration, exercising the GGX body.
    assert_parity(
        &ctx,
        &gpu,
        AnisoConductorQuery::new(GOLD_ETA, GOLD_K, 0.4, 0.0, wo, wi, normal),
    );
}

#[test]
fn anisotropic_tilted_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisoConductor::new(&ctx);
    // A strongly anisotropic lobe under a shading normal tilted off +z, well
    // away from the Duff basis pole at -z.
    let normal = normalize3([0.2, 0.3, 1.0]);
    let wo = direction_about(normal, 0.7, 0.8, 0.2);
    let wi = direction_about(normal, 0.6, -0.3, 0.9);
    assert_parity(
        &ctx,
        &gpu,
        AnisoConductorQuery::new(GOLD_ETA, GOLD_K, 0.5, 0.9, wo, wi, normal),
    );
}

#[test]
fn below_horizon_outgoing_is_invalid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisoConductor::new(&ctx);
    // Outgoing direction below the surface: valid = 0, outputs cleared.
    assert_parity(
        &ctx,
        &gpu,
        AnisoConductorQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.3,
            0.2,
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
    let gpu = GpuAnisoConductor::new(&ctx);
    // Incoming direction below the surface: valid = 0, outputs cleared.
    assert_parity(
        &ctx,
        &gpu,
        AnisoConductorQuery::new(
            GOLD_ETA,
            GOLD_K,
            0.3,
            0.2,
            [0.2, 0.1, 0.9],
            [0.1, 0.1, -0.9],
            [0.0, 0.0, 1.0],
        ),
    );
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuAnisoConductor::new(&ctx);
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
    let gpu = GpuAnisoConductor::new(&ctx);
    let mut rng = Lcg::new(0x4B_1D_9E_27);
    let mut queries = Vec::with_capacity(512);
    for _ in 0..512 {
        // A shading normal with a comfortably positive z keeps the Duff basis
        // away from its -z pole.
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
        let anisotropy = rng.next_range(0.0, 0.95);
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
        queries.push(AnisoConductorQuery::new(
            eta, k, roughness, anisotropy, wo, wi, normal,
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
