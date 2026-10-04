//! Real-device parity for the tetrahedron-quality twin:
//! [`GpuTetQualityMetrics`](prism_volumetric_gpu::tet_quality_metrics::GpuTetQualityMetrics)
//! must reproduce the `CPU` golden `tet_quality` of
//! `prism_physics_core::collider::tet_quality` for a single tetrahedron.
//!
//! The oracle here is an independent re-implementation of that closed form,
//! written out directly so the test never imports `prism_render_architecture`,
//! `prism_physics_core` or `glam`. It threads through the same operator order as
//! the device kernel: the signed volume, the six edge lengths, the four outward
//! face normals and six dihedral angles, and the circumradius from the `3x3`
//! edge system solved with the same cofactor inverse. The dihedral angle is
//! routed through `f64` `acos`, exactly matching the golden path, while the
//! circumradius stays in `f32` as the golden `glam` solve does. Non-finite
//! inputs short-circuit on both sides to `inverted = degenerate = true` with the
//! continuous fields zeroed, so the two sides agree deterministically rather
//! than racing propagated `NaN`.
//!
//! # Parity criterion
//!
//! The continuous arithmetic threads through operators a `GPU` may contract, so
//! `CPU` and `GPU` evaluate the same closed form but need not be bit-exact. The
//! continuous scalars are compared with `abs <= 1e-4 || rel <= 1e-3`
//! (`REL_FLOOR = 1e-6`); the discrete `inverted` and `degenerate` flags are
//! compared exactly. The well-conditioned fixtures keep every dihedral angle in
//! `[25, 150]` degrees (where `acos` is well conditioned) and the radius ratio
//! clear of the `|det|` degeneracy edge, so neither flag can flip under
//! round-off. Degenerate and non-finite fixtures assert the flags and the zeroed
//! `radius_ratio` only.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::collider::tet_quality`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::tet_quality_metrics::{
    GpuTetQualityMetrics, TetQualityMetricsQuery, TetQualityMetricsResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Finite-magnitude limit mirroring the kernel's `FINITE_LIMIT`.
const FINITE_LIMIT: f32 = 3.0e38;

/// The `|det|` degeneracy edge mirroring the kernel's `DEGENERATE_DET`.
const DEGENERATE_DET: f32 = 1.0e-12;

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// Outward unit normal of face `(a, b, c)` pointing away from `apex`, matching
/// the kernel's `outward_face_normal` operator for operator.
fn outward_face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3], apex: [f32; 3]) -> [f32; 3] {
    let nvec = cross(sub(b, a), sub(c, a));
    let len = length(nvec);
    let ok = len > 1.0e-20;
    let safe_len = if ok { len } else { 1.0 };
    let unit = [nvec[0] / safe_len, nvec[1] / safe_len, nvec[2] / safe_len];
    let outward = dot(unit, sub(apex, a)) > 0.0;
    let oriented = if outward {
        [-unit[0], -unit[1], -unit[2]]
    } else {
        unit
    };
    if ok {
        oriented
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Dihedral angle in degrees between two outward face normals, re-implementing
/// the golden `f64` `acos` path so the oracle matches the reference exactly.
fn dihedral_deg(n0: [f32; 3], n1: [f32; 3]) -> f32 {
    let cosv = dot(n0, n1).clamp(-1.0, 1.0);
    let angle = std::f64::consts::PI - f64::from(cosv).acos();
    (angle * 180.0 / std::f64::consts::PI) as f32
}

/// Independent host oracle for `tet_quality`, mirroring the device kernel's
/// operator order and its non-finite short-circuit.
fn oracle(q: &TetQualityMetricsQuery) -> TetQualityMetricsResult {
    let v0 = q.v0;
    let v1 = q.v1;
    let v2 = q.v2;
    let v3 = q.v3;

    let finite_in = [v0, v1, v2, v3]
        .iter()
        .all(|v| v.iter().all(|&c| c.abs() < FINITE_LIMIT));
    if !finite_in {
        return TetQualityMetricsResult {
            volume: 0.0,
            radius_ratio: 0.0,
            min_dihedral_deg: 0.0,
            max_dihedral_deg: 0.0,
            min_edge: 0.0,
            max_edge: 0.0,
            inverted: true,
            degenerate: true,
        };
    }

    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let e3 = sub(v3, v0);

    let vol6 = dot(e1, cross(e2, e3));
    let volume = vol6 / 6.0;
    let inverted = !(volume > 0.0);

    let d10 = length(e1);
    let d20 = length(e2);
    let d30 = length(e3);
    let d21 = length(sub(v2, v1));
    let d31 = length(sub(v3, v1));
    let d32 = length(sub(v3, v2));
    let min_edge = d10.min(d20).min(d30).min(d21).min(d31).min(d32);
    let max_edge = d10.max(d20).max(d30).max(d21).max(d31).max(d32);

    let a0 = 0.5 * length(cross(sub(v2, v1), sub(v3, v1)));
    let a1 = 0.5 * length(cross(e2, e3));
    let a2 = 0.5 * length(cross(e1, e3));
    let a3 = 0.5 * length(cross(e1, e2));
    let area_total = a0 + a1 + a2 + a3;

    let n0 = outward_face_normal(v1, v2, v3, v0);
    let n1 = outward_face_normal(v0, v2, v3, v1);
    let n2 = outward_face_normal(v0, v1, v3, v2);
    let n3 = outward_face_normal(v0, v1, v2, v3);
    let da = dihedral_deg(n0, n1);
    let db = dihedral_deg(n0, n2);
    let dc = dihedral_deg(n0, n3);
    let dd = dihedral_deg(n1, n2);
    let de = dihedral_deg(n1, n3);
    let df = dihedral_deg(n2, n3);
    let min_dih = da.min(db).min(dc).min(dd).min(de).min(df);
    let max_dih = da.max(db).max(dc).max(dd).max(de).max(df);

    // Circumradius via the 3x3 edge system solved with the cofactor inverse,
    // matching the kernel's column assembly operator for operator.
    let ax = [e1[0], e2[0], e3[0]];
    let ay = [e1[1], e2[1], e3[1]];
    let az = [e1[2], e2[2], e3[2]];
    let tmp0 = cross(ay, az);
    let tmp1 = cross(az, ax);
    let tmp2 = cross(ax, ay);
    let det = dot(az, tmp2);
    let det_ok = det.abs() >= DEGENERATE_DET;
    let inv_det = 1.0 / if det_ok { det } else { 1.0 };
    let inv_x = [tmp0[0] * inv_det, tmp1[0] * inv_det, tmp2[0] * inv_det];
    let inv_y = [tmp0[1] * inv_det, tmp1[1] * inv_det, tmp2[1] * inv_det];
    let inv_z = [tmp0[2] * inv_det, tmp1[2] * inv_det, tmp2[2] * inv_det];
    let bb = [0.5 * dot(e1, e1), 0.5 * dot(e2, e2), 0.5 * dot(e3, e3)];
    let centre = [
        inv_x[0] * bb[0] + inv_y[0] * bb[1] + inv_z[0] * bb[2],
        inv_x[1] * bb[0] + inv_y[1] * bb[1] + inv_z[1] * bb[2],
        inv_x[2] * bb[0] + inv_y[2] * bb[1] + inv_z[2] * bb[2],
    ];
    let r_circ = length(centre);

    let nondegen = det_ok && r_circ > 0.0 && area_total > 0.0;
    let r_in = 3.0 * volume.abs() / if area_total > 0.0 { area_total } else { 1.0 };
    let ratio_raw = 3.0 * r_in / if r_circ > 0.0 { r_circ } else { 1.0 };
    let ratio = ratio_raw.clamp(0.0, 1.0);
    let radius_ratio = if nondegen { ratio } else { 0.0 };

    TetQualityMetricsResult {
        volume,
        radius_ratio,
        min_dihedral_deg: min_dih,
        max_dihedral_deg: max_dih,
        min_edge,
        max_edge,
        inverted,
        degenerate: !nondegen,
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

/// Asserts the discrete flags match exactly.
fn assert_flags(gpu: &TetQualityMetricsResult, want: &TetQualityMetricsResult, label: &str) {
    assert_eq!(
        gpu.inverted, want.inverted,
        "{label}: inverted flag mismatch"
    );
    assert_eq!(
        gpu.degenerate, want.degenerate,
        "{label}: degenerate flag mismatch"
    );
}

/// Asserts a well-conditioned result matches the oracle in every field.
fn assert_full(gpu: &TetQualityMetricsResult, q: &TetQualityMetricsQuery, label: &str) {
    let want = oracle(q);
    assert_flags(gpu, &want, label);
    assert!(
        close(gpu.volume, want.volume),
        "{label}: volume gpu={} oracle={}",
        gpu.volume,
        want.volume
    );
    assert!(
        close(gpu.radius_ratio, want.radius_ratio),
        "{label}: radius_ratio gpu={} oracle={}",
        gpu.radius_ratio,
        want.radius_ratio
    );
    assert!(
        close(gpu.min_dihedral_deg, want.min_dihedral_deg),
        "{label}: min_dihedral gpu={} oracle={}",
        gpu.min_dihedral_deg,
        want.min_dihedral_deg
    );
    assert!(
        close(gpu.max_dihedral_deg, want.max_dihedral_deg),
        "{label}: max_dihedral gpu={} oracle={}",
        gpu.max_dihedral_deg,
        want.max_dihedral_deg
    );
    assert!(
        close(gpu.min_edge, want.min_edge),
        "{label}: min_edge gpu={} oracle={}",
        gpu.min_edge,
        want.min_edge
    );
    assert!(
        close(gpu.max_edge, want.max_edge),
        "{label}: max_edge gpu={} oracle={}",
        gpu.max_edge,
        want.max_edge
    );
}

/// True when a tetrahedron is well away from every degeneracy knee: positive
/// volume, not degenerate, radius ratio in `[0.2, 0.95]`, every dihedral angle
/// in `[25, 150]` degrees and a non-trivial volume magnitude.
fn well_conditioned(res: &TetQualityMetricsResult) -> bool {
    !res.inverted
        && !res.degenerate
        && res.radius_ratio >= 0.2
        && res.radius_ratio <= 0.95
        && res.min_dihedral_deg >= 25.0
        && res.max_dihedral_deg <= 150.0
        && res.volume.abs() >= 0.05
}

#[test]
fn regular_tetrahedron_has_unit_radius_ratio() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetQualityMetrics::new(&ctx);
    let q = TetQualityMetricsQuery::new(
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert!(!out[0].degenerate, "regular tet is not degenerate");
    assert!(
        close(out[0].radius_ratio, 1.0),
        "ratio {}",
        out[0].radius_ratio
    );
    // All dihedral angles equal arccos(1/3) ~ 70.5288 degrees.
    assert!(close(out[0].min_dihedral_deg, 70.5288));
    assert!(close(out[0].max_dihedral_deg, 70.5288));
    // Full parity vs the independent oracle (flags + all continuous fields).
    let want = oracle(&q);
    assert_flags(&out[0], &want, "regular");
    assert!(close(out[0].volume, want.volume));
    assert!(close(out[0].radius_ratio, want.radius_ratio));
    assert!(close(out[0].min_edge, want.min_edge));
    assert!(close(out[0].max_edge, want.max_edge));
}

#[test]
fn inverted_tetrahedron_is_detected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetQualityMetrics::new(&ctx);
    // Positive-orientation tetrahedron.
    let good = TetQualityMetricsQuery::new(
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
    );
    // Swapping two vertices flips the orientation.
    let bad = TetQualityMetricsQuery::new(
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    );
    let out = gpu.evaluate(&ctx, &[good, bad]);
    assert_eq!(out.len(), 2);
    let good_ref = oracle(&good);
    let bad_ref = oracle(&bad);
    assert!(!good_ref.inverted, "good orientation is not inverted");
    assert!(bad_ref.inverted, "swapped orientation is inverted");
    assert_flags(&out[0], &good_ref, "good");
    assert_flags(&out[1], &bad_ref, "bad");
    assert!(close(out[0].volume, good_ref.volume));
    assert!(close(out[1].volume, bad_ref.volume));
    assert!(out[0].volume > 0.0);
    assert!(out[1].volume < 0.0);
}

#[test]
fn flat_tetrahedron_is_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetQualityMetrics::new(&ctx);
    // Four coplanar points (z = 0).
    let q = TetQualityMetricsQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    );
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    let want = oracle(&q);
    assert!(
        want.degenerate && want.inverted,
        "coplanar tet is degenerate+inverted"
    );
    assert_flags(&out[0], &want, "flat");
    assert_eq!(out[0].radius_ratio, 0.0, "degenerate radius ratio is zero");
}

#[test]
fn non_finite_input_is_rejected() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetQualityMetrics::new(&ctx);
    let nan_q = TetQualityMetricsQuery::new(
        [f32::NAN, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    let inf_q = TetQualityMetricsQuery::new(
        [0.0, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    let out = gpu.evaluate(&ctx, &[nan_q, inf_q]);
    assert_eq!(out.len(), 2);
    for (res, label) in out.iter().zip(["nan", "inf"]) {
        assert!(res.inverted, "{label}: non-finite is inverted");
        assert!(res.degenerate, "{label}: non-finite is degenerate");
        assert_eq!(res.volume, 0.0, "{label}: volume zeroed");
        assert_eq!(res.radius_ratio, 0.0, "{label}: radius_ratio zeroed");
        assert_eq!(res.min_dihedral_deg, 0.0, "{label}: min_dih zeroed");
        assert_eq!(res.max_dihedral_deg, 0.0, "{label}: max_dih zeroed");
        assert_eq!(res.min_edge, 0.0, "{label}: min_edge zeroed");
        assert_eq!(res.max_edge, 0.0, "{label}: max_edge zeroed");
    }
}

#[test]
fn well_conditioned_fixtures_match_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetQualityMetrics::new(&ctx);
    // Collect several well-conditioned tetrahedra via rejection sampling.
    let mut lcg = Lcg::new(0x7E7_90018);
    let mut queries = Vec::with_capacity(8);
    let mut guard = 0u32;
    while queries.len() < 8 && guard < 100_000 {
        guard += 1;
        let q = random_tet(&mut lcg);
        if well_conditioned(&oracle(&q)) {
            queries.push(q);
        }
    }
    assert_eq!(
        queries.len(),
        8,
        "could not sample enough well-conditioned tets"
    );
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_full(res, q, &format!("fixture[{i}]"));
    }
}

#[test]
fn batch_mixes_cases_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetQualityMetrics::new(&ctx);
    // Sample one well-conditioned element to lead the mixed batch.
    let mut lcg = Lcg::new(0x1234_5678);
    let mut good = random_tet(&mut lcg);
    let mut guard = 0u32;
    while !well_conditioned(&oracle(&good)) && guard < 100_000 {
        guard += 1;
        good = random_tet(&mut lcg);
    }
    assert!(well_conditioned(&oracle(&good)), "failed to seed good tet");

    let flat = TetQualityMetricsQuery::new(
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    );
    let nan_q = TetQualityMetricsQuery::new(
        [0.0, 0.0, f32::NAN],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    let unit = TetQualityMetricsQuery::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    );
    let queries = vec![good, flat, nan_q, unit];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());

    assert_full(&out[0], &queries[0], "batch_good");

    let flat_ref = oracle(&flat);
    assert_flags(&out[1], &flat_ref, "batch_flat");
    assert_eq!(out[1].radius_ratio, 0.0);

    let nan_ref = oracle(&nan_q);
    assert_flags(&out[2], &nan_ref, "batch_nan");
    assert_eq!(out[2].volume, 0.0);
    assert_eq!(out[2].radius_ratio, 0.0);

    // The unit corner tetrahedron is a positive, non-degenerate element; verify
    // it against the oracle in full to confirm the trailing stride slot.
    let unit_ref = oracle(&unit);
    assert_flags(&out[3], &unit_ref, "batch_unit");
    assert!(close(out[3].volume, unit_ref.volume));
    assert!(close(out[3].radius_ratio, unit_ref.radius_ratio));
    assert!(close(out[3].min_edge, unit_ref.min_edge));
    assert!(close(out[3].max_edge, unit_ref.max_edge));
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetQualityMetrics::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuTetQualityMetrics::new(&ctx);
    let mut lcg = Lcg::new(0x0CA9_11A5);
    let mut queries = Vec::with_capacity(512);
    let mut guard = 0u32;
    // Rejection-sample well-conditioned tetrahedra away from every knee.
    while queries.len() < 512 && guard < 5_000_000 {
        guard += 1;
        let q = random_tet(&mut lcg);
        if well_conditioned(&oracle(&q)) {
            queries.push(q);
        }
    }
    assert_eq!(
        queries.len(),
        512,
        "could not sample enough well-conditioned tets"
    );
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_full(res, q, &format!("sweep[{i}]"));
    }
}

/// Builds a random tetrahedron with vertices in the box `[-2, 2]^3`.
fn random_tet(lcg: &mut Lcg) -> TetQualityMetricsQuery {
    let mut vertex = || {
        [
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
            lcg.next_range(-2.0, 2.0),
        ]
    };
    TetQualityMetricsQuery::new(vertex(), vertex(), vertex(), vertex())
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
