//! Real-device parity for the camera-projection matrix twin:
//! [`GpuProjection`](prism_volumetric_gpu::projection::GpuProjection) must
//! reproduce the `CPU` golden perspective/orthographic constructors of
//! `prism_math::projection`. Each query selects one of ten pure-scalar
//! variants by an enum id and returns a column-major `[f32; 16]` matching
//! `Mat4::from_cols` (column 0 is `m[0..4]`, column 1 is `m[4..8]`, and so on).
//!
//! The oracle here is an independent re-implementation of those closed forms —
//! the focal term `f = 1 / tan(fovy * 0.5)`, the near/far depth remaps, and the
//! orthographic extent reciprocals — written out directly in `f32` so the test
//! never imports `prism_math`, `prism_render_architecture`, `prism_physics_core`,
//! or `glam`.
//!
//! The fixtures cover every variant `0..=9` with representative parameters, a
//! mixed batch of two or more elements (including an out-of-range `variant_id`
//! that must zero the matrix and clear `valid`) that validates the `std430`
//! stride, and an empty batch the host short-circuits with no dispatch. A sweep
//! over random master-valid parameters follows, cycling through all ten
//! variants.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The device evaluates the focal term with the native `f32` `tan`, while the
//! oracle uses the same `f32` `tan`; the two agree to well within tolerance.
//! Each of the sixteen continuous matrix entries is compared with
//! `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid` flag
//! is compared exactly. A recognised `variant_id` (`<= 9`) yields `valid = 1`;
//! anything else yields `valid = 0` with an all-zero matrix.
//!
//! Provenance: 孪生自本仓 `prism_math::projection`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::projection::{GpuProjection, ProjectionQuery, ProjectionResult};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Independent host oracle: reproduces the selected `prism_math::projection`
/// constructor in the golden operator order and `f32` precision, returning the
/// column-major matrix and the discrete validity flag. An unrecognised
/// `variant_id` returns an all-zero matrix with `valid = 0`.
fn oracle(q: &ProjectionQuery) -> ([f32; 16], u32) {
    let mut m = [0.0f32; 16];
    let a0 = q.a[0];
    let a1 = q.a[1];
    let a2 = q.a[2];
    let a3 = q.a[3];
    let a4 = q.a[4];
    let a5 = q.a[5];
    match q.variant_id {
        0 => {
            // perspective_rh
            let f = 1.0 / (a0 * 0.5).tan();
            let r = a3 / (a2 - a3);
            m[0] = f / a1;
            m[5] = f;
            m[10] = r;
            m[11] = -1.0;
            m[14] = r * a2;
        }
        1 => {
            // perspective_rh_gl
            let f = 1.0 / (a0 * 0.5).tan();
            let inv = 1.0 / (a2 - a3);
            m[0] = f / a1;
            m[5] = f;
            m[10] = (a3 + a2) * inv;
            m[11] = -1.0;
            m[14] = 2.0 * a3 * a2 * inv;
        }
        2 => {
            // perspective_reverse_z_rh
            let f = 1.0 / (a0 * 0.5).tan();
            let inv = 1.0 / (a3 - a2);
            m[0] = f / a1;
            m[5] = f;
            m[10] = a2 * inv;
            m[11] = -1.0;
            m[14] = a3 * a2 * inv;
        }
        3 => {
            // perspective_infinite_rh
            let f = 1.0 / (a0 * 0.5).tan();
            m[0] = f / a1;
            m[5] = f;
            m[10] = -1.0;
            m[11] = -1.0;
            m[14] = -a2;
        }
        4 => {
            // perspective_infinite_reverse_z_rh
            let f = 1.0 / (a0 * 0.5).tan();
            m[0] = f / a1;
            m[5] = f;
            m[10] = 0.0;
            m[11] = -1.0;
            m[14] = a2;
        }
        5 => {
            // perspective_lh
            let f = 1.0 / (a0 * 0.5).tan();
            let r = a3 / (a3 - a2);
            m[0] = f / a1;
            m[5] = f;
            m[10] = r;
            m[11] = 1.0;
            m[14] = -r * a2;
        }
        6 => {
            // perspective_lh_gl
            let f = 1.0 / (a0 * 0.5).tan();
            let inv = 1.0 / (a3 - a2);
            m[0] = f / a1;
            m[5] = f;
            m[10] = (a3 + a2) * inv;
            m[11] = 1.0;
            m[14] = -2.0 * a3 * a2 * inv;
        }
        7 => {
            // orthographic_rh
            let rcp_w = 1.0 / (a1 - a0);
            let rcp_h = 1.0 / (a3 - a2);
            let rcp_d = 1.0 / (a4 - a5);
            m[0] = 2.0 * rcp_w;
            m[5] = 2.0 * rcp_h;
            m[10] = rcp_d;
            m[12] = -(a1 + a0) * rcp_w;
            m[13] = -(a3 + a2) * rcp_h;
            m[14] = a4 * rcp_d;
            m[15] = 1.0;
        }
        8 => {
            // orthographic_rh_gl
            let rcp_w = 1.0 / (a1 - a0);
            let rcp_h = 1.0 / (a3 - a2);
            let rcp_d = 1.0 / (a5 - a4);
            m[0] = 2.0 * rcp_w;
            m[5] = 2.0 * rcp_h;
            m[10] = -2.0 * rcp_d;
            m[12] = -(a1 + a0) * rcp_w;
            m[13] = -(a3 + a2) * rcp_h;
            m[14] = -(a5 + a4) * rcp_d;
            m[15] = 1.0;
        }
        9 => {
            // orthographic_lh
            let rcp_w = 1.0 / (a1 - a0);
            let rcp_h = 1.0 / (a3 - a2);
            let rcp_d = 1.0 / (a5 - a4);
            m[0] = 2.0 * rcp_w;
            m[5] = 2.0 * rcp_h;
            m[10] = rcp_d;
            m[12] = -(a1 + a0) * rcp_w;
            m[13] = -(a3 + a2) * rcp_h;
            m[14] = -a4 * rcp_d;
            m[15] = 1.0;
        }
        _ => {
            return (m, 0);
        }
    }
    (m, 1)
}

/// Mixed absolute-or-relative closeness for a continuous quantity.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= 1.0e-4 {
        return true;
    }
    diff <= 1.0e-3 * a.abs().max(b.abs()).max(REL_FLOOR)
}

/// Asserts a single `GPU` result matches the independent oracle: the discrete
/// `valid` flag exactly, then every column-major matrix entry to tolerance.
fn assert_parity(gpu: &ProjectionResult, q: &ProjectionQuery, label: &str) {
    let (m, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    for i in 0..16 {
        assert!(
            close(gpu.m[i], m[i]),
            "{label}: m[{i}] mismatch gpu={} oracle={}",
            gpu.m[i],
            m[i]
        );
    }
}

/// Representative master-valid parameters for each of the ten variants.
fn fixture_queries() -> Vec<ProjectionQuery> {
    vec![
        ProjectionQuery::perspective(0, 60.0_f32.to_radians(), 1.777, 0.1, 100.0),
        ProjectionQuery::perspective(1, 45.0_f32.to_radians(), 1.0, 0.1, 100.0),
        ProjectionQuery::perspective(2, 90.0_f32.to_radians(), 1.333, 0.01, 1000.0),
        ProjectionQuery::perspective(3, 50.0_f32.to_radians(), 2.0, 0.5, 0.0),
        ProjectionQuery::perspective(4, 75.0_f32.to_radians(), 0.5, 0.25, 0.0),
        ProjectionQuery::perspective(5, 60.0_f32.to_radians(), 1.5, 0.1, 50.0),
        ProjectionQuery::perspective(6, 30.0_f32.to_radians(), 1.0, 0.2, 200.0),
        ProjectionQuery::orthographic(7, -2.0, 2.0, -1.5, 1.5, 0.1, 100.0),
        ProjectionQuery::orthographic(8, -3.0, 1.0, -2.0, 2.0, 0.5, 50.0),
        ProjectionQuery::orthographic(9, -1.0, 1.0, -1.0, 1.0, 0.1, 10.0),
    ]
}

#[test]
fn every_variant_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjection::new(&ctx);
    let queries = fixture_queries();
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "variant {i} should be valid");
        assert_parity(res, q, &format!("variant[{}]", q.variant_id));
    }
}

#[test]
fn perspective_rh_known_entries() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjection::new(&ctx);
    // A square 90-degree frustum: f = 1/tan(45deg) = 1, so m[0]=m[5]=1.
    let q = ProjectionQuery::perspective(0, 90.0_f32.to_radians(), 1.0, 1.0, 2.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    assert!(close(out[0].m[0], 1.0), "m0={}", out[0].m[0]);
    assert!(close(out[0].m[5], 1.0), "m5={}", out[0].m[5]);
    assert!(close(out[0].m[11], -1.0), "m11={}", out[0].m[11]);
    assert_parity(&out[0], &q, "perspective_rh_known");
}

#[test]
fn out_of_range_variant_zeroes_matrix() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjection::new(&ctx);
    let q = ProjectionQuery::perspective(99, 60.0_f32.to_radians(), 1.0, 0.1, 100.0);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0, "out-of-range variant must clear valid");
    for (i, &v) in out[0].m.iter().enumerate() {
        assert_eq!(v, 0.0, "m[{i}] should be zero for invalid variant");
    }
    assert_parity(&out[0], &q, "out_of_range");
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjection::new(&ctx);
    let queries = vec![
        ProjectionQuery::perspective(0, 60.0_f32.to_radians(), 1.777, 0.1, 100.0),
        ProjectionQuery::orthographic(7, -2.0, 2.0, -1.5, 1.5, 0.1, 100.0),
        ProjectionQuery::perspective(99, 45.0_f32.to_radians(), 1.0, 0.1, 10.0),
        ProjectionQuery::perspective(3, 50.0_f32.to_radians(), 2.0, 0.5, 0.0),
        ProjectionQuery::orthographic(9, -1.0, 1.0, -1.0, 1.0, 0.1, 10.0),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
    // The out-of-range element in the middle must not corrupt its neighbours.
    assert_eq!(out[2].valid, 0);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[3].valid, 1);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjection::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjection::new(&ctx);
    let mut lcg = Lcg::new(0x00C0_FFEE);
    let fovy_choices = [30.0f32, 45.0, 60.0, 90.0, 120.0];
    let aspect_choices = [0.5f32, 1.0, 1.777, 2.35];
    let near_choices = [0.01f32, 0.1, 1.0];
    let far_choices = [10.0f32, 100.0, 1000.0];
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let variant = (lcg.next_u32() % 10) as u32;
        if variant <= 6 {
            // Perspective family.
            let fovy = fovy_choices[(lcg.next_u32() % 5) as usize].to_radians();
            let aspect = aspect_choices[(lcg.next_u32() % 4) as usize];
            let near = near_choices[(lcg.next_u32() % 3) as usize];
            let far = far_choices[(lcg.next_u32() % 3) as usize];
            // Guarantee far > near for the finite variants.
            let far = far.max(near + 1.0);
            queries.push(ProjectionQuery::perspective(
                variant, fovy, aspect, near, far,
            ));
        } else {
            // Orthographic family: l < r, b < t, near < far.
            let left = -1.0 - lcg.next_range(0.0, 3.0);
            let right = 1.0 + lcg.next_range(0.0, 3.0);
            let bottom = -1.0 - lcg.next_range(0.0, 3.0);
            let top = 1.0 + lcg.next_range(0.0, 3.0);
            let near = near_choices[(lcg.next_u32() % 3) as usize];
            let far = far_choices[(lcg.next_u32() % 3) as usize].max(near + 1.0);
            queries.push(ProjectionQuery::orthographic(
                variant, left, right, bottom, top, near, far,
            ));
        }
    }
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "sweep[{i}] should be valid");
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

    /// A `[lo, hi)` sample derived from `next_unit`.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_unit()
    }
}
