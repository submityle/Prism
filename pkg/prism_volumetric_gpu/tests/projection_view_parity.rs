//! Real-device parity for the view-matrix twin:
//! [`GpuProjectionView`](prism_volumetric_gpu::projection_view::GpuProjectionView)
//! must reproduce the `CPU` golden `look_at`/`look_to` constructors of
//! `prism_math::projection`. Each query selects one of four orientation
//! variants by an enum id and returns a column-major `[f32; 16]` matching
//! `Mat4::from_cols` (column `0` is `m[0..4]`, column `1` is `m[4..8]`, and so
//! on).
//!
//! The oracle here is an independent re-implementation of those closed forms —
//! `f = normalize(dir)`, the right/left-handed side and up vectors from cross
//! products, and the eye-dot translation column — written out directly in
//! `f32` with hand-rolled `cross`/`dot`/`normalize` so the test never imports
//! `prism_math`, `prism_render_architecture`, `prism_physics_core`, or `glam`.
//!
//! The fixtures cover every variant `0..=3` with representative camera poses, a
//! known-basis sanity check, a mixed batch of two or more elements (including
//! an out-of-range `variant_id` and a degenerate direction that must zero the
//! matrix and clear `valid`) that validates the `std430` stride, and an empty
//! batch the host short-circuits with no dispatch. A sweep over random
//! master-valid poses follows, cycling through all four variants.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The device builds the basis with the native `f32` `length`/`cross`/`dot`,
//! while the oracle uses the same `f32` operator order; the two agree to well
//! within tolerance. Each of the sixteen continuous matrix entries is compared
//! with `abs <= 1e-4 || rel <= 1e-3` (`REL_FLOOR = 1e-6`); the discrete `valid`
//! flag is compared exactly. A recognised `variant_id` (`<= 3`) with a finite
//! basis yields `valid = 1`; anything else yields `valid = 0` with an all-zero
//! matrix.
//!
//! Provenance: 孪生自本仓 `prism_math::projection`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::projection_view::{
    GpuProjectionView, ProjectionViewQuery, ProjectionViewResult,
};
use prism_volumetric_gpu::GpuContext;

/// Relative-tolerance floor, so the relative test never divides by a magnitude
/// smaller than this.
const REL_FLOOR: f32 = 1.0e-6;

/// Degenerate guard threshold, matching the kernel: a direction or cross
/// product whose length is `<= 1e-20` collapses the basis.
const EPS: f32 = 1.0e-20;

/// Vector difference, component-wise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Standard right-handed cross product.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean length, `sqrt(dot(v, v))`.
fn length(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}

/// Golden-form normalization: `v * (1 / length(v))`, matching
/// `prism_math::Vec3::normalize` and the kernel's reciprocal-scale form so the
/// oracle shares the same rounding.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let inv = 1.0 / length(v);
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Independent host oracle: reproduces the selected `prism_math::projection`
/// view constructor in the golden operator order and `f32` precision, returning
/// the column-major matrix and the discrete validity flag. A degenerate basis
/// or an unrecognised `variant_id` returns an all-zero matrix with `valid = 0`.
fn oracle(q: &ProjectionViewQuery) -> ([f32; 16], u32) {
    let mut m = [0.0f32; 16];
    let eye = q.eye;
    let up = q.up;
    let vid = q.variant_id;
    if vid > 3 {
        return (m, 0);
    }
    // look_at (0, 2) forms dir = target - eye; look_to (1, 3) reads it directly.
    let dir = if vid == 0 || vid == 2 {
        sub(q.target_or_dir, eye)
    } else {
        q.target_or_dir
    };
    let dir_len = length(dir);
    if dir_len <= EPS {
        return (m, 0);
    }
    let f = normalize(dir);
    if vid == 0 || vid == 1 {
        // Right-handed basis.
        let sc = cross(f, up);
        if length(sc) <= EPS {
            return (m, 0);
        }
        let s = normalize(sc);
        let u = cross(s, f);
        m[0] = s[0];
        m[1] = u[0];
        m[2] = -f[0];
        m[3] = 0.0;
        m[4] = s[1];
        m[5] = u[1];
        m[6] = -f[1];
        m[7] = 0.0;
        m[8] = s[2];
        m[9] = u[2];
        m[10] = -f[2];
        m[11] = 0.0;
        m[12] = -dot(s, eye);
        m[13] = -dot(u, eye);
        m[14] = dot(f, eye);
        m[15] = 1.0;
    } else {
        // Left-handed basis (vid == 2 or 3).
        let sc = cross(up, f);
        if length(sc) <= EPS {
            return (m, 0);
        }
        let s = normalize(sc);
        let u = cross(f, s);
        m[0] = s[0];
        m[1] = u[0];
        m[2] = f[0];
        m[3] = 0.0;
        m[4] = s[1];
        m[5] = u[1];
        m[6] = f[1];
        m[7] = 0.0;
        m[8] = s[2];
        m[9] = u[2];
        m[10] = f[2];
        m[11] = 0.0;
        m[12] = -dot(s, eye);
        m[13] = -dot(u, eye);
        m[14] = -dot(f, eye);
        m[15] = 1.0;
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
fn assert_parity(gpu: &ProjectionViewResult, q: &ProjectionViewQuery, label: &str) {
    let (m, valid) = oracle(q);
    assert_eq!(gpu.valid, valid, "{label}: valid flag mismatch");
    if valid == 0 {
        for (i, &v) in gpu.m.iter().enumerate() {
            assert_eq!(v, 0.0, "{label}: m[{i}] should be zero for invalid result");
        }
        return;
    }
    for i in 0..16 {
        assert!(
            close(gpu.m[i], m[i]),
            "{label}: m[{i}] mismatch gpu={} oracle={}",
            gpu.m[i],
            m[i]
        );
    }
}

/// Representative master-valid camera poses, one per variant.
fn fixture_queries() -> Vec<ProjectionViewQuery> {
    vec![
        ProjectionViewQuery::look_at(0, [2.0, 3.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ProjectionViewQuery::look_to(1, [1.0, 2.0, 3.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ProjectionViewQuery::look_at(2, [-4.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ProjectionViewQuery::look_to(3, [0.0, 5.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ]
}

#[test]
fn every_variant_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjectionView::new(&ctx);
    let queries = fixture_queries();
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 1, "variant {i} should be valid");
        assert_parity(res, q, &format!("variant[{}]", q.variant_id));
    }
}

#[test]
fn look_to_rh_known_basis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjectionView::new(&ctx);
    // Eye at origin, looking down -Z with +Y up: the right-handed view is the
    // identity rotation, so s = +X, u = +Y, -f = +Z. With eye at origin the
    // translation column is zero.
    let q = ProjectionViewQuery::look_to(1, [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 1);
    // Column 0 = (s.x, u.x, -f.x, 0) = (1, 0, 0, 0).
    assert!(close(out[0].m[0], 1.0), "m0={}", out[0].m[0]);
    assert!(close(out[0].m[5], 1.0), "m5={}", out[0].m[5]);
    assert!(close(out[0].m[10], 1.0), "m10={}", out[0].m[10]);
    assert!(close(out[0].m[15], 1.0), "m15={}", out[0].m[15]);
    assert_parity(&out[0], &q, "look_to_rh_known");
}

#[test]
fn out_of_range_variant_zeroes_matrix() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjectionView::new(&ctx);
    let q = ProjectionViewQuery::look_at(99, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let out = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].valid, 0, "out-of-range variant must clear valid");
    for (i, &v) in out[0].m.iter().enumerate() {
        assert_eq!(v, 0.0, "m[{i}] should be zero for invalid variant");
    }
    assert_parity(&out[0], &q, "out_of_range");
}

#[test]
fn degenerate_direction_zeroes_matrix() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjectionView::new(&ctx);
    let queries = vec![
        // Zero-length direction: target equals eye.
        ProjectionViewQuery::look_at(0, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]),
        // Direction parallel to up: cross collapses.
        ProjectionViewQuery::look_to(1, [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0]),
        // Left-handed parallel-to-up case.
        ProjectionViewQuery::look_to(3, [0.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_eq!(res.valid, 0, "degenerate[{i}] must clear valid");
        assert_parity(res, q, &format!("degenerate[{i}]"));
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjectionView::new(&ctx);
    let queries = vec![
        ProjectionViewQuery::look_at(0, [2.0, 3.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ProjectionViewQuery::look_at(99, [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ProjectionViewQuery::look_to(1, [1.0, 2.0, 3.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ProjectionViewQuery::look_to(3, [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 0.0]),
        ProjectionViewQuery::look_at(2, [-4.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    let out = gpu.evaluate(&ctx, &queries);
    assert_eq!(out.len(), queries.len());
    for (i, (res, q)) in out.iter().zip(queries.iter()).enumerate() {
        assert_parity(res, q, &format!("mixed[{i}]"));
    }
    // The out-of-range and degenerate elements must not corrupt their
    // neighbours.
    assert_eq!(out[1].valid, 0);
    assert_eq!(out[3].valid, 0);
    assert_eq!(out[0].valid, 1);
    assert_eq!(out[2].valid, 1);
    assert_eq!(out[4].valid, 1);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjectionView::new(&ctx);
    let out = gpu.evaluate(&ctx, &[]);
    assert!(out.is_empty());
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuProjectionView::new(&ctx);
    let mut lcg = Lcg::new(0x00C0_FFEE);
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let variant = lcg.next_u32() % 4;
        // A camera pose with eye and target well separated and a direction far
        // from parallel to up, so the basis is master-valid and clear of the
        // 1e-20 collapse knees.
        let eye = [
            lcg.next_range(-6.0, 6.0),
            lcg.next_range(-6.0, 6.0),
            lcg.next_range(-6.0, 6.0),
        ];
        let up = [0.0, 1.0, 0.0];
        // Build a direction with a substantial horizontal component so
        // cross(dir, up) stays well away from zero.
        let dir = [
            lcg.next_range(-4.0, 4.0),
            lcg.next_range(-1.0, 1.0),
            lcg.next_range(-4.0, 4.0),
        ];
        let horiz = (dir[0] * dir[0] + dir[2] * dir[2]).sqrt();
        if horiz < 1.0 {
            continue;
        }
        let dir_len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if dir_len < 1.0 {
            continue;
        }
        let q = if variant == 0 || variant == 2 {
            // look_at: synthesize a target from eye + dir so the direction is
            // the same well-conditioned vector.
            let target = [eye[0] + dir[0], eye[1] + dir[1], eye[2] + dir[2]];
            ProjectionViewQuery::look_at(variant, eye, target, up)
        } else {
            ProjectionViewQuery::look_to(variant, eye, dir, up)
        };
        queries.push(q);
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
