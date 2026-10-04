//! Real-device parity for the backstop-clamp twin:
//! [`GpuSoftApplyBackstop`](prism_volumetric_gpu::soft_apply_backstop::GpuSoftApplyBackstop)
//! must reproduce the `CPU` golden
//! `prism_physics_core::soft::collision::body::apply_backstop`, the clamp that
//! keeps a soft-body particle from sinking too far behind a skinned surface.
//!
//! The oracle here is an independent re-implementation of that closed form —
//! `len_sq = dot(normal, normal)`; an inert normal (`len_sq <= 1e-12`) echoes
//! the position; otherwise `n = normal / sqrt(len_sq)`,
//! `s = dot(n, pos - origin)`, `min_s = -distance`, and `s < min_s` pushes the
//! point to `pos + n * (min_s - s)` while any other case holds — written out
//! directly so the test never imports `prism_physics_core` or
//! `prism_render_architecture`.
//!
//! The fixtures cover the branches the kernel must honor: an inert (near-zero)
//! normal echoes the position with `valid = 0`, a particle already on the front
//! side (`s >= min_s`) echoes with `valid = 0`, a particle behind the limit
//! (`s < min_s`) is pushed with `valid = 1` and the pushed point is explicitly
//! checked to land on the limiting plane (`dot(n, new_pos - origin) ≈ min_s`),
//! and a mixed multi-element batch validates the `std430` stride. A `512`-step
//! sweep over random positions, anchors, normals and distances follows, kept a
//! safe margin away from both the inert-normal and `s == min_s` knees (which the
//! named fixtures already pin), plus an empty batch the host short-circuits with
//! no dispatch.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The position (`new_px`, `new_py`, `new_pz`) is a continuous `f32` clamp, so
//! parity uses an absolute-or-relative tolerance (`abs <= 1e-4 || rel <= 1e-3`,
//! with a `1e-6` relative floor so near-zero coordinates compare on the absolute
//! leg). `valid` is discrete and compared exactly: `1` only when the particle
//! was pushed, `0` for an inert normal or an already-ahead echo.
//!
//! Provenance: 孪生自本仓 `prism_physics_core::soft::collision::body`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::soft_apply_backstop::{GpuSoftApplyBackstop, SoftApplyBackstopQuery};
use prism_volumetric_gpu::GpuContext;

/// Signed-distance guard matching the golden `EPS_LEN_SQ`.
const EPS_LEN_SQ: f32 = 1.0e-12;
/// Absolute tolerance leg for the continuous position comparison.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance leg for the continuous position comparison.
const REL_EPS: f32 = 1.0e-3;
/// Relative-tolerance floor so near-zero coordinates fall back to the absolute
/// leg instead of demanding an impossible relative match.
const REL_FLOOR: f32 = 1.0e-6;

/// Absolute-or-relative closeness for a single `f32` lane.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff <= REL_EPS * scale
}

/// Independent oracle for one query, returning `([new_px, new_py, new_pz],
/// valid)`.
///
/// Reproduces the body of `apply_backstop`: an inert normal echoes the position
/// with `valid = 0`; otherwise the point is pushed onto the limiting plane when
/// its signed distance drops below `-distance` (`valid = 1`), and held with
/// `valid = 0` when it is already on the front side.
fn oracle(q: &SoftApplyBackstopQuery) -> ([f32; 3], u32) {
    let pos = [q.px, q.py, q.pz];
    let len_sq = q.nx * q.nx + q.ny * q.ny + q.nz * q.nz;
    if len_sq <= EPS_LEN_SQ {
        return (pos, 0);
    }
    let inv = 1.0 / len_sq.sqrt();
    let n = [q.nx * inv, q.ny * inv, q.nz * inv];
    let d = [q.px - q.ox, q.py - q.oy, q.pz - q.oz];
    let s = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
    let min_s = -q.distance;
    if s < min_s {
        let push = min_s - s;
        (
            [q.px + n[0] * push, q.py + n[1] * push, q.pz + n[2] * push],
            1,
        )
    } else {
        (pos, 0)
    }
}

/// Dispatches one query and asserts the device result matches the oracle.
fn assert_parity(ctx: &GpuContext, gpu: &GpuSoftApplyBackstop, q: SoftApplyBackstopQuery) {
    let results = gpu.evaluate(ctx, &[q]);
    assert_eq!(results.len(), 1, "one result per query");
    let r = results[0];
    let ([nx, ny, nz], valid) = oracle(&q);
    assert_eq!(r.valid, valid, "valid mismatch: query={q:?}");
    assert!(
        close(r.new_px, nx) && close(r.new_py, ny) && close(r.new_pz, nz),
        "position mismatch: query={q:?} gpu=({}, {}, {}) oracle=({nx}, {ny}, {nz})",
        r.new_px,
        r.new_py,
        r.new_pz
    );
}

#[test]
fn zero_normal_echoes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftApplyBackstop::new(&ctx);
    // A (near) zero normal has no defined plane, so the position is echoed.
    assert_parity(
        &ctx,
        &gpu,
        SoftApplyBackstopQuery::new(1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    );
}

#[test]
fn front_side_echoes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftApplyBackstop::new(&ctx);
    // Particle well ahead of the anchor: s = +2 >= min_s = -1, so it holds.
    assert_parity(
        &ctx,
        &gpu,
        SoftApplyBackstopQuery::new(0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    );
}

#[test]
fn behind_limit_pushes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftApplyBackstop::new(&ctx);
    // Particle behind the limit along +y: pos.y = -3, origin at 0, distance 1,
    // so s = -3 < min_s = -1 and it is pushed to y = -1.
    let q = SoftApplyBackstopQuery::new(0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0);
    assert_parity(&ctx, &gpu, q);
    // Explicit landing check: the pushed point sits on the limiting plane.
    let r = gpu.evaluate(&ctx, &[q])[0];
    assert_eq!(r.valid, 1, "pushed query reports valid");
    let s_after = r.new_py; // n = (0,1,0), origin 0 => dot(n, new_pos-origin) = new_py.
    assert!(
        close(s_after, -q.distance),
        "pushed point should land on the limiting plane: s_after={s_after} min_s={}",
        -q.distance
    );
}

#[test]
fn pushes_along_non_axis_normal() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftApplyBackstop::new(&ctx);
    // Non-unit, non-axis normal to exercise the normalize path and the plane
    // projection off-origin.
    let q = SoftApplyBackstopQuery::new(-2.0, -2.0, 0.0, 1.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.5);
    assert_parity(&ctx, &gpu, q);
    let r = gpu.evaluate(&ctx, &[q])[0];
    if r.valid == 1 {
        // n = (0.6, 0.8, 0): verify the clamped point lands on the plane.
        let s_after = 0.6 * (r.new_px - 1.0) + 0.8 * (r.new_py - 0.0);
        assert!(
            close(s_after, -q.distance),
            "pushed point should land on the limiting plane: s_after={s_after} min_s={}",
            -q.distance
        );
    }
}

#[test]
fn mixed_batch_validates_stride() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftApplyBackstop::new(&ctx);
    // A multi-element batch exercises the std430 stride: adjacent slots must
    // decode independently and in order, spanning inert-normal echo,
    // front-side echo and two push cases.
    let queries = [
        SoftApplyBackstopQuery::new(1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        SoftApplyBackstopQuery::new(0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        SoftApplyBackstopQuery::new(0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        SoftApplyBackstopQuery::new(-2.0, -2.0, 0.0, 1.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.5),
        SoftApplyBackstopQuery::new(0.5, 0.5, -4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.5),
    ];
    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let ([nx, ny, nz], valid) = oracle(q);
        assert_eq!(r.valid, valid, "batch valid mismatch: query={q:?}");
        assert!(
            close(r.new_px, nx) && close(r.new_py, ny) && close(r.new_pz, nz),
            "batch position mismatch: query={q:?} gpu=({}, {}, {}) oracle=({nx}, {ny}, {nz})",
            r.new_px,
            r.new_py,
            r.new_pz
        );
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSoftApplyBackstop::new(&ctx);
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
    let gpu = GpuSoftApplyBackstop::new(&ctx);
    let mut rng = Lcg::new(0x5A_CC_17_31);
    // Margins keep the normal well away from the inert knee and the signed
    // distance well away from the push/hold knee, so both branches are
    // decisive; those boundaries are pinned by the named fixtures instead.
    const LEN_SQ_MARGIN: f32 = 0.01;
    const KNEE_MARGIN: f32 = 0.01;
    let mut queries = Vec::with_capacity(512);
    while queries.len() < 512 {
        let nx = rng.next_range(-2.0, 2.0);
        let ny = rng.next_range(-2.0, 2.0);
        let nz = rng.next_range(-2.0, 2.0);
        let len_sq = nx * nx + ny * ny + nz * nz;
        // Reject near-zero normals so the normalize path is well-conditioned.
        if len_sq < LEN_SQ_MARGIN {
            continue;
        }
        let px = rng.next_range(-2.0, 2.0);
        let py = rng.next_range(-2.0, 2.0);
        let pz = rng.next_range(-2.0, 2.0);
        let ox = rng.next_range(-2.0, 2.0);
        let oy = rng.next_range(-2.0, 2.0);
        let oz = rng.next_range(-2.0, 2.0);
        let distance = rng.next_range(0.0, 2.0);
        // Reject queries sitting on the s == min_s knee so the branch is
        // decisive under f32 noise.
        let inv = 1.0 / len_sq.sqrt();
        let s = (px - ox) * nx * inv + (py - oy) * ny * inv + (pz - oz) * nz * inv;
        let min_s = -distance;
        if (s - min_s).abs() < KNEE_MARGIN {
            continue;
        }
        queries.push(SoftApplyBackstopQuery::new(
            px, py, pz, ox, oy, oz, nx, ny, nz, distance,
        ));
    }

    let results = gpu.evaluate(&ctx, &queries);
    assert_eq!(results.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(results.iter()) {
        let ([nx, ny, nz], valid) = oracle(q);
        assert_eq!(r.valid, valid, "sweep valid mismatch: query={q:?}");
        assert!(
            close(r.new_px, nx) && close(r.new_py, ny) && close(r.new_pz, nz),
            "sweep position mismatch: query={q:?} gpu=({}, {}, {}) oracle=({nx}, {ny}, {nz})",
            r.new_px,
            r.new_py,
            r.new_pz
        );
    }
}
