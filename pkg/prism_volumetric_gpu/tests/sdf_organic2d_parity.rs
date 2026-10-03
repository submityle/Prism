//! Real-device parity for the analytic two-dimensional organic
//! signed-distance twin:
//! [`GpuSdfOrganic2d`](prism_volumetric_gpu::sdf_organic2d::GpuSdfOrganic2d)
//! must reproduce the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the Inigo Quilez
//! `heart_2d`, `egg_2d`, `moon` and `cross_2d` outlines — across interior,
//! exterior and surface points, every feature branch of each shape, and a
//! randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed forms: the heart's
//! lobe/cusp/flank split, the egg's three arcs, the crescent's cusp-tip and
//! difference branches, and the cross's folded box. Because the reference and
//! this oracle are both scalar `f32`, a `GPU == oracle` pass is direct evidence
//! the ported kernel computes the same distances the reference does.
//!
//! # Parity criterion
//!
//! Every distance threads through products, quotients and a `sqrt`, so a `GPU`
//! result may land a few units in the last place from the scalar oracle; each
//! is asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with a relative
//! floor of `1e-6` so a near-zero expected value does not inflate the relative
//! error.
//!
//! # Conditioning
//!
//! Each shape's internal branch select sits on a locus where the two branches
//! agree (the heart's `|x| + y = 1` diagonal, the egg's arc seams, the moon's
//! cusp line, the cross's reentrant corner), and where a `signum` flips the
//! magnitude simultaneously passes through zero, so no branch disagreement can
//! produce a distance cliff. The randomized sweep still rejects samples near
//! each branch, fold or sign boundary to keep the comparison far from any such
//! edge, and draws only well-formed shape parameters (`egg_ra > egg_rb > 0`,
//! `|moon_ra - moon_rb| < moon_d < moon_ra + moon_rb`, `cross_arm >=
//! cross_thickness`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_organic2d::{GpuSdfOrganic2d, SdfOrganic2dQuery, SdfOrganic2dResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on any distance. A `GPU` `sqrt`/divide may land a few units
/// in the last place from the scalar oracle; `1e-4` admits that legal slack
/// while still failing a wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative bound on any distance, applied for larger magnitudes where a few
/// units in the last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// `sqrt(2) / 4`, the heart's lobe-circle radius, matching the reference
/// constant exactly.
const HEART_R: f32 = 0.353_553_38;

/// `sqrt(3)`, the egg's cheek-arc geometry constant, matching the reference
/// constant exactly.
const EGG_K: f32 = 1.732_050_8;

/// Returns whether `a` and `b` agree within the absolute or relative bound
/// (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Euclidean length of a 2-vector, matching the reference `length2`.
fn length2(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Rust `f32::signum` reimplemented to match the kernel's `select`-based sign:
/// `+1` for a non-negative argument, `-1` otherwise. Where each shape uses this
/// the magnitude passes through zero at the same place, so the choice at an
/// exact zero never governs.
fn sign_of(x: f32) -> f32 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::heart_2d`: the upper lobe circle above the
/// `|x| + y = 1` diagonal, else the nearer of the top cusp and the `y = x`
/// flank signed by `sign(|x| - y)`.
fn heart_2d_host(point: [f32; 2]) -> f32 {
    let x = point[0].abs();
    let y = point[1];
    if y + x > 1.0 {
        let dx = x - 0.25;
        let dy = y - 0.75;
        (dx * dx + dy * dy).sqrt() - HEART_R
    } else {
        let cx = x;
        let cy = y - 1.0;
        let d_cusp = cx * cx + cy * cy;
        let s = 0.5 * (x + y).max(0.0);
        let fx = x - s;
        let fy = y - s;
        let d_flank = fx * fx + fy * fy;
        d_cusp.min(d_flank).sqrt() * sign_of(x - y)
    }
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::egg_2d`: the three-arc egg folded to `x >= 0`
/// and routed to the bottom circle, the top cap or a cheek arc.
fn egg_2d_host(point: [f32; 2], ra: f32, rb: f32) -> f32 {
    let px = point[0].abs();
    let py = point[1];
    let r = ra - rb;
    let d = if py < 0.0 {
        length2([px, py]) - r
    } else if EGG_K * (px + r) < py {
        length2([px, py - EGG_K * r])
    } else {
        length2([px + r, py]) - 2.0 * r
    };
    d - rb
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::moon`: the crescent as a constructive-solid
/// difference of two disks with an explicit cusp-tip branch.
fn moon_host(point: [f32; 2], d: f32, ra: f32, rb: f32) -> f32 {
    let p = [point[0], point[1].abs()];
    let a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    let b = (ra * ra - a * a).max(0.0).sqrt();
    if d * (p[0] * b - p[1] * a) > d * d * (b - p[1]).max(0.0) {
        return length2([p[0] - a, p[1] - b]);
    }
    let outer = length2(p) - ra;
    let inner = length2([p[0] - d, p[1]]) - rb;
    outer.max(-inner)
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::cross_2d`: the rounded plus folded into the
/// octant `x >= y >= 0`, a box distance outside and the reentrant corner
/// inside.
fn cross_2d_host(point: [f32; 2], arm: f32, thickness: f32, r: f32) -> f32 {
    let mut p = [point[0].abs(), point[1].abs()];
    if p[1] > p[0] {
        p = [p[1], p[0]];
    }
    let q = [p[0] - arm, p[1] - thickness];
    let k = q[0].max(q[1]);
    let w = if k > 0.0 { q } else { [thickness - p[0], -k] };
    sign_of(k) * length2([w[0].max(0.0), w[1].max(0.0)]) - r
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at ten-thousandth resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 10_001) as f32 / 10_000.0 * (hi - lo)
}

/// Default egg radii used by the point-focused fixtures.
const EGG_RA: f32 = 1.0;
const EGG_RB: f32 = 0.4;
/// Default crescent parameters (`|ra - rb| = 0.3 < d = 0.8 < ra + rb = 1.7`).
const MOON_D: f32 = 0.8;
const MOON_RA: f32 = 1.0;
const MOON_RB: f32 = 0.7;
/// Default rounded-cross parameters (`arm >= thickness`).
const CROSS_ARM: f32 = 1.0;
const CROSS_THICK: f32 = 0.35;
const CROSS_R: f32 = 0.1;

/// Builds one query at `point` with the default per-shape parameters so every
/// shape is exercised on every dispatch.
fn q_at(point: [f32; 2]) -> SdfOrganic2dQuery {
    SdfOrganic2dQuery {
        point,
        egg_ra: EGG_RA,
        egg_rb: EGG_RB,
        moon_d: MOON_D,
        moon_ra: MOON_RA,
        moon_rb: MOON_RB,
        cross_arm: CROSS_ARM,
        cross_thickness: CROSS_THICK,
        cross_r: CROSS_R,
    }
}

/// Dispatches `queries` and asserts every distance matches the host oracles.
fn check_batch(ctx: &GpuContext, gpu: &GpuSdfOrganic2d, queries: &[SdfOrganic2dQuery]) {
    let got: Vec<SdfOrganic2dResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden_heart = heart_2d_host(q.point);
        let golden_egg = egg_2d_host(q.point, q.egg_ra, q.egg_rb);
        let golden_moon = moon_host(q.point, q.moon_d, q.moon_ra, q.moon_rb);
        let golden_cross = cross_2d_host(q.point, q.cross_arm, q.cross_thickness, q.cross_r);
        assert!(
            close(r.heart, golden_heart),
            "heart mismatch: gpu={} golden={} (point={:?})",
            r.heart,
            golden_heart,
            q.point
        );
        assert!(
            close(r.egg, golden_egg),
            "egg mismatch: gpu={} golden={} (point={:?} ra={} rb={})",
            r.egg,
            golden_egg,
            q.point,
            q.egg_ra,
            q.egg_rb
        );
        assert!(
            close(r.moon, golden_moon),
            "moon mismatch: gpu={} golden={} (point={:?} d={} ra={} rb={})",
            r.moon,
            golden_moon,
            q.point,
            q.moon_d,
            q.moon_ra,
            q.moon_rb
        );
        assert!(
            close(r.cross, golden_cross),
            "cross mismatch: gpu={} golden={} (point={:?} arm={} thickness={} r={})",
            r.cross,
            golden_cross,
            q.point,
            q.cross_arm,
            q.cross_thickness,
            q.cross_r
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf_organic2d parity: no wgpu adapter available");
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no results");
}

#[test]
fn heart_interior_and_exterior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    // Interior points sit below the cusp; a far point is well outside.
    let queries = [
        q_at([0.0, 0.6]),
        q_at([0.0, 0.3]),
        q_at([0.3, 0.9]),
        q_at([2.0, 2.0]),
        q_at([-1.5, -1.0]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn heart_upper_lobe_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    // Points above the |x| + y = 1 diagonal exercise the lobe-circle branch.
    let queries = [
        q_at([0.5, 1.0]),
        q_at([-0.5, 1.0]),
        q_at([0.35, 0.95]),
        q_at([0.6, 1.3]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn heart_lower_flank_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    // Points below the diagonal exercise the cusp/flank nearer-feature branch
    // on both sides of the y = x sign locus.
    let queries = [
        q_at([0.7, 0.1]),
        q_at([0.1, 0.7]),
        q_at([-0.7, 0.1]),
        q_at([0.4, -0.3]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn egg_bottom_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    // py < 0 routes to the bottom circle; include interior and exterior.
    let queries = [
        q_at([0.0, -1.0]),
        q_at([0.0, -0.3]),
        q_at([0.4, -0.6]),
        q_at([1.2, -0.9]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn egg_cap_and_cheek_branches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    // High, near-axis points hit the top cap; wide points hit a cheek arc.
    let queries = [
        q_at([0.0, 1.4]),
        q_at([0.05, 1.2]),
        q_at([0.8, 0.3]),
        q_at([-0.7, 0.2]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn moon_cusp_branch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    // Points beyond the cusp tip take the distance-to-cusp branch.
    let queries = [q_at([1.1, 0.9]), q_at([1.0, -0.8]), q_at([0.9, 1.1])];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn moon_difference_branch_and_symmetry() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    // Points inside/near the crescent body use the disk-difference branch; the
    // field is symmetric about the x axis, so mirrored pairs must agree.
    let queries = [
        q_at([-0.2, 0.0]),
        q_at([0.2, 0.3]),
        q_at([0.2, -0.3]),
        q_at([-0.6, 0.4]),
        q_at([-0.6, -0.4]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn cross_interior_and_exterior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    // Centre and arm interiors (k < 0) plus corner/exterior points (k > 0).
    let queries = [
        q_at([0.0, 0.0]),
        q_at([0.8, 0.0]),
        q_at([0.0, 0.2]),
        q_at([1.3, 0.6]),
        q_at([1.5, 1.5]),
        q_at([-1.2, 0.1]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn mixed_single_dispatch_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    // A heterogeneous batch with varied per-shape parameters in one dispatch.
    let queries = [
        SdfOrganic2dQuery {
            point: [0.3, 0.5],
            egg_ra: 1.2,
            egg_rb: 0.5,
            moon_d: 0.9,
            moon_ra: 1.1,
            moon_rb: 0.6,
            cross_arm: 1.3,
            cross_thickness: 0.4,
            cross_r: 0.05,
        },
        SdfOrganic2dQuery {
            point: [-0.8, 0.9],
            egg_ra: 0.9,
            egg_rb: 0.3,
            moon_d: 0.7,
            moon_ra: 1.0,
            moon_rb: 0.8,
            cross_arm: 0.8,
            cross_thickness: 0.2,
            cross_r: 0.15,
        },
        SdfOrganic2dQuery {
            point: [1.4, -1.1],
            egg_ra: 1.5,
            egg_rb: 0.6,
            moon_d: 1.0,
            moon_ra: 1.2,
            moon_rb: 0.5,
            cross_arm: 1.1,
            cross_thickness: 0.5,
            cross_r: 0.1,
        },
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfOrganic2d::new(&ctx);
    let mut state: u64 = 0x2d6b_e17a_84c1_90f3;
    // Margin kept away from each shape's branch/fold/sign locus. Those loci are
    // continuous (the branches agree there), so this is extra safety only.
    let margin = 0.03_f32;
    let mut queries = Vec::with_capacity(384);
    while queries.len() < 384 {
        let point = [draw(&mut state, -1.8, 1.8), draw(&mut state, -1.8, 1.8)];

        // Egg: ra > rb > 0.
        let egg_ra = draw(&mut state, 0.6, 1.5);
        let egg_rb = draw(&mut state, 0.2, egg_ra - 0.1);
        // Moon: |ra - rb| < d < ra + rb.
        let moon_ra = draw(&mut state, 0.8, 1.5);
        let moon_rb = draw(&mut state, 0.3, 0.9);
        let lo = (moon_ra - moon_rb).abs() + margin;
        let hi = moon_ra + moon_rb - margin;
        if hi <= lo {
            continue;
        }
        let moon_d = draw(&mut state, lo, hi);
        // Cross: arm >= thickness.
        let cross_arm = draw(&mut state, 0.5, 1.5);
        let cross_thickness = draw(&mut state, 0.1, cross_arm);
        let cross_r = draw(&mut state, 0.0, 0.2);

        // Heart branch/sign margins.
        let hx = point[0].abs();
        let hy = point[1];
        if (hy + hx - 1.0).abs() < margin || (hx - hy).abs() < margin {
            continue;
        }
        // Egg branch margins (py near 0; cheek/cap seam).
        let r = egg_ra - egg_rb;
        if hy.abs() < margin || (EGG_K * (hx + r) - hy).abs() < margin {
            continue;
        }
        // Moon cusp-line margin (scaled by d*d naturally).
        let mp = [point[0], point[1].abs()];
        let ma = (moon_ra * moon_ra - moon_rb * moon_rb + moon_d * moon_d) / (2.0 * moon_d);
        let mb = (moon_ra * moon_ra - ma * ma).max(0.0).sqrt();
        let cusp = moon_d * (mp[0] * mb - mp[1] * ma) - moon_d * moon_d * (mb - mp[1]).max(0.0);
        if cusp.abs() < margin {
            continue;
        }
        // Cross fold and k margins.
        let mut cp = [point[0].abs(), point[1].abs()];
        if cp[1] > cp[0] {
            cp = [cp[1], cp[0]];
        }
        if (point[0].abs() - point[1].abs()).abs() < margin {
            continue;
        }
        let ck = (cp[0] - cross_arm).max(cp[1] - cross_thickness);
        if ck.abs() < margin {
            continue;
        }

        queries.push(SdfOrganic2dQuery {
            point,
            egg_ra,
            egg_rb,
            moon_d,
            moon_ra,
            moon_rb,
            cross_arm,
            cross_thickness,
            cross_r,
        });
    }
    check_batch(&ctx, &gpu, &queries);
}
