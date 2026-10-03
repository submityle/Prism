//! Real-device parity for the analytic two-dimensional arc/disk
//! signed-distance twin:
//! [`GpuSdfArc2d`](prism_volumetric_gpu::sdf_arc2d::GpuSdfArc2d) must reproduce
//! the `CPU` closed forms of
//! `prism_render_architecture::ray_scene::sdf_primitives` — the Inigo Quilez
//! `pie`, `cut_disk_2d`, `horseshoe_2d` and `tunnel_2d` outlines — across
//! interior, exterior and surface points, every feature branch of each shape,
//! and a randomized sweep compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed forms: the pie wedge's
//! arc-versus-edge split, the cut disk's arc/chord/corner regions, the
//! horseshoe's rotate-and-fold strip, and the tunnel's box-plus-cap distance.
//! Because the reference and this oracle are both scalar `f32`, a
//! `GPU == oracle` pass is direct evidence the ported kernel computes the same
//! distances the reference does.
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
//! agree (the pie wedge's edge ray, the cut disk's region seams, the
//! horseshoe's rotated-frame axes, the tunnel's wall/cap seam), and where a
//! `signum` flips the magnitude it simultaneously passes through zero, so no
//! branch disagreement can produce a distance cliff. The randomized sweep still
//! rejects samples near each branch, fold or sign boundary to keep the
//! comparison far from any such edge, and draws only well-formed shape
//! parameters (`-radius < cut_height < radius`, `hs_thickness < hs_radius`,
//! `hs_cos > 0`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::sdf_primitives`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::sdf_arc2d::{GpuSdfArc2d, SdfArc2dQuery, SdfArc2dResult};
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

/// Baked pie half-aperture `(sin, cos)` for a `60`-degree wedge (`sqrt(3)/2`,
/// `1/2`), keeping `cos > 0`.
const PIE_SIN: f32 = 0.866_025_4;
const PIE_COS: f32 = 0.5;

/// Baked horseshoe mouth-aperture `(sin, cos)` for a `30`-degree opening
/// (`1/2`, `sqrt(3)/2`), keeping `cos > 0`.
const HS_SIN: f32 = 0.5;
const HS_COS: f32 = 0.866_025_4;

/// Default pie sector radius for the point-focused fixtures.
const PIE_RADIUS: f32 = 1.0;
/// Default cut-disk radius and chord height (`-radius < cut_height < radius`).
const CUT_RADIUS: f32 = 1.0;
const CUT_HEIGHT: f32 = 0.3;
/// Default horseshoe ring radius, prong length and ring half-thickness.
const HS_RADIUS: f32 = 0.7;
const HS_ARM: f32 = 0.5;
const HS_THICK: f32 = 0.2;
/// Default rounded-tunnel half-width and arch height.
const TUNNEL_HALF_WIDTH: f32 = 0.5;
const TUNNEL_HEIGHT: f32 = 0.8;

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

/// Squared length of a 2-vector, matching the reference `dot2_2`.
fn dot2_2(v: [f32; 2]) -> f32 {
    v[0] * v[0] + v[1] * v[1]
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
/// `ray_scene::sdf_primitives::pie`: a circular sector centred on `+y`, the arc
/// circle clamped against the straight edge ray and signed by the wedge side.
fn pie_host(point: [f32; 2], sc: [f32; 2], radius: f32) -> f32 {
    let px = point[0].abs();
    let p = [px, point[1]];
    let l = length2(p) - radius;
    let t = (p[0] * sc[0] + p[1] * sc[1]).clamp(0.0, radius);
    let m = length2([p[0] - sc[0] * t, p[1] - sc[1] * t]);
    let edge_sign = sign_of(sc[1] * p[0] - sc[0] * p[1]);
    l.max(m * edge_sign)
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::cut_disk_2d`: a disk sliced by a horizontal
/// chord, with arc, chord and corner regions each carrying their own distance.
fn cut_disk_2d_host(point: [f32; 2], radius: f32, cut_height: f32) -> f32 {
    let r = radius;
    let h = -cut_height;
    let w = (r * r - h * h).max(0.0).sqrt();
    let p = [point[0].abs(), -point[1]];
    let s = ((h - r) * p[0] * p[0] + w * w * (h + r - 2.0 * p[1])).max(h * p[0] - w * p[1]);
    if s < 0.0 {
        length2(p) - r
    } else if p[0] < w {
        h - p[1]
    } else {
        length2([p[0] - w, p[1] - h])
    }
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::horseshoe_2d`: fold by `|x|`, rotate into the
/// arc frame, and measure a half-infinite rounded strip.
fn horseshoe_2d_host(point: [f32; 2], sc: [f32; 2], radius: f32, arm: f32, thickness: f32) -> f32 {
    let c_cos = sc[1];
    let c_sin = sc[0];
    let px = point[0].abs();
    let l = length2([px, point[1]]);
    let rx = -c_cos * px + c_sin * point[1];
    let ry = c_sin * px + c_cos * point[1];
    let qx = if ry > 0.0 || rx > 0.0 {
        rx
    } else {
        l * sign_of(-c_cos)
    };
    let qy = if rx > 0.0 { ry } else { l };
    let bx = qx - arm;
    let by = (qy - radius).abs() - thickness;
    length2([bx.max(0.0), by.max(0.0)]) + bx.max(by).min(0.0)
}

/// Independent reimplementation of
/// `ray_scene::sdf_primitives::tunnel_2d`: a rounded arch, a box half-space
/// combined with the capping semicircle.
fn tunnel_2d_host(point: [f32; 2], half_width: f32, height: f32) -> f32 {
    let px = point[0].abs();
    let py = -point[1];
    let mut qx = px - half_width;
    let qy = py - height;
    let d1 = dot2_2([qx.max(0.0), qy]);
    qx = if py > 0.0 {
        qx
    } else {
        length2([px, py]) - half_width
    };
    let d2 = dot2_2([qx, qy.max(0.0)]);
    let d = d1.min(d2).sqrt();
    if qx.max(qy) < 0.0 {
        -d
    } else {
        d
    }
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

/// Builds one query at `point` with the default per-shape parameters so every
/// shape is exercised on every dispatch.
fn q_at(point: [f32; 2]) -> SdfArc2dQuery {
    SdfArc2dQuery {
        point,
        pie_sin: PIE_SIN,
        pie_cos: PIE_COS,
        pie_radius: PIE_RADIUS,
        cut_radius: CUT_RADIUS,
        cut_height: CUT_HEIGHT,
        hs_sin: HS_SIN,
        hs_cos: HS_COS,
        hs_radius: HS_RADIUS,
        hs_arm: HS_ARM,
        hs_thickness: HS_THICK,
        tunnel_half_width: TUNNEL_HALF_WIDTH,
        tunnel_height: TUNNEL_HEIGHT,
    }
}

/// Dispatches `queries` and asserts every distance matches the host oracles.
fn check_batch(ctx: &GpuContext, gpu: &GpuSdfArc2d, queries: &[SdfArc2dQuery]) {
    let got: Vec<SdfArc2dResult> = gpu.evaluate(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden_pie = pie_host(q.point, [q.pie_sin, q.pie_cos], q.pie_radius);
        let golden_cut = cut_disk_2d_host(q.point, q.cut_radius, q.cut_height);
        let golden_hs = horseshoe_2d_host(
            q.point,
            [q.hs_sin, q.hs_cos],
            q.hs_radius,
            q.hs_arm,
            q.hs_thickness,
        );
        let golden_tunnel = tunnel_2d_host(q.point, q.tunnel_half_width, q.tunnel_height);
        assert!(
            close(r.pie, golden_pie),
            "pie mismatch: gpu={} golden={} (point={:?})",
            r.pie,
            golden_pie,
            q.point
        );
        assert!(
            close(r.cut_disk, golden_cut),
            "cut_disk mismatch: gpu={} golden={} (point={:?} radius={} cut_height={})",
            r.cut_disk,
            golden_cut,
            q.point,
            q.cut_radius,
            q.cut_height
        );
        assert!(
            close(r.horseshoe, golden_hs),
            "horseshoe mismatch: gpu={} golden={} (point={:?} radius={} arm={} thickness={})",
            r.horseshoe,
            golden_hs,
            q.point,
            q.hs_radius,
            q.hs_arm,
            q.hs_thickness
        );
        assert!(
            close(r.tunnel, golden_tunnel),
            "tunnel mismatch: gpu={} golden={} (point={:?} half_width={} height={})",
            r.tunnel,
            golden_tunnel,
            q.point,
            q.tunnel_half_width,
            q.tunnel_height
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
        eprintln!("skipping sdf_arc2d parity: no wgpu adapter available");
        return;
    };
    let gpu = GpuSdfArc2d::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "empty batch yields no results");
}

#[test]
fn pie_interior_and_exterior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfArc2d::new(&ctx);
    // Points inside the sector (negative) and a far exterior point (positive).
    let queries = [
        q_at([0.1, 0.5]),
        q_at([0.0, 0.3]),
        q_at([2.0, 2.0]),
        q_at([-1.6, -1.4]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn pie_edge_and_arc_branches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfArc2d::new(&ctx);
    // Points near the straight edge ray and beyond the arc radius exercise the
    // edge-distance and arc-distance halves on both wedge sides.
    let queries = [
        q_at([0.9, 0.2]),
        q_at([-0.9, 0.2]),
        q_at([0.4, 1.3]),
        q_at([0.2, -0.9]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn cut_disk_arc_chord_and_corner_regions() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfArc2d::new(&ctx);
    // Arc region (s < 0, deep inside), chord region (flat top), and corner
    // region (past the chord endpoint).
    let queries = [
        q_at([0.0, -0.4]),
        q_at([0.0, 0.6]),
        q_at([1.2, 0.5]),
        q_at([-0.3, 0.9]),
        q_at([0.95, 0.35]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn horseshoe_bend_and_prong_branches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfArc2d::new(&ctx);
    // Points inside the ring bend (fallback to radial magnitude) and along the
    // prongs (rotated-frame branch), plus mirrored pairs.
    let queries = [
        q_at([0.0, 0.7]),
        q_at([0.5, 0.5]),
        q_at([-0.5, 0.5]),
        q_at([0.7, -0.1]),
        q_at([0.2, -0.6]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn tunnel_wall_arch_and_interior() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfArc2d::new(&ctx);
    // Interior (negative), side wall, and the rounded cap above the arch.
    let queries = [
        q_at([0.0, 0.0]),
        q_at([0.0, -0.4]),
        q_at([0.9, 0.1]),
        q_at([0.2, 1.2]),
        q_at([-0.8, 0.5]),
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn mixed_single_dispatch_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfArc2d::new(&ctx);
    // A heterogeneous batch with varied per-shape parameters in one dispatch.
    let queries = [
        SdfArc2dQuery {
            point: [0.3, 0.5],
            pie_sin: PIE_SIN,
            pie_cos: PIE_COS,
            pie_radius: 1.2,
            cut_radius: 1.1,
            cut_height: 0.2,
            hs_sin: HS_SIN,
            hs_cos: HS_COS,
            hs_radius: 0.8,
            hs_arm: 0.6,
            hs_thickness: 0.25,
            tunnel_half_width: 0.6,
            tunnel_height: 1.0,
        },
        SdfArc2dQuery {
            point: [-0.8, 0.9],
            pie_sin: PIE_SIN,
            pie_cos: PIE_COS,
            pie_radius: 0.9,
            cut_radius: 1.3,
            cut_height: -0.3,
            hs_sin: HS_SIN,
            hs_cos: HS_COS,
            hs_radius: 1.0,
            hs_arm: 0.4,
            hs_thickness: 0.15,
            tunnel_half_width: 0.4,
            tunnel_height: 0.7,
        },
        SdfArc2dQuery {
            point: [1.4, -1.1],
            pie_sin: PIE_SIN,
            pie_cos: PIE_COS,
            pie_radius: 1.5,
            cut_radius: 0.8,
            cut_height: 0.1,
            hs_sin: HS_SIN,
            hs_cos: HS_COS,
            hs_radius: 1.1,
            hs_arm: 0.9,
            hs_thickness: 0.3,
            tunnel_half_width: 0.9,
            tunnel_height: 1.4,
        },
    ];
    check_batch(&ctx, &gpu, &queries);
}

#[test]
fn randomized_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSdfArc2d::new(&ctx);
    let mut state: u64 = 0x51a3_7cd9_2e04_b6f1;
    // Margin kept away from each shape's branch/fold/sign locus. Those loci are
    // continuous (the branches agree there), so this is extra safety only.
    let margin = 0.03_f32;
    let mut queries = Vec::with_capacity(384);
    while queries.len() < 384 {
        let point = [draw(&mut state, -1.8, 1.8), draw(&mut state, -1.8, 1.8)];

        // Pie: positive sector radius.
        let pie_radius = draw(&mut state, 0.6, 1.5);
        // Cut disk: -radius < cut_height < radius with margin.
        let cut_radius = draw(&mut state, 0.6, 1.5);
        let cut_height = draw(&mut state, -cut_radius + margin, cut_radius - margin);
        // Horseshoe: thickness < radius, positive prong.
        let hs_radius = draw(&mut state, 0.5, 1.2);
        let hs_thickness = draw(&mut state, 0.1, hs_radius - 0.1);
        let hs_arm = draw(&mut state, 0.3, 1.0);
        // Tunnel: positive half-width and height.
        let tunnel_half_width = draw(&mut state, 0.3, 1.0);
        let tunnel_height = draw(&mut state, 0.3, 1.5);

        let px = point[0].abs();
        let py = point[1];

        // Pie edge-sign locus.
        if (PIE_COS * px - PIE_SIN * py).abs() < margin {
            continue;
        }
        // Cut-disk region seams: s = 0 and the chord endpoint p.x = w.
        let r = cut_radius;
        let h = -cut_height;
        let w = (r * r - h * h).max(0.0).sqrt();
        let cp = [px, -py];
        let s =
            ((h - r) * cp[0] * cp[0] + w * w * (h + r - 2.0 * cp[1])).max(h * cp[0] - w * cp[1]);
        if s.abs() < margin || (cp[0] - w).abs() < margin {
            continue;
        }
        // Horseshoe rotated-frame axes.
        let rx = -HS_COS * px + HS_SIN * py;
        let ry = HS_SIN * px + HS_COS * py;
        if rx.abs() < margin || ry.abs() < margin {
            continue;
        }
        // Tunnel wall/cap seam (py = 0 predicate) and sign flip.
        let tpy = -py;
        let tqx = px - tunnel_half_width;
        let tqy = tpy - tunnel_height;
        let tqx2 = if tpy > 0.0 {
            tqx
        } else {
            length2([px, tpy]) - tunnel_half_width
        };
        if tpy.abs() < margin || tqx2.max(tqy).abs() < margin {
            continue;
        }

        queries.push(SdfArc2dQuery {
            point,
            pie_sin: PIE_SIN,
            pie_cos: PIE_COS,
            pie_radius,
            cut_radius,
            cut_height,
            hs_sin: HS_SIN,
            hs_cos: HS_COS,
            hs_radius,
            hs_arm,
            hs_thickness,
            tunnel_half_width,
            tunnel_height,
        });
    }
    check_batch(&ctx, &gpu, &queries);
}
