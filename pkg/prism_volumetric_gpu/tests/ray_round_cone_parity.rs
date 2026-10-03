//! Real-device parity for the analytic ray/round-cone twin:
//! [`GpuRayRoundCone`](prism_volumetric_gpu::ray_round_cone::GpuRayRoundCone)
//! must reproduce the closed-form ray/round-cone intersection across the three
//! analytic pieces — the tangent cone band and the two clipped end spheres —
//! and across the degenerate single-sphere (engulf) case.
//!
//! # Oracle
//!
//! This suite is deliberately self-contained: it does **not** depend on the
//! golden crate. The reference `intersect` closed form is re-implemented here
//! (`intersect_oracle`) directly from its published mathematics so the twin is
//! validated against an independent port rather than a shared binary. The port
//! uses the same guard magnitudes the kernel uses (`COEFF_EPS` for a quadratic
//! leading coefficient treated as zero, `RR_EPS` for the signed radius
//! difference treated as zero), so the on-device twin and this oracle take the
//! same branch on every query the fixtures admit.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and a
//! few `sqrt`s, so the `CPU` oracle and the `GPU` twin evaluate the same closed
//! form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on the `f32` fields (the hit
//! parameter and the normal channels) while pinning the hit flag and the
//! `front_face` classification exactly.
//!
//! # Conditioning
//!
//! Every random fixture is screened by `well_conditioned`, which re-evaluates
//! the oracle under many small input perturbations and rejects any query whose
//! hit flag, `front_face` sign, chosen `t` or normal would move across a branch
//! boundary. That keeps the fixtures clear of the razor-thin critical bands —
//! a near-zero discriminant, a sphere/cone switch, a near-tie between pieces,
//! a grazing `front_face` sign, a root pinned to an interval endpoint — where a
//! few units in the last place of `GPU`/`CPU` slack could flip a decision. The
//! named fixtures use clean axis-aligned geometry hand-checked to land well
//! inside a single piece.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::round_cone`；
//! 无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::ray_round_cone::{
    GpuRayRoundCone, RayRoundConeQuery, RayRoundConeResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound. A `GPU` may fuse a multiply-add the scalar reference
/// leaves separate, perturbing the low mantissa bits by a few units in the last
/// place; `1e-4` admits that legal slack while still failing a wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Guard magnitude for a quadratic leading coefficient treated as zero. It
/// matches the kernel's `COEFF_EPS` so both take the same no-root branch.
const COEFF_EPS: f32 = 1.0e-20;

/// Guard magnitude for the signed radius difference treated as zero (the
/// capsule/cylinder construction). It matches the kernel's `RR_EPS`.
const RR_EPS: f32 = 1.0e-9;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// A looser agreement test used only by the conditioning screen: a perturbation
/// of a well-conditioned query moves `t` and the normal smoothly, so a jump
/// past this margin signals a branch boundary and rejects the candidate.
fn stable(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= 1.5e-2 || rel <= 1.5e-2
}

/// Subtracts `b` from `a` componentwise.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Adds `a` and `b` componentwise.
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scales `a` by scalar `s`.
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Euclidean dot product.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// World hit position `origin + t * direction`.
fn point_at(origin: [f32; 3], direction: [f32; 3], t: f32) -> [f32; 3] {
    add3(origin, scale3(direction, t))
}

/// Solves the sphere quadratic `|rel + t*d|^2 = r2`, returning both roots or
/// `None`. Mirrors the reference `sphere_roots`.
fn sphere_roots(rel: [f32; 3], direction: [f32; 3], dd: f32, r2: f32) -> Option<(f32, f32)> {
    let b = 2.0 * dot3(rel, direction);
    let c = dot3(rel, rel) - r2;
    let disc = b * b - 4.0 * dd * c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_disc = disc.sqrt();
    let inv_2a = 1.0 / (2.0 * dd);
    Some(((-b - sqrt_disc) * inv_2a, (-b + sqrt_disc) * inv_2a))
}

/// The running best `(t, outward-normal)` across every analytic piece.
#[derive(Clone, Copy)]
struct Best {
    t: f32,
    outward: [f32; 3],
}

/// The fully-decoded query geometry shared by every piece of the oracle.
struct Scene {
    origin: [f32; 3],
    direction: [f32; 3],
    dd: f32,
    a: [f32; 3],
    b: [f32; 3],
    radius_a: f32,
    radius_b: f32,
    t_min: f32,
    t_max: f32,
}

impl Scene {
    /// Keeps the nearest in-interval root, mirroring the reference `consider`.
    fn consider(&self, best: &mut Best, t: f32, outward: [f32; 3]) {
        if !(t >= self.t_min && t <= self.t_max) || t >= best.t {
            return;
        }
        best.t = t;
        best.outward = outward;
    }

    /// Emits both roots of a sphere of radius `radius` at `center`, with the
    /// outward normal `normalize(p - center)`. Mirrors `emit_sphere`.
    fn emit_sphere(&self, best: &mut Best, center: [f32; 3], radius: f32) {
        let rel = sub3(self.origin, center);
        if let Some((t0, t1)) = sphere_roots(rel, self.direction, self.dd, radius * radius) {
            for t in [t0, t1] {
                let p = point_at(self.origin, self.direction, t);
                let radial = sub3(p, center);
                let len2 = dot3(radial, radial);
                if len2 > 0.0 {
                    self.consider(best, t, scale3(radial, 1.0 / len2.sqrt()));
                }
            }
        }
    }

    /// Equal-radius lateral band: a cylinder of radius `r` clipped to the axial
    /// range `[0, l]`, plus hemispheres clipped at the band ends. Mirrors
    /// `intersect_cylinder`.
    fn cylinder(&self, best: &mut Best, n: [f32; 3], l: f32, r: f32) {
        let r2 = r * r;
        let oa = sub3(self.origin, self.a);
        let za = dot3(oa, n);
        let zd = dot3(self.direction, n);
        let ad = dot3(oa, self.direction);
        let aa = dot3(oa, oa);

        let coeff_a = self.dd - zd * zd;
        let coeff_b = 2.0 * (ad - za * zd);
        let coeff_c = aa - za * za - r2;
        if coeff_a.abs() > COEFF_EPS {
            let disc = coeff_b * coeff_b - 4.0 * coeff_a * coeff_c;
            if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let inv_2a = 1.0 / (2.0 * coeff_a);
                for t in [
                    (-coeff_b - sqrt_disc) * inv_2a,
                    (-coeff_b + sqrt_disc) * inv_2a,
                ] {
                    let z = za + t * zd;
                    if z >= 0.0 && z <= l {
                        let axis_point = add3(self.a, scale3(n, z));
                        let p = point_at(self.origin, self.direction, t);
                        let radial = sub3(p, axis_point);
                        let len2 = dot3(radial, radial);
                        if len2 > 0.0 {
                            self.consider(best, t, scale3(radial, 1.0 / len2.sqrt()));
                        }
                    }
                }
            }
        }

        // Hemisphere at `a` (band start, axial <= 0).
        if let Some((t0, t1)) = sphere_roots(oa, self.direction, self.dd, r2) {
            for t in [t0, t1] {
                let p = point_at(self.origin, self.direction, t);
                if dot3(sub3(p, self.a), n) <= 0.0 {
                    let radial = sub3(p, self.a);
                    let len2 = dot3(radial, radial);
                    if len2 > 0.0 {
                        self.consider(best, t, scale3(radial, 1.0 / len2.sqrt()));
                    }
                }
            }
        }

        // Hemisphere at `b` (band end, axial >= l). The clip uses `p - a`.
        let ob = sub3(self.origin, self.b);
        if let Some((t0, t1)) = sphere_roots(ob, self.direction, self.dd, r2) {
            for t in [t0, t1] {
                let p = point_at(self.origin, self.direction, t);
                if dot3(sub3(p, self.a), n) >= l {
                    let radial = sub3(p, self.b);
                    let len2 = dot3(radial, radial);
                    if len2 > 0.0 {
                        self.consider(best, t, scale3(radial, 1.0 / len2.sqrt()));
                    }
                }
            }
        }
    }

    /// Unequal-radius lateral band: the external tangent cone clipped to the
    /// axial band between the tangent circles, plus the end spheres clipped to
    /// the caps those circles leave exposed. Mirrors `intersect_tapered`.
    fn tapered(&self, best: &mut Best, n: [f32; 3], l: f32, rr: f32) {
        let sin_a = rr / l;
        let cos_a = (1.0 - sin_a * sin_a).sqrt();
        let band_lo = self.radius_a * sin_a;
        let band_hi = l + self.radius_b * sin_a;

        let oa = sub3(self.origin, self.a);
        let za = dot3(oa, n);
        let zd = dot3(self.direction, n);
        let ad = dot3(oa, self.direction);
        let aa = dot3(oa, oa);
        let k = cos_a * cos_a;
        let rhs0 = self.radius_a - sin_a * za;
        let rhs1 = -sin_a * zd;
        let coeff_a = k * (self.dd - zd * zd) - rhs1 * rhs1;
        let coeff_b = k * (ad - za * zd) - rhs0 * rhs1;
        let coeff_c = k * (aa - za * za) - rhs0 * rhs0;
        if coeff_a.abs() > COEFF_EPS {
            // `coeff_b` is the half-b form, so the discriminant drops the 4.
            let disc = coeff_b * coeff_b - coeff_a * coeff_c;
            if disc >= 0.0 {
                let sqrt_disc = disc.sqrt();
                let inv_a = 1.0 / coeff_a;
                for t in [
                    (-coeff_b - sqrt_disc) * inv_a,
                    (-coeff_b + sqrt_disc) * inv_a,
                ] {
                    let axial = za + t * zd;
                    if !(axial >= band_lo && axial <= band_hi) {
                        continue;
                    }
                    // Keep only the physical nappe.
                    if rhs0 + t * rhs1 < 0.0 {
                        continue;
                    }
                    let p = point_at(self.origin, self.direction, t);
                    let radial = sub3(sub3(p, self.a), scale3(n, axial));
                    let len2 = dot3(radial, radial);
                    if len2 <= 0.0 {
                        continue;
                    }
                    let radial_u = scale3(radial, 1.0 / len2.sqrt());
                    let outward = add3(scale3(radial_u, cos_a), scale3(n, sin_a));
                    self.consider(best, t, outward);
                }
            }
        }

        // Sphere at `a`, clipped to the cap below the tangent circle.
        if let Some((t0, t1)) =
            sphere_roots(oa, self.direction, self.dd, self.radius_a * self.radius_a)
        {
            for t in [t0, t1] {
                let p = point_at(self.origin, self.direction, t);
                if dot3(sub3(p, self.a), n) <= band_lo {
                    let radial = sub3(p, self.a);
                    let len2 = dot3(radial, radial);
                    if len2 > 0.0 {
                        self.consider(best, t, scale3(radial, 1.0 / len2.sqrt()));
                    }
                }
            }
        }

        // Sphere at `b`, clipped to the cap above the tangent circle.
        let ob = sub3(self.origin, self.b);
        if let Some((t0, t1)) =
            sphere_roots(ob, self.direction, self.dd, self.radius_b * self.radius_b)
        {
            for t in [t0, t1] {
                let p = point_at(self.origin, self.direction, t);
                if dot3(sub3(p, self.b), n) >= self.radius_b * sin_a {
                    let radial = sub3(p, self.b);
                    let len2 = dot3(radial, radial);
                    if len2 > 0.0 {
                        self.consider(best, t, scale3(radial, 1.0 / len2.sqrt()));
                    }
                }
            }
        }
    }
}

/// Independent closed-form oracle for one ray/round-cone query, ported from the
/// reference `RoundCone::intersect` with the kernel's guard magnitudes.
fn intersect_oracle(q: &RayRoundConeQuery) -> RayRoundConeResult {
    let miss = RayRoundConeResult {
        hit: 0,
        t: 0.0,
        normal: [0.0, 0.0, 0.0],
        front_face: 0,
    };

    let direction = q.direction;
    let dd = dot3(direction, direction);
    if dd <= 0.0 {
        return miss;
    }
    // Fold both radii to their magnitude, mirroring `RoundCone::new`.
    let radius_a = q.radius_a.abs();
    let radius_b = q.radius_b.abs();
    if radius_a <= 0.0 && radius_b <= 0.0 {
        return miss;
    }

    let scene = Scene {
        origin: q.origin,
        direction,
        dd,
        a: q.a,
        b: q.b,
        radius_a,
        radius_b,
        t_min: q.t_min,
        t_max: q.t_max,
    };

    let w = sub3(scene.b, scene.a);
    let l2 = dot3(w, w);
    let rr = radius_a - radius_b;

    let mut best = Best {
        t: f32::INFINITY,
        outward: [0.0, 0.0, 0.0],
    };

    if l2 <= rr * rr {
        // One sphere engulfs the other (or the axis is zero): larger sphere.
        let (center, radius) = if radius_a >= radius_b {
            (scene.a, radius_a)
        } else {
            (scene.b, radius_b)
        };
        scene.emit_sphere(&mut best, center, radius);
    } else {
        let inv_l = 1.0 / l2.sqrt();
        let n = scale3(w, inv_l);
        let l = l2 * inv_l;
        if rr.abs() <= RR_EPS {
            scene.cylinder(&mut best, n, l, radius_a);
        } else {
            scene.tapered(&mut best, n, l, rr);
        }
    }

    if !best.t.is_finite() {
        return miss;
    }
    let front = dot3(direction, best.outward) < 0.0;
    let normal = if front {
        best.outward
    } else {
        scale3(best.outward, -1.0)
    };
    RayRoundConeResult {
        hit: 1,
        t: best.t,
        normal,
        front_face: u32::from(front),
    }
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random value in `[lo, hi)` drawn from `state`.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * lcg(state)
}

/// A pseudo-random value in `[-span, span)`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> [f32; 3] {
    [
        signed(state, span),
        signed(state, span),
        signed(state, span),
    ]
}

/// A unit vector drawn from `state`, retried until comfortably non-zero.
fn rand_unit(state: &mut u64) -> [f32; 3] {
    loop {
        let v = rand_vec(state, 1.0);
        let len2 = dot3(v, v);
        if len2 > 0.2 {
            return scale3(v, 1.0 / len2.sqrt());
        }
    }
}

/// Builds a random, clearly separated round cone and a ray aimed roughly at its
/// surface. The axis length dominates the radius difference, so the engulf case
/// is never reached; radii are either exactly equal (the cylinder branch) or
/// clearly unequal (the tapered branch).
fn random_query(state: &mut u64) -> RayRoundConeQuery {
    let axis = rand_unit(state);
    let length = uniform(state, 2.0, 4.0);
    let a = rand_vec(state, 2.0);
    let b = add3(a, scale3(axis, length));
    let radius_a = uniform(state, 0.3, 1.2);
    let radius_b = if lcg(state) < 0.5 {
        radius_a
    } else {
        loop {
            let rb = uniform(state, 0.3, 1.2);
            if (rb - radius_a).abs() > 0.1 {
                break rb;
            }
        }
    };

    let mid = add3(a, scale3(axis, length * 0.5));
    let out = rand_unit(state);
    let dist = uniform(state, 4.0, 8.0);
    let origin = add3(mid, scale3(out, dist));
    let s = lcg(state);
    let on_axis = add3(a, scale3(axis, length * s));
    let target = add3(on_axis, rand_vec(state, radius_a * 1.3));
    let direction = sub3(target, origin);

    RayRoundConeQuery {
        origin,
        direction,
        t_min: 0.0,
        t_max: 100.0,
        a,
        b,
        radius_a,
        radius_b,
    }
}

/// Returns `q` with every input scalar nudged by an independent `+/- delta`.
fn jittered(q: &RayRoundConeQuery, state: &mut u64, delta: f32) -> RayRoundConeQuery {
    let mut c = *q;
    for v in &mut c.origin {
        *v += signed(state, delta);
    }
    for v in &mut c.direction {
        *v += signed(state, delta);
    }
    c.a[0] += signed(state, delta);
    c.a[1] += signed(state, delta);
    c.a[2] += signed(state, delta);
    c.b[0] += signed(state, delta);
    c.b[1] += signed(state, delta);
    c.b[2] += signed(state, delta);
    c.radius_a += signed(state, delta);
    c.radius_b += signed(state, delta);
    c
}

/// Screens a candidate: re-evaluates the oracle under many small perturbations
/// and accepts only when the hit flag, `front_face`, chosen `t` and normal all
/// stay on the same side of every branch. Rejects near-critical fixtures.
fn well_conditioned(q: &RayRoundConeQuery, state: &mut u64) -> bool {
    let base = intersect_oracle(q);
    if base.hit == 1 && !(base.t > q.t_min + 1.0e-2 && base.t < q.t_max - 1.0e-2) {
        return false;
    }
    for _ in 0..24 {
        let probe = jittered(q, state, 1.5e-3);
        let r = intersect_oracle(&probe);
        if r.hit != base.hit {
            return false;
        }
        if base.hit == 1 {
            if r.front_face != base.front_face {
                return false;
            }
            if !stable(r.t, base.t) {
                return false;
            }
            let normals_stable = stable(r.normal[0], base.normal[0])
                && stable(r.normal[1], base.normal[1])
                && stable(r.normal[2], base.normal[2]);
            if !normals_stable {
                return false;
            }
        }
    }
    true
}

/// Pins one `GPU` result against the independent oracle for query `idx`: the
/// hit flag and `front_face` exactly, the hit parameter and normal within bound.
fn check_one(idx: usize, got: &RayRoundConeResult, want: &RayRoundConeResult) {
    assert_eq!(
        got.hit, want.hit,
        "query {idx}: hit flag must match the oracle"
    );
    assert_eq!(
        got.front_face, want.front_face,
        "query {idx}: front_face must match the oracle"
    );
    if want.hit == 1 {
        assert!(
            close(got.t, want.t),
            "query {idx} t: gpu {} vs cpu {}",
            got.t,
            want.t
        );
        for (axis, (g, w)) in got.normal.iter().zip(want.normal.iter()).enumerate() {
            assert!(
                close(*g, *w),
                "query {idx} normal[{axis}]: gpu {g} vs cpu {w}"
            );
        }
    }
}

/// Dispatches `queries` on the `GPU` and pins every result against the oracle.
fn check(ctx: &GpuContext, gpu: &GpuRayRoundCone, queries: &[RayRoundConeQuery]) {
    let got = gpu.evaluate(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        check_one(idx, result, &intersect_oracle(query));
    }
}

/// A canonical unit-radius capsule (equal radii) along +X from `a` to `b`.
fn unit_capsule(origin: [f32; 3], direction: [f32; 3]) -> RayRoundConeQuery {
    RayRoundConeQuery {
        origin,
        direction,
        t_min: 0.0,
        t_max: 100.0,
        a: [0.0, 0.0, 0.0],
        b: [4.0, 0.0, 0.0],
        radius_a: 1.0,
        radius_b: 1.0,
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer
    // cannot be zero-sized) and returns an empty vector.
    let got = gpu.evaluate(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn side_flank_hit_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // Equal-radius capsule flank: a ray down -y strikes the cylindrical wall at
    // y = 1, axial x = 2 (clearly inside [0, 4]); outward normal +y.
    let query = unit_capsule([2.0, 5.0, 0.0], [0.0, -1.0, 0.0]);
    assert_eq!(intersect_oracle(&query).hit, 1, "flank fixture should hit");
    check(&ctx, &gpu, &[query]);
}

#[test]
fn near_cap_hit_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // A ray down +x enters the near end sphere at x = -1 (axial <= 0); outward
    // normal -x.
    let query = unit_capsule([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    assert_eq!(
        intersect_oracle(&query).hit,
        1,
        "near-cap fixture should hit"
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn far_cap_hit_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // A ray down -x strikes the far end sphere at x = 5 (axial >= l = 4);
    // outward normal +x.
    let query = unit_capsule([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
    assert_eq!(
        intersect_oracle(&query).hit,
        1,
        "far-cap fixture should hit"
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn tapered_flank_hit_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // Unequal radii (1.0 -> 0.5): a ray down -y strikes the tangent cone band
    // at axial x = 2, well inside the band and on the physical nappe.
    let query = RayRoundConeQuery {
        origin: [2.0, 5.0, 0.0],
        direction: [0.0, -1.0, 0.0],
        t_min: 0.0,
        t_max: 100.0,
        a: [0.0, 0.0, 0.0],
        b: [4.0, 0.0, 0.0],
        radius_a: 1.0,
        radius_b: 0.5,
    };
    assert_eq!(
        intersect_oracle(&query).hit,
        1,
        "tapered fixture should hit"
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn engulf_reduces_to_larger_sphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // The axis is shorter than the radius difference, so the b-sphere is
    // swallowed and the solid is the radius-1.5 sphere at `a`. A ray down -x
    // strikes it at x = 1.5; outward normal +x.
    let query = RayRoundConeQuery {
        origin: [5.0, 0.0, 0.0],
        direction: [-1.0, 0.0, 0.0],
        t_min: 0.0,
        t_max: 100.0,
        a: [0.0, 0.0, 0.0],
        b: [0.3, 0.0, 0.0],
        radius_a: 1.5,
        radius_b: 0.2,
    };
    assert_eq!(intersect_oracle(&query).hit, 1, "engulf fixture should hit");
    check(&ctx, &gpu, &[query]);
}

#[test]
fn zero_axis_reduces_to_single_sphere() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // A zero-length axis reduces to the larger end sphere (radius 1.0 at `a`).
    let query = RayRoundConeQuery {
        origin: [5.0, 0.0, 0.0],
        direction: [-1.0, 0.0, 0.0],
        t_min: 0.0,
        t_max: 100.0,
        a: [0.0, 0.0, 0.0],
        b: [0.0, 0.0, 0.0],
        radius_a: 1.0,
        radius_b: 0.5,
    };
    assert_eq!(
        intersect_oracle(&query).hit,
        1,
        "zero-axis fixture should hit"
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn out_of_interval_misses() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // The flank hit lands at t = 4, but the ray interval ends at t_max = 1, so
    // the whole batch misses.
    let mut query = unit_capsule([2.0, 5.0, 0.0], [0.0, -1.0, 0.0]);
    query.t_max = 1.0;
    assert_eq!(
        intersect_oracle(&query).hit,
        0,
        "clipped fixture should miss"
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn clear_miss_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // A ray fired along +z, offset far off the axis in y, clears every piece.
    let query = unit_capsule([2.0, 5.0, 0.0], [0.0, 0.0, 1.0]);
    assert_eq!(intersect_oracle(&query).hit, 0, "miss fixture should miss");
    check(&ctx, &gpu, &[query]);
}

#[test]
fn fixture_batch_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // Every named fixture, dispatched together so the per-thread indexing and
    // the contiguous storage layout are both exercised, then pinned one by one.
    let mut tapered = unit_capsule([2.0, 5.0, 0.0], [0.0, -1.0, 0.0]);
    tapered.radius_b = 0.5;
    let mut clipped = unit_capsule([2.0, 5.0, 0.0], [0.0, -1.0, 0.0]);
    clipped.t_max = 1.0;
    let queries = vec![
        unit_capsule([2.0, 5.0, 0.0], [0.0, -1.0, 0.0]),
        unit_capsule([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        unit_capsule([10.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        tapered,
        clipped,
        unit_capsule([2.0, 5.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_sweep_matches_oracle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRayRoundCone::new(&ctx);
    // A large sweep (several workgroups' worth) of well-conditioned random
    // round cones and rays, pinned element-for-element against the oracle.
    let mut state = 0x0f0e_0d0c_0b0a_0908_u64;
    let mut queries: Vec<RayRoundConeQuery> = Vec::new();
    let mut attempts = 0u32;
    while queries.len() < 512 && attempts < 40_000 {
        attempts += 1;
        let candidate = random_query(&mut state);
        if well_conditioned(&candidate, &mut state) {
            queries.push(candidate);
        }
    }
    assert!(
        queries.len() >= 256,
        "expected a healthy sweep, built {}",
        queries.len()
    );
    check(&ctx, &gpu, &queries);
}
