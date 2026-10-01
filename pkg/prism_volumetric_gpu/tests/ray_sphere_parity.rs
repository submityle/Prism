//! Real-device parity for the analytic ray-sphere twin:
//! [`GpuRaySphere`](prism_volumetric_gpu::ray_sphere::GpuRaySphere) must
//! reproduce the `CPU` golden
//! [`ray_sphere`](prism_render_architecture::particle::ray_sphere) across clear
//! hits (a well-separated two-root crossing), clear misses (a line that never
//! reaches the sphere), origin-inside probes (only the far wall is a forward
//! hit) and behind-origin probes (both roots negative, so the unsigned
//! `intersects` still sees the crossing while `first_hit` reports nothing), plus
//! a randomized batch of all four categories compared element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each probe is a fixed, non-reorderable sequence of multiplies, adds and one
//! `sqrt`, so `CPU` and `GPU` evaluate the same closed form in the same order.
//! They are not bit-exact: a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on the `f32` fields while pinning the integer / boolean
//! classification (`solved`, `is_tangent`, `intersects`, hit presence) exactly.
//!
//! # Conditioning
//!
//! Every fixture is deliberately well away from the degenerate regions and the
//! classification boundaries: no tangent graze (the discriminant is always
//! clearly non-zero), no origin exactly on the surface, no near-zero radius and
//! no non-unit direction. Hit impact parameters stay below half the radius so
//! the two roots are well separated and the outward normal has unit length (far
//! from the zero vector), and misses clear the sphere by more than its radius so
//! the discriminant is clearly negative. This keeps `CPU` and `GPU` on the same
//! side of every branch regardless of a few units in the last place of slack.
//!
//! Provenance: twinned from this repository's
//! [`ray_sphere`](prism_render_architecture::particle::ray_sphere); no
//! third-party engine source or derived code.

use prism_render_architecture::particle::ray_sphere::{Ray, Sphere, Vec3};
use prism_volumetric_gpu::ray_sphere::{GpuRaySphere, RaySphereProbe, RaySphereResult};
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

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Asserts two vectors agree channel-for-channel within the parity bound.
fn close_vec(label: &str, idx: usize, got: Vec3, want: Vec3) {
    assert!(
        close(got.x, want.x) && close(got.y, want.y) && close(got.z, want.z),
        "probe {idx} {label}: gpu ({}, {}, {}) vs cpu ({}, {}, {})",
        got.x,
        got.y,
        got.z,
        want.x,
        want.y,
        want.z
    );
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

/// A pseudo-random value in `[-span, span)` drawn from `state`.
fn signed(state: &mut u64, span: f32) -> f32 {
    (lcg(state) * 2.0 - 1.0) * span
}

/// A pseudo-random vector with each component in `[-span, span)`.
fn rand_vec(state: &mut u64, span: f32) -> Vec3 {
    Vec3::new(
        signed(state, span),
        signed(state, span),
        signed(state, span),
    )
}

/// A unit vector drawn from `state`, retried until it is comfortably non-zero so
/// the normalization is well conditioned.
fn rand_unit(state: &mut u64) -> Vec3 {
    loop {
        let v = rand_vec(state, 1.0);
        if v.length_squared() > 0.2 {
            return v.normalize_or_zero();
        }
    }
}

/// Returns a unit vector perpendicular to `axis`, built by crossing `axis` with
/// whichever cardinal axis is least parallel to it.
fn perp_unit(axis: Vec3) -> Vec3 {
    let helper = if axis.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    axis.cross(helper).normalize_or_zero()
}

/// Builds a clearly-intersecting probe: the ray starts well outside the sphere
/// and aims near its center with an impact parameter below half the radius, so
/// the discriminant is clearly positive and the two roots are well separated.
fn hit_probe(state: &mut u64) -> RaySphereProbe {
    let center = rand_vec(state, 6.0);
    let radius = 1.0 + lcg(state) * 2.0;
    let away = rand_unit(state);
    let dist = radius + 4.0 + lcg(state) * 4.0;
    let origin = center.plus(away.scale(dist));
    let offset = signed(state, radius * 0.5);
    let target = center.plus(perp_unit(away).scale(offset));
    let dir = target.minus(origin);
    RaySphereProbe::new(
        Ray::new_normalized(origin, dir),
        Sphere::new(center, radius),
    )
}

/// Builds a clear miss: the ray's aim clears the center by more than twice the
/// radius, so the line never reaches the sphere and the discriminant is clearly
/// negative.
fn miss_probe(state: &mut u64) -> RaySphereProbe {
    let center = rand_vec(state, 6.0);
    let radius = 1.0 + lcg(state) * 2.0;
    let away = rand_unit(state);
    let dist = radius + 4.0 + lcg(state) * 4.0;
    let origin = center.plus(away.scale(dist));
    let offset = radius * (2.5 + lcg(state) * 2.0);
    let target = center.plus(perp_unit(away).scale(offset));
    let dir = target.minus(origin);
    RaySphereProbe::new(
        Ray::new_normalized(origin, dir),
        Sphere::new(center, radius),
    )
}

/// Builds an origin-inside probe: the origin sits well within the sphere, so the
/// near root is behind the origin and the far wall is the first forward hit.
fn inside_probe(state: &mut u64) -> RaySphereProbe {
    let center = rand_vec(state, 6.0);
    let radius = 1.5 + lcg(state) * 2.0;
    let inward = rand_unit(state).scale(radius * (lcg(state) * 0.4));
    let origin = center.plus(inward);
    let dir = rand_unit(state);
    RaySphereProbe::new(Ray::new(origin, dir), Sphere::new(center, radius))
}

/// Builds a behind-origin probe: the sphere lies entirely behind the ray, so the
/// line crosses it (unsigned `intersects` is true) but both roots are negative
/// and `first_hit` reports nothing.
fn behind_probe(state: &mut u64) -> RaySphereProbe {
    let origin = rand_vec(state, 6.0);
    let dir = rand_unit(state);
    let radius = 1.0 + lcg(state) * 2.0;
    let back = radius + 4.0 + lcg(state) * 4.0;
    let offset = signed(state, radius * 0.5);
    let center = origin
        .minus(dir.scale(back))
        .plus(perp_unit(dir).scale(offset));
    RaySphereProbe::new(Ray::new(origin, dir), Sphere::new(center, radius))
}

/// Pins one `GPU` result against the `CPU` golden for `probe`: the solve flag,
/// the ordered roots, the tangent and intersect predicates and the nearest
/// forward hit must all agree, booleans exactly and `f32` fields within bound.
fn pin(idx: usize, probe: &RaySphereProbe, got: &RaySphereResult) {
    let want_roots = probe.sphere.solve(probe.ray);
    let want_intersects = probe.sphere.intersects(probe.ray);
    let want_hit = probe.sphere.first_hit(probe.ray);

    assert_eq!(
        got.solved,
        want_roots.is_some(),
        "probe {idx}: solved flag must match the reference"
    );
    assert_eq!(
        got.intersects, want_intersects,
        "probe {idx}: intersects must match the reference"
    );

    if let Some(roots) = want_roots {
        assert_eq!(
            got.is_tangent,
            roots.is_tangent(),
            "probe {idx}: tangent flag must match the reference"
        );
        assert!(
            close(got.roots.t_near, roots.t_near),
            "probe {idx} t_near: gpu {} vs cpu {}",
            got.roots.t_near,
            roots.t_near
        );
        assert!(
            close(got.roots.t_far, roots.t_far),
            "probe {idx} t_far: gpu {} vs cpu {}",
            got.roots.t_far,
            roots.t_far
        );
    }

    match (got.hit, want_hit) {
        (Some(g), Some(w)) => {
            assert!(
                close(g.t, w.t),
                "probe {idx} hit.t: gpu {} vs cpu {}",
                g.t,
                w.t
            );
            close_vec("hit.point", idx, g.point, w.point);
            close_vec("hit.normal", idx, g.normal, w.normal);
        }
        (None, None) => {}
        (g, w) => panic!("probe {idx}: hit presence mismatch gpu {g:?} vs cpu {w:?}"),
    }
}

/// Dispatches `probes` on the `GPU` and pins every result against the reference.
fn check(ctx: &GpuContext, gpu: &GpuRaySphere, probes: &[RaySphereProbe]) {
    let got = gpu.eval(ctx, probes);
    assert_eq!(
        got.len(),
        probes.len(),
        "result count must match the input count"
    );
    for (idx, (probe, result)) in probes.iter().zip(got.iter()).enumerate() {
        pin(idx, probe, result);
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaySphere::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn axis_hit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaySphere::new(&ctx);
    // A ray down +x through a sphere centered at x = 5: a textbook clear hit
    // whose near wall is at x = 4 with an inward-facing outward normal of -x.
    let probe = RaySphereProbe::new(
        Ray::new_normalized(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
        Sphere::new(Vec3::new(5.0, 0.0, 0.0), 1.0),
    );
    check(&ctx, &gpu, &[probe]);
}

#[test]
fn clear_miss_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaySphere::new(&ctx);
    // A ray down +x well below a sphere high on +y: a clear miss (no solve, no
    // intersect, no hit).
    let probe = RaySphereProbe::new(
        Ray::new_normalized(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
        Sphere::new(Vec3::new(0.0, 10.0, 0.0), 1.0),
    );
    check(&ctx, &gpu, &[probe]);
}

#[test]
fn origin_inside_hits_far_wall() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaySphere::new(&ctx);
    // Origin at the center: the near root is behind the origin and the far wall
    // at x = 2 is the first forward hit, with an outward normal of +x.
    let probe = RaySphereProbe::new(
        Ray::new_normalized(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
        Sphere::new(Vec3::ZERO, 2.0),
    );
    check(&ctx, &gpu, &[probe]);
}

#[test]
fn behind_origin_crosses_but_does_not_hit() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaySphere::new(&ctx);
    // Sphere entirely behind the origin: the line crosses it (intersects is
    // unsigned) but both roots are negative so there is no forward hit.
    let probe = RaySphereProbe::new(
        Ray::new_normalized(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
        Sphere::new(Vec3::new(-6.0, 0.0, 0.0), 1.0),
    );
    check(&ctx, &gpu, &[probe]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaySphere::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing all four well-conditioned categories, dispatched together
    // so the per-thread indexing and the contiguous storage layout are both
    // exercised, then pinned element-for-element.
    let mut probes = Vec::new();
    for _ in 0..32 {
        probes.push(hit_probe(&mut state));
        probes.push(miss_probe(&mut state));
        probes.push(inside_probe(&mut state));
        probes.push(behind_probe(&mut state));
    }
    check(&ctx, &gpu, &probes);
}

#[test]
fn many_hits_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuRaySphere::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep of clear hits (several workgroups' worth) pins the roots,
    // hit point and outward normal across many random geometries.
    let probes: Vec<RaySphereProbe> = (0..200).map(|_| hit_probe(&mut state)).collect();
    check(&ctx, &gpu, &probes);
}
