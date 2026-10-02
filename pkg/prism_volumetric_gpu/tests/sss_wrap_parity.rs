//! Real-device parity for the `subsurface`-`wrap` twin:
//! [`GpuSssWrap`](prism_volumetric_gpu::sss_wrap::GpuSssWrap) must reproduce the
//! `CPU` golden
//! [`sss_wrap`](prism_render_architecture::particle::sss_wrap) across the whole
//! [`SssParams::evaluate`](prism_render_architecture::particle::sss_wrap::SssParams::evaluate)
//! sample — the three-channel wrapped diffuse (the scalar
//! [`wrap_ndotl`](prism_render_architecture::particle::sss_wrap::wrap_ndotl) term
//! broadened per channel by
//! [`scatter_color`](prism_render_architecture::particle::sss_wrap::scatter_color)
//! and lifted by the ambient floor) and the scalar back transmission from
//! [`thickness_transmission`](prism_render_architecture::particle::sss_wrap::thickness_transmission).
//!
//! The fixtures cover the regimes the golden calls out: a positive-`wrap` back
//! lit query whose shadow terminator glows warm, a negative-`wrap` forward query
//! whose terminator tightens, a zero-`wrap` query that collapses the diffuse to
//! clamped Lambert plus scatter, a thick-medium query whose back lobe is heavily
//! attenuated, a forward-view query whose back transmission gates to zero, a
//! blue-dominant scatter query, a mixed batch, and two randomized sweeps
//! compared element-for-element. Every `wrap_ndotl` argument (the main term and
//! all three scatter channels) is held clear of the `clamp` endpoints and the
//! back lobe is held clear of its `clamp` endpoints, so no fixture sits on a
//! branch tie; the integer `transmission_power` multiply chain stays bit-exact.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds, divides
//! and one `sqrt` per direction normalize, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) on every
//! `f32` lane of the diffuse and the transmission.
//!
//! # Conditioning
//!
//! Every random fixture is kept well away from the `clamp` cracks: the main
//! `wrap_ndotl` ratio `(n_dot_l + wrap) / (1 + wrap)` and each per-channel ratio
//! `(n_dot_l + wrap + radius) / (1 + wrap + radius)` are held inside
//! `0.05..=0.95`, the scatter `extra = wrapped - core` is held clear of its
//! `max(., 0)` tie, and the back-facing cosine `-dot(view, light)` is held
//! either comfortably below zero (a solid gate to zero transmission) or inside
//! `0.1..=0.9` (an interior lobe), never on the `0` or `1` `clamp` tie. `n_dot_l`
//! itself is held in `0.1..=0.9` in magnitude so neither the Lambert core nor
//! the normalize sits on an endpoint.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::sss_wrap`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::sss_wrap::SssParams;
use prism_volumetric_gpu::sss_wrap::{golden, GpuSssWrap, SssWrapQuery, SssWrapSample};
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

/// Soft-edge guard mirroring the reference `MIN_EDGE`; a squared length below it
/// normalizes to the zero vector instead of dividing by a vanishing length.
const MIN_EDGE: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Returns whether two three-channel values agree lane for lane within the
/// bound.
fn close_vec3(a: [f32; 3], b: [f32; 3]) -> bool {
    close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
}

/// Dot product of two hand-rolled `vec3` values, mirroring the reference `dot3`.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Unit vector along `v`, or the zero vector for a (near-)zero input, mirroring
/// the reference `normalize3`. Uses only one `sqrt`, so no transcendental method
/// is called.
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = dot3(v, v);
    if len_sq < MIN_EDGE {
        return [0.0, 0.0, 0.0];
    }
    let inv_len = 1.0 / len_sq.sqrt();
    [v[0] * inv_len, v[1] * inv_len, v[2] * inv_len]
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

/// A pseudo-random, exactly normalized direction, rejecting a vector too short
/// to normalize cleanly. Uses only `normalize3` (one `sqrt`), so no
/// transcendental method is called.
fn rand_dir(state: &mut u64) -> [f32; 3] {
    loop {
        let v = [signed(state, 1.0), signed(state, 1.0), signed(state, 1.0)];
        if dot3(v, v) > 0.2 {
            return normalize3(v);
        }
    }
}

/// The reference main `wrap_ndotl` ratio before the `clamp`, used for
/// conditioning so a fixture never samples a `clamp` endpoint.
fn wrap_ratio(n_dot_l: f32, wrap: f32) -> f32 {
    (n_dot_l + wrap) / (1.0 + wrap).max(MIN_EDGE)
}

/// Builds a clearly-conditioned query by rejection sampling. `n_dot_l` is held
/// in `0.1..=0.9` in magnitude (covering back-lit and front-lit), the main and
/// per-channel `wrap_ndotl` ratios are held inside `0.05..=0.95`, each scatter
/// `extra = wrapped - core` is held at least `0.03` from its `max(., 0)` tie,
/// and the back-facing cosine is held either below `-0.1` (a solid zero gate) or
/// inside `0.1..=0.9` (an interior lobe).
fn rand_query(state: &mut u64) -> SssWrapQuery {
    loop {
        let normal = rand_dir(state);
        let light = rand_dir(state);
        let view = rand_dir(state);
        let n = normalize3(normal);
        let l = normalize3(light);
        let v = normalize3(view);
        let n_dot_l = dot3(n, l);
        if n_dot_l.abs() < 0.1 || n_dot_l.abs() > 0.9 {
            continue;
        }
        // wrap in [-0.3, 0.8] so the guarded denominator (1 + wrap) stays well
        // above zero and both the hard and the wrapped terminator are exercised.
        let wrap = lcg(state) * 1.1 - 0.3;
        let main = wrap_ratio(n_dot_l, wrap);
        if !(0.05..=0.95).contains(&main) {
            continue;
        }
        // Each scatter radius in [0.05, 0.8]; the lower bound keeps every radius
        // strictly positive so the per-channel max(radius, 0) is never on a tie.
        let scatter = [
            lcg(state) * 0.75 + 0.05,
            lcg(state) * 0.75 + 0.05,
            lcg(state) * 0.75 + 0.05,
        ];
        let core = n_dot_l.clamp(0.0, 1.0);
        let mut channels_ok = true;
        for radius in scatter {
            let ratio = wrap_ratio(n_dot_l, wrap + radius);
            if !(0.05..=0.95).contains(&ratio) {
                channels_ok = false;
            }
            // ratio is interior here, so the clamped wrapped value equals it; the
            // extra beyond the Lambert core must clear its max(., 0) tie.
            if (ratio - core).abs() < 0.03 {
                channels_ok = false;
            }
        }
        if !channels_ok {
            continue;
        }
        let raw_back = -dot3(v, l);
        // Reject the clamp ties: the region around zero and the endpoint at one.
        if raw_back > -0.1 && raw_back < 0.1 {
            continue;
        }
        if raw_back > 0.9 {
            continue;
        }
        let thickness = lcg(state) * 3.8 + 0.2;
        let thickness_scale = lcg(state) * 2.7 + 0.3;
        let ambient = lcg(state) * 0.08;
        let transmission_power = ((lcg(state) * 7.0) as u32).min(6);
        let params = SssParams::new(wrap, scatter, thickness_scale, transmission_power, ambient);
        return SssWrapQuery::new(normal, light, view, thickness, params);
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: the diffuse and
/// the transmission must agree lane for lane within the bound.
fn pin(idx: usize, query: &SssWrapQuery, got: &SssWrapSample) {
    let want = golden(query);
    assert!(
        close_vec3(got.diffuse, want.diffuse),
        "query {idx} diffuse: gpu {:?} vs cpu {:?}",
        got.diffuse,
        want.diffuse
    );
    assert!(
        close(got.transmission, want.transmission),
        "query {idx} transmission: gpu {} vs cpu {}",
        got.transmission,
        want.transmission
    );
}

/// Dispatches `queries` on the `GPU` and pins every result against the
/// reference.
fn check(ctx: &GpuContext, gpu: &GpuSssWrap, queries: &[SssWrapQuery]) {
    let got = gpu.eval(ctx, queries);
    assert_eq!(
        got.len(),
        queries.len(),
        "result count must match the input count"
    );
    for (idx, (query, result)) in queries.iter().zip(got.iter()).enumerate() {
        pin(idx, query, result);
    }
}

/// Builds an exactly normalized direction from raw components.
fn unit(x: f32, y: f32, z: f32) -> [f32; 3] {
    normalize3([x, y, z])
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn positive_wrap_backlit_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    // Light biased behind the surface (n_dot_l about -0.3) with a positive wrap
    // and a red-dominant scatter: the terminator glows warm. The view looks
    // through the medium (interior back lobe).
    let params = SssParams::new(0.5, [0.8, 0.5, 0.3], 2.0, 4, 0.02);
    let query = SssWrapQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(0.0, 0.954, -0.3),
        unit(0.0, 0.0, 1.0),
        0.5,
        params,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn negative_wrap_forward_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    // A negative wrap tightens the terminator; the moderate front-lit angle keeps
    // the main and per-channel ratios interior. The forward view gates the back
    // transmission solidly to zero.
    let params = SssParams::new(-0.2, [0.6, 0.4, 0.15], 1.0, 3, 0.01);
    let query = SssWrapQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(0.0, 0.9165, 0.4),
        unit(0.0, 0.0, 1.0),
        1.0,
        params,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn zero_wrap_is_clamped_lambert_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    // wrap == 0 reduces the main term to clamp(n_dot_l, 0, 1); the scatter still
    // broadens each channel. The view looks through the medium for an interior
    // back lobe.
    let params = SssParams::new(0.0, [0.5, 0.3, 0.15], 1.5, 2, 0.0);
    let query = SssWrapQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ),
        unit(0.0, 0.3, -0.95),
        0.8,
        params,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn thick_medium_attenuates_transmission_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    // A thick medium with a steep attenuation rate keeps the back lobe on but
    // heavily damped; the interior back cosine avoids the clamp ties.
    let params = SssParams::new(0.6, [0.8, 0.6, 0.4], 3.0, 5, 0.05);
    let query = SssWrapQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            -std::f32::consts::FRAC_1_SQRT_2,
        ),
        unit(0.0, 0.0, 1.0),
        4.0,
        params,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn forward_view_gates_transmission_to_zero_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    // View aligned with the light: -dot(view, light) is solidly negative, so the
    // back lobe gates to exactly zero on both devices while the diffuse stays
    // interior.
    let params = SssParams::new(0.3, [0.4, 0.25, 0.1], 1.0, 3, 0.02);
    let query = SssWrapQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(0.0, 0.6, 0.8),
        unit(0.0, 0.6, 0.8),
        0.5,
        params,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn blue_dominant_scatter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    // A blue-dominant scatter radius with a wide wrap on a back-lit point: the
    // shadow terminator cools blue, the opposite of the warm fixture.
    let params = SssParams::new(0.75, [0.2, 0.4, 0.8], 0.8, 2, 0.03);
    let query = SssWrapQuery::new(
        unit(0.0, 0.0, 1.0),
        unit(0.0, 0.5, -0.4),
        unit(0.0, 0.0, 1.0),
        1.5,
        params,
    );
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    // A heterogeneous batch exercises several regimes in one dispatch: warm
    // back-lit, negative-wrap forward, zero-wrap Lambert, thick medium, forward
    // gate and blue-dominant scatter.
    let queries = [
        SssWrapQuery::new(
            unit(0.0, 0.0, 1.0),
            unit(0.0, 0.954, -0.3),
            unit(0.0, 0.0, 1.0),
            0.5,
            SssParams::new(0.5, [0.8, 0.5, 0.3], 2.0, 4, 0.02),
        ),
        SssWrapQuery::new(
            unit(0.0, 0.0, 1.0),
            unit(0.0, 0.9165, 0.4),
            unit(0.0, 0.0, 1.0),
            1.0,
            SssParams::new(-0.2, [0.6, 0.4, 0.15], 1.0, 3, 0.01),
        ),
        SssWrapQuery::new(
            unit(0.0, 0.0, 1.0),
            unit(
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ),
            unit(0.0, 0.3, -0.95),
            0.8,
            SssParams::new(0.0, [0.5, 0.3, 0.15], 1.5, 2, 0.0),
        ),
        SssWrapQuery::new(
            unit(0.0, 0.0, 1.0),
            unit(
                0.0,
                std::f32::consts::FRAC_1_SQRT_2,
                -std::f32::consts::FRAC_1_SQRT_2,
            ),
            unit(0.0, 0.0, 1.0),
            4.0,
            SssParams::new(0.6, [0.8, 0.6, 0.4], 3.0, 5, 0.05),
        ),
        SssWrapQuery::new(
            unit(0.0, 0.0, 1.0),
            unit(0.0, 0.6, 0.8),
            unit(0.0, 0.6, 0.8),
            0.5,
            SssParams::new(0.3, [0.4, 0.25, 0.1], 1.0, 3, 0.02),
        ),
        SssWrapQuery::new(
            unit(0.0, 0.0, 1.0),
            unit(0.0, 0.5, -0.4),
            unit(0.0, 0.0, 1.0),
            1.5,
            SssParams::new(0.75, [0.2, 0.4, 0.8], 0.8, 2, 0.03),
        ),
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn random_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    let mut state = 0x5eed_1234_abcd_0001_u64;
    let queries: Vec<SssWrapQuery> = (0..64).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSssWrap::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins the diffuse and the
    // transmission across many random geometries.
    let queries: Vec<SssWrapQuery> = (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
