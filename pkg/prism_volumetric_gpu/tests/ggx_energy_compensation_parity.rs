//! Real-device parity for the `GGX` energy-compensation twin:
//! [`GpuGgxEnergyCompensation`](prism_volumetric_gpu::ggx_energy_compensation::GpuGgxEnergyCompensation)
//! must reproduce the `CPU` golden
//! [`ggx_energy_compensation`](prism_render_architecture::particle::ggx_energy_compensation)
//! across a smooth near-head-on sample, a rough grazing sample, a mid-roughness
//! coloured-metal sample, a white-furnace sample (where single plus multiple
//! scattering conserves unit energy), a zero-roughness sample that drives the
//! bidirectional lobe into its collapse branch, and a randomized batch compared
//! element-for-element.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and
//! guarded divides, so `CPU` and `GPU` evaluate the same closed form in the same
//! order. They are not bit-exact: a `GPU` may fuse a multiply-add the scalar
//! reference leaves separate, perturbing the low mantissa bits by a few units in
//! the last place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` on every `f32` field.
//!
//! # Conditioning
//!
//! Every random fixture keeps `roughness` comfortably above zero, so the
//! missing-energy factor `1 - E_avg` stays far above the compare epsilon and
//! both devices take the same side of the bidirectional lobe's collapse branch.
//! The `mu`/`roughness`/`f0` inputs are drawn from the open interior of
//! `[0, 1]`, well clear of the `clamp01` boundaries, so no fixture sits on a
//! clamp tie.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::ggx_energy_compensation`；
//! no third-party engine source or derived code.

use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::ggx_energy_compensation::{
    golden, GgxEnergyCompensationQuery, GgxEnergyCompensationResult, GpuGgxEnergyCompensation,
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

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
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

/// Builds a clearly-conditioned query: every cosine, roughness and `F0` is drawn
/// from the open interior of `[0, 1]`, and `roughness` is kept comfortably above
/// zero so the bidirectional lobe never lands on its collapse branch.
fn rand_query(state: &mut u64) -> GgxEnergyCompensationQuery {
    GgxEnergyCompensationQuery {
        mu: uniform(state, 0.1, 0.95),
        mu_out: uniform(state, 0.1, 0.95),
        mu_in: uniform(state, 0.1, 0.95),
        roughness: uniform(state, 0.15, 0.95),
        f0: uniform(state, 0.05, 0.95),
        single_scatter_albedo: uniform(state, 0.05, 0.6),
        e_avg: uniform(state, 0.5, 0.99),
        f_avg: uniform(state, 0.05, 0.99),
        f0_rgb: Vec3::new(
            uniform(state, 0.05, 0.95),
            uniform(state, 0.05, 0.95),
            uniform(state, 0.05, 0.95),
        ),
        single_scatter_albedo_rgb: Vec3::new(
            uniform(state, 0.05, 0.6),
            uniform(state, 0.05, 0.6),
            uniform(state, 0.05, 0.6),
        ),
    }
}

/// Pins one `GPU` result against the `CPU` golden for `query`: every scalar and
/// every `vec3` lane the reference reports must agree within bound.
fn pin(idx: usize, query: &GgxEnergyCompensationQuery, got: &GgxEnergyCompensationResult) {
    let want = golden(query);
    let scalars = [
        (
            "single_scatter_directional_albedo",
            got.single_scatter_directional_albedo,
            want.single_scatter_directional_albedo,
        ),
        ("average_albedo", got.average_albedo, want.average_albedo),
        ("average_fresnel", got.average_fresnel, want.average_fresnel),
        (
            "multiscatter_fresnel_scale",
            got.multiscatter_fresnel_scale,
            want.multiscatter_fresnel_scale,
        ),
        (
            "multiscatter_directional_albedo",
            got.multiscatter_directional_albedo,
            want.multiscatter_directional_albedo,
        ),
        (
            "kulla_conty_multiscatter_brdf",
            got.kulla_conty_multiscatter_brdf,
            want.kulla_conty_multiscatter_brdf,
        ),
        (
            "compensated_specular_albedo",
            got.compensated_specular_albedo,
            want.compensated_specular_albedo,
        ),
        (
            "average_fresnel_rgb.x",
            got.average_fresnel_rgb.x,
            want.average_fresnel_rgb.x,
        ),
        (
            "average_fresnel_rgb.y",
            got.average_fresnel_rgb.y,
            want.average_fresnel_rgb.y,
        ),
        (
            "average_fresnel_rgb.z",
            got.average_fresnel_rgb.z,
            want.average_fresnel_rgb.z,
        ),
        (
            "compensated_specular_albedo_rgb.x",
            got.compensated_specular_albedo_rgb.x,
            want.compensated_specular_albedo_rgb.x,
        ),
        (
            "compensated_specular_albedo_rgb.y",
            got.compensated_specular_albedo_rgb.y,
            want.compensated_specular_albedo_rgb.y,
        ),
        (
            "compensated_specular_albedo_rgb.z",
            got.compensated_specular_albedo_rgb.z,
            want.compensated_specular_albedo_rgb.z,
        ),
    ];
    for (name, gpu, cpu) in scalars {
        assert!(
            close(gpu, cpu),
            "query {idx} {name}: gpu {gpu} vs cpu {cpu}"
        );
    }
}

/// Dispatches `queries`, asserts the result count matches, and pins every result
/// against the `CPU` golden.
fn check(ctx: &GpuContext, gpu: &GpuGgxEnergyCompensation, queries: &[GgxEnergyCompensationQuery]) {
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

/// A white-furnace query (`F0 = 1`): single plus multiple scattering restores
/// unit energy, the hardest conservation case to port.
fn white_furnace_query() -> GgxEnergyCompensationQuery {
    GgxEnergyCompensationQuery {
        mu: 0.3,
        mu_out: 0.6,
        mu_in: 0.4,
        roughness: 0.8,
        f0: 1.0,
        single_scatter_albedo: 0.5,
        e_avg: 0.9,
        f_avg: 1.0,
        f0_rgb: Vec3::new(1.0, 1.0, 1.0),
        single_scatter_albedo_rgb: Vec3::new(0.5, 0.5, 0.5),
    }
}

#[test]
fn empty_input_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxEnergyCompensation::new(&ctx);
    // An empty batch short-circuits before any dispatch (a storage buffer cannot
    // be zero-sized) and returns an empty vector.
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch produces no results");
}

#[test]
fn smooth_head_on_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxEnergyCompensation::new(&ctx);
    // A nearly head-on view of a smooth dielectric: little energy is lost, so the
    // compensation term is small but non-zero.
    let query = GgxEnergyCompensationQuery {
        mu: 0.9,
        mu_out: 0.85,
        mu_in: 0.8,
        roughness: 0.25,
        f0: 0.04,
        single_scatter_albedo: 0.3,
        e_avg: 0.95,
        f_avg: 0.1,
        f0_rgb: Vec3::new(0.04, 0.05, 0.06),
        single_scatter_albedo_rgb: Vec3::new(0.25, 0.3, 0.35),
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn rough_grazing_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxEnergyCompensation::new(&ctx);
    // A grazing view of a rough metal: single scattering loses the most energy,
    // so the compensation term is largest here.
    let query = GgxEnergyCompensationQuery {
        mu: 0.15,
        mu_out: 0.2,
        mu_in: 0.18,
        roughness: 0.9,
        f0: 0.9,
        single_scatter_albedo: 0.2,
        e_avg: 0.6,
        f_avg: 0.9,
        f0_rgb: Vec3::new(0.9, 0.85, 0.8),
        single_scatter_albedo_rgb: Vec3::new(0.2, 0.22, 0.25),
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn coloured_metal_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxEnergyCompensation::new(&ctx);
    // A mid-roughness coloured metal exercises the per-channel RGB paths with
    // three distinct F0 lanes.
    let query = GgxEnergyCompensationQuery {
        mu: 0.55,
        mu_out: 0.6,
        mu_in: 0.5,
        roughness: 0.6,
        f0: 0.5,
        single_scatter_albedo: 0.35,
        e_avg: 0.8,
        f_avg: 0.5,
        f0_rgb: Vec3::new(0.95, 0.64, 0.54),
        single_scatter_albedo_rgb: Vec3::new(0.4, 0.3, 0.25),
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn white_furnace_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxEnergyCompensation::new(&ctx);
    check(&ctx, &gpu, &[white_furnace_query()]);
}

#[test]
fn zero_roughness_collapse_branch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxEnergyCompensation::new(&ctx);
    // At roughness 0 the missing-energy factor 1 - E_avg is exactly zero, so both
    // devices take the bidirectional lobe's collapse branch and return 0; the
    // compensation also vanishes while E_ss is 1.
    let query = GgxEnergyCompensationQuery {
        mu: 0.5,
        mu_out: 0.5,
        mu_in: 0.5,
        roughness: 0.0,
        f0: 0.5,
        single_scatter_albedo: 0.3,
        e_avg: 0.9,
        f_avg: 0.5,
        f0_rgb: Vec3::new(0.3, 0.5, 0.7),
        single_scatter_albedo_rgb: Vec3::new(0.3, 0.3, 0.3),
    };
    check(&ctx, &gpu, &[query]);
}

#[test]
fn mixed_batch_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxEnergyCompensation::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    // One batch mixing deterministic fixtures with many random queries,
    // dispatched together so the per-thread indexing and the contiguous storage
    // layout are both exercised, then pinned element-for-element.
    let mut queries = vec![white_furnace_query()];
    for _ in 0..48 {
        queries.push(rand_query(&mut state));
    }
    check(&ctx, &gpu, &queries);
}

#[test]
fn many_queries_sweep_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGgxEnergyCompensation::new(&ctx);
    let mut state = 0x0bad_c0de_cafe_f00d_u64;
    // A larger sweep (several workgroups' worth) pins every output across many
    // random inputs.
    let queries: Vec<GgxEnergyCompensationQuery> =
        (0..200).map(|_| rand_query(&mut state)).collect();
    check(&ctx, &gpu, &queries);
}
